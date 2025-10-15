import collections
import dataclasses
import itertools
import json
import logging
import pathlib
from abc import ABC
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    OrderedDict,
    Protocol,
    runtime_checkable,
    Sequence,
    TypeVar,
)

import torch
import typing_extensions
from more_itertools.more import first
from torch import Tensor
from typing_extensions import Self

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD = True
except ImportError as ie:
    TENSORBOARD = False
    SummaryWriter = None  # type: ignore

InitializeType = TypeVar("InitializeType")


def merge_update(base: dict, update_base: dict):
    output = dict()
    keys = set(base.keys()) | set(update_base.keys())
    for k in keys:
        v = base.get(k, None)
        update_v = update_base.get(k, None)
        if k not in base and update_v is None:
            continue

        is_special = (
            isinstance(update_v, dict)
            and "$$update" in update_v.keys()
            and len(update_v.keys()) == 1
        )
        if is_special:
            if v is None:
                raise ValueError(
                    f"Expected a pre-initialized dict or list for {k!r} to update"
                )
            elif not isinstance(v, (MutableSequence, MutableMapping)):
                raise ValueError(
                    f"Can only update dicts and lists, got {type(v)!r} aka {v!r}"
                )

            updates = update_v["$$update"]
            for update in updates:
                if not isinstance(update, Mapping):
                    raise ValueError(
                        f"Update must be a mapping, got a {update!r} ({type(update)!r})"
                    )

                action = first(update.keys(), None)
                if action is None:
                    raise ValueError(
                        f"Invalid update document, got {update!r}, expected single key mapping"
                    )

                values = update[action]
                if not isinstance(values, Sequence):
                    raise ValueError(
                        f"Values for {action!r} must be a sequence, got {values!r}"
                    )

                if isinstance(v, MutableSequence):
                    if action == "$add":
                        v.extend(values)
                    elif action == "$remove":
                        for element in values:
                            v.remove(element)
                    else:
                        raise ValueError(action)
                else:
                    # v is a MutableMapping
                    for value in values:
                        if not isinstance(value, Mapping):
                            raise ValueError(
                                f"Expected a mapping for values of {action!r}, got {value!r}"
                            )

                        if action == "$add":
                            for kk, kv in value.items():
                                v[kk] = kv
                        elif action == "$remove":
                            for (
                                kk,
                                kv,
                            ) in value.items():
                                if kv is not None:
                                    # remove only if value matches
                                    if v[kk] == kv:
                                        v.pop(kk)
                                else:
                                    # remove in any case
                                    v.pop(kk)
                        else:
                            raise ValueError(action)
            output[k] = v
        elif isinstance(v, dict):
            assert update_v is None or isinstance(update_v, dict)
            update_v = update_v or dict()
            output[k] = merge_update(v, update_v)
        else:
            output[k] = update_v if update_v is not None else v
    return output


def find_config_path(name_or_path: str | Path) -> Path:
    if isinstance(name_or_path, Path) and name_or_path.exists():
        return name_or_path
    elif isinstance(name_or_path, Path):
        name_or_path = name_or_path.absolute().as_posix()
    if not name_or_path.endswith(".json"):
        name_or_path = name_or_path + ".json"

    path = Path(name_or_path)
    local_path = Path().cwd() / "cfg" / name_or_path

    if path.exists():
        return path
    elif local_path.exists():
        return local_path
    else:
        raise ValueError(
            f"Cannot find config '{name_or_path}' as a path or as a local config ({local_path.as_posix()})"
        )


def load_config_from_name_or_path(name_or_path: str | Path, recursive: bool = True):
    config_path = find_config_path(name_or_path)
    with config_path.open() as f:
        config = json.load(f)

    if "base" in config and recursive:
        baseconfig = load_config_from_name_or_path(config["base"], recursive)
        config = merge_update(baseconfig, config)
    return config


@runtime_checkable
class ConfigProtocol(Protocol):
    section_name: ClassVar[Optional[str]]
    include_filepath_in_init: ClassVar[bool]


def initialize_type_from_config(
    cls: type[InitializeType],
    name_or_path: str | pathlib.Path,
    /,
    section_name=None,
    setup=False,
    setup_function_name=None,
    overrides=None,
    **init_kwargs,
) -> InitializeType:
    config = load_config_from_name_or_path(name_or_path)
    include_filepath_in_init = None
    if isinstance(cls, ConfigProtocol):
        if section_name is None and cls.section_name is not None:
            section_name = cls.section_name
        if include_filepath_in_init is None and cls.include_filepath_in_init:
            include_filepath_in_init = True

    kwargs = config[section_name]
    if include_filepath_in_init:
        kwargs["file_path"] = str(name_or_path)

    # apply overrides specified via config files ("mixins")
    if overrides is not None:
        for override in overrides:
            override_config = load_config_from_name_or_path(override)
            if section_name in override_config:
                kwargs = merge_update(kwargs, override_config[section_name])

    # apply overrides specified via kwargs
    kwargs = merge_update(kwargs, init_kwargs)

    # create a new config / whatever type class from the kwargs!
    conf = cls(**kwargs)

    if setup and setup_function_name is not None:
        getattr(conf, setup_function_name)()
    return conf


def pad_to_length(inp: str) -> str:
    return inp.rjust(12)


def format_tensor(values: torch.Tensor):
    fmt = "%.6f"
    if not values.dtype.is_floating_point:
        fmt = "%d"
    return "[" + ", ".join([pad_to_length(fmt % (elem,)) for elem in values]) + "]"


class NameableConfig(ABC):
    section_name: ClassVar[Optional[str]] = None
    include_filepath_in_init: ClassVar[bool] = False

    @classmethod
    def from_name(
        cls: type[InitializeType],
        name_or_path: str | pathlib.Path,
        /,
        section_name=None,
        setup=False,
        setup_function_name=None,
        overrides=None,
        **init_kwargs,
    ) -> InitializeType:
        return initialize_type_from_config(
            cls,
            name_or_path,
            section_name=section_name,
            setup=setup,
            setup_function_name=setup_function_name,
            overrides=overrides,
            **init_kwargs,
        )

    @classmethod
    def load_from_name(cls, name: str):
        assert cls.section_name is not None
        config_path = cls.find_config(name)
        with config_path.open() as f:
            config = json.load(f)
        if "base" in config:
            baseconfig = cls.load_from_name(config["base"])
            config = merge_update(baseconfig, config)
        return config


@dataclasses.dataclass(kw_only=True)
class BasicHparams(NameableConfig):
    section_name = "training"
    include_filepath_in_init: ClassVar[bool] = False

    optimize_for: str = "accuracy"
    patience: int = 0
    batch_size: int = 2
    gradient_accumulation: int = 1

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)


class BasicMetrics:
    ignore_in_repr: ClassVar[set[str] | None] = None
    include_in_repr: ClassVar[set[str] | None] = None

    def __init__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, inp: Mapping[str, Any]) -> typing_extensions.Self:
        return cls(**inp)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        inner = list()
        for key, value in self.__dict__.items():
            if key in (self.__class__.ignore_in_repr or set()):
                continue
            if (
                self.__class__.include_in_repr
                and key not in self.__class__.include_in_repr
            ):
                continue
            if value is None:
                continue
            elif isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    dims = value.dim()
                    if dims > 1:
                        inner_inner = list()
                        inner_inner.append(f"{key}={'[' * (dims - 1)}\n")
                        for indices in itertools.product(
                            *[range(value.size(dim)) for dim in range(dims - 1)]
                        ):
                            inner_inner.append(f"\t{format_tensor(value[indices])},\n")
                        inner_inner.append("]" * (dims - 1))
                        inner.append("".join(inner_inner))
                    else:
                        inner.append(f"{key}={format_tensor(value)}")
                    continue
                value = value.item()
            elif not isinstance(value, float):
                inner.append(f"{key}={value}")
                continue
            inner.append(f"{key}={value:.6f}")
        return f"{{{', '.join(inner)}}}"

    def __str__(self):
        return self.__repr__()

    def __invert__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, (float, torch.Tensor)):
                if "loss" in key:
                    self.__dict__[key] = -self.__dict__[key]
                else:  # metrics (f1 etc)
                    self.__dict__[key] = 1 - self.__dict__[key]

        return self


M = TypeVar("M", bound=BasicMetrics)


class MetricsDict(Generic[M], BasicMetrics):
    metrics: "OrderedDict[str, M | MetricsDict[M]]"

    def __init__(self, primary: Optional[str] = None, **kwargs: "M | MetricsDict[M]"):
        super().__init__()
        self.primary_metrics = primary
        self.metrics = collections.OrderedDict(kwargs)

    def __getitem__(self, item) -> "M | MetricsDict[M]":
        return self.metrics[item]

    def __contains__(self, item: str):
        return item in self.metrics

    def first(self):
        for _, value in self.metrics.items():
            return value
        raise ValueError(f"Empty {self}")

    def __getattr__(self, item):
        if hasattr(self.first(), item):
            return {key: getattr(value, item) for key, value in self.metrics.items()}
        return super().__getattribute__(item)  # todo check getattr or getattribute

    def to_dict(self):
        dicts = {k: metric.to_dict() for k, metric in self.metrics.items()}
        target = dict()
        for elem_key, elem in dicts.items():
            for k, v in elem.items():
                target[elem_key + "__" + k] = v
        return target

    def __repr__(self):
        inner = list()
        for key, value in self.metrics.items():
            if key in (self.__class__.ignore_in_repr or set()):
                continue

            value_repr = repr(value)
            value_repr = prepend_newlines_with_spacing(value_repr, "  ")
            inner.append(f"  {key}={value_repr}")
        breaker = ",\n"
        return f"{{\n{breaker.join(inner)}\n}}"


def prepend_newlines_with_spacing(s: str, spacing: str) -> str:
    s = s.replace("\n", "\n" + spacing)
    return s


def _filter_hparams(hparams: dict[str, Any]):
    logger = logging.getLogger("_filter_hparams")

    def _instance_check(k: str, x: Any) -> bool:
        if isinstance(x, (int, float, str, bool, torch.Tensor)):
            return True
        logger.warning(
            f"Cannot log hyperparameter {k} ({x}), must be int, float, str, bool or torch.Tensor, got {type(x)}"
        )
        return False

    return {key: value for key, value in hparams.items() if _instance_check(key, value)}


class TensorboardLogger:
    _writer: Optional[SummaryWriter] = None
    hparams: dict

    def __init__(
        self,
        log_dir: str,
        hparams: Optional[dict[str, Any]] = None,
        disable: bool = False,
    ):
        self.noop = not TENSORBOARD or disable
        if not self.noop:
            self._writer = SummaryWriter(log_dir)
        self.hparams = hparams or dict()

    def add_hparams(self, hparams: Mapping[str, Any]):
        self.hparams.update(hparams)

    @property
    def writer(self):
        if self._writer is None:
            raise ValueError(f"{self.noop=}")
        return self._writer

    def log_metric(
        self,
        name: str,
        scalar_value: int | float | torch.Tensor,
        global_step=None,
    ):
        if self.noop:
            return

        if isinstance(scalar_value, (int, float)) or (
            isinstance(scalar_value, torch.Tensor) and scalar_value.numel() == 1
        ):
            self.writer.add_scalar(
                name,
                scalar_value,
                global_step,
            )
        else:
            if isinstance(scalar_value, tuple) and all(
                isinstance(elem, torch.Tensor) for elem in scalar_value
            ):
                scalar_value = torch.tensor(scalar_value)

            if not isinstance(scalar_value, torch.Tensor):
                raise ValueError(
                    f"scalar_value '{name}' must be a torch.Tensor, "
                    f"but is {type(scalar_value)}: {scalar_value}"
                )
            self.writer.add_tensor(
                name,
                scalar_value,
                global_step,
            )

    def log_metrics(
        self,
        metrics: Mapping[str, int | float | torch.Tensor],
        global_step=None,
    ):
        for name, scalar_value in metrics.items():
            self.log_metric(name, scalar_value, global_step)

    def log_test_metrics(
        self,
        hparams: dict[str, int | float | bool | str],
        metrics: dict[str, int | float | torch.Tensor],
    ):
        self.add_hparams(hparams)

        if self.noop:
            return

        metrics = {
            key: value
            for key, value in metrics.items()
            if (
                (not isinstance(value, torch.Tensor) or value.numel() == 1)
                and isinstance(value, (bool, int, float, torch.Tensor))
            )
        }

        self._add_hparams(
            self.hparams,
            metrics,
            global_step=0,
        )
        self.writer.add_custom_scalars(
            {
                "RocketLeague": {
                    "ballchase": ["Multiline", ["precision", "recall", "f1"]]
                }
            }
        )

    def _add_hparams(
        self,
        hparam_dict: dict,
        metric_dict: dict,
        global_step: Optional[int] = None,
    ):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = torch.utils.tensorboard.writer.hparams(
            _filter_hparams(hparam_dict), metric_dict, None
        )

        self.writer.file_writer.add_summary(exp, global_step)
        self.writer.file_writer.add_summary(ssi, global_step)
        self.writer.file_writer.add_summary(sei, global_step)
        for k, v in metric_dict.items():
            self.writer.add_scalar(k, v, global_step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.close()


def _to_device(
    inp: Any, device: torch.device | str | int, non_blocking: Optional[bool] = None
):
    if isinstance(inp, Tensor):
        copy_non_blocking = non_blocking
        if copy_non_blocking is None:
            copy_non_blocking = inp.is_pinned()
        return inp.to(device, non_blocking=copy_non_blocking)
    elif isinstance(inp, (list, set, tuple)):
        return type(inp)(_to_device(val, device, non_blocking) for val in inp)
    return inp


class Batch(dict[str, Tensor]):
    def to(self, device, non_blocking: Optional[bool] = None) -> Self:
        for key, tensor in self.items():
            self[key] = _to_device(tensor, device, non_blocking)
        return self

    def with_prefix(self, prefix: str):
        out = self.__class__()
        for k, v in self.items():
            out[prefix + k] = v
        return out

    def rename(self, key: str, new_key: str):
        copy = Batch(self)
        value = copy.pop(key)
        copy[new_key] = value
        return copy

    def __copy__(self):
        return Batch(self)

    def __setattr__(self, key: str, value: Tensor):
        self[key] = value

    def __getattr__(self, item: str) -> Tensor:
        if item in {"__getstate__", "__setstate__"}:
            return super().__getattribute__(item)
        return self[item]
