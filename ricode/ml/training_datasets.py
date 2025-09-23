import dataclasses
import functools
import inspect
import json
import logging
import os.path
import warnings
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sized,
    TypeAlias,
    Union,
)

import torch
from more_itertools import first
from safetensors_dataset import load_safetensors, SafetensorsDataset, SafetensorsDict
from safetensors_dataset.dict_dataset import ShardedSafetensorsDataset
from safetensors_dataset.loading import exists_safetensors
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PretrainedConfig

from ricode.ml.training_basics import NameableConfig
from ricode.ml.training_fsdp import rank_zero_first
from ricode.ml.training_types import TDataset, THparams, TrainingArgs
from ricode.ml.training_utils import build_chunks

try:
    import datasets.load as load

    DATASETS = True
except ImportError:
    load = None
    DATASETS = False

Data: TypeAlias = Union[
    SafetensorsDataset, MutableMapping[str, SafetensorsDataset], SafetensorsDict
]
DataDict: TypeAlias = MutableMapping[str, Data]


@dataclasses.dataclass()
class SkipExample:
    cause: str

    def __bool__(self):
        return False


@dataclasses.dataclass(kw_only=True)
class SplitInfo:
    name: str | None = None
    name_or_file: str
    extension: str = ".safetensors"
    types: str | None = None

    @property
    def path(self):
        return Path(self.name_or_file)


ReturnData: TypeAlias = Union[
    Mapping[str, Tensor], list[Mapping[str, Tensor]], Literal[False], SkipExample
]
SetupElementFunc: TypeAlias = Callable[[str, SplitInfo, Any], ReturnData]
SetupSplitFunc: TypeAlias = Callable[
    [str, SplitInfo, Iterable[Any]], SafetensorsDataset | SafetensorsDict
]


def append_to_map(
    output: MutableMapping[str, list[Tensor]], mapping: Mapping[str, Tensor]
):
    have_grace = not any(len(elem) for elem in output.values())
    for k, v in mapping.items():
        if k not in output:
            if not have_grace:
                raise ValueError(
                    f"Cannot append additional key {k} after keys {output.keys()} were already filled with values"
                )
            output[k] = list()
        output[k].append(v)


def _multiprocessing_argument_unpack(func, split, args):
    return [func(split, element) for element in args]


def _guess_split_name(value: str):
    if "." in value:
        value = value.split(".")[0]
        if "_" in value:
            return value.split("_")[0]
        return value
    return value


def _resolve_split_info(value: Any):  # value is str or dict[str, str]
    if isinstance(value, str):
        return SplitInfo(name=_guess_split_name(value), name_or_file=value)
    elif not isinstance(value, dict):
        raise ValueError(value)

    try:
        return SplitInfo(**value)
    except TypeError:
        return {key: _resolve_split_info(inner) for key, inner in value.items()}


@dataclasses.dataclass(kw_only=True)
class BasicDataset(NameableConfig):
    with_filepath: ClassVar[bool] = True
    storage_prefix: ClassVar[str] = ""
    precomputed_pair_flags: ClassVar[bool] = False
    split_names: ClassVar[set[str]] = {"train", "test", "eval"}
    use_multiprocessing: ClassVar[bool] = False
    use_shards: ClassVar[bool] = True
    shard_size: ClassVar[int] = 1000

    name: str
    data_dir: str
    splits: Mapping[str, SplitInfo | dict[str, SplitInfo]]

    data: DataDict = dataclasses.field(default_factory=OrderedDict)
    is_hpo: bool = False
    max_size: int = 0
    force_setup: Optional[bool] = None
    memory_only: Optional[bool] = None
    multiprocessing_num_proc: int = -1
    multiprocessing_chunk_size: int = int(
        os.getenv("multiprocessing_chunk_size".upper(), "32")
    )
    file_path: Optional[str] = None
    _storage_prefix: Optional[str] = None

    def __post_init__(self):
        setattr(
            self,
            "splits",
            {key: _resolve_split_info(value) for key, value in self.splits.items()},
        )
        setattr(self, "setup_examples", self.guess_setup_func_by_extension())

    @property
    def data_path(self) -> Path:
        if "/" not in self.data_dir:
            return Path(os.path.join("datasets", self.data_dir))
        return Path(self.data_dir)

    def __getitem__(self, item: str) -> SafetensorsDataset:
        return self.data[item]

    def to(self, device: str | int | torch.device):
        def move_to_device(d: DataDict):
            for k, v in d.items():
                if isinstance(v, (SafetensorsDataset, SafetensorsDict)):
                    d[k] = v.to(device)
                else:
                    move_to_device(v)

        move_to_device(self.data)

    def setup_examples(
        self, split: str, split_info: SplitInfo, data: Any
    ) -> SafetensorsDataset:
        raise NotImplementedError(self.__class__.__name__ + ".setup_examples")

    def json_setup_example(
        self, split: str, split_info: SplitInfo, json_item: Any
    ) -> ReturnData:
        raise NotImplementedError(self.__class__.__name__ + ".json_setup_example")

    def guess_setup_func_by_extension(self):
        extensions = set()
        for filepath_or_dict_of_filepaths in self.splits.values():
            if isinstance(filepath_or_dict_of_filepaths, dict):
                for split_info in filepath_or_dict_of_filepaths.values():
                    extensions.add(split_info.path.suffix)
            else:
                extensions.add(filepath_or_dict_of_filepaths.path.suffix)
        if len(extensions & {".json", ".jsonl"}) > 0:
            return self.json_list_setup_examples
        return self.setup_examples

    def list_splits(self, logger: logging.Logger):
        def format_split(dataset):
            return repr(dataset).replace("\n", "\n  ")

        def format_keys(keys):
            return ", ".join(sorted(map(lambda k: '"' + k + '"', keys)))

        def format_dataset(key, dataset):
            if isinstance(dataset, SafetensorsDataset):
                return (
                    f"  {key}=SafetensorsDataset("
                    f"size={len(dataset)}, "
                    f"keys={format_keys(dataset.keys())})"
                )
            elif isinstance(dataset, ShardedSafetensorsDataset):
                return (
                    f"  {key}=ShardedSafetensorsDataset("
                    f"size={len(dataset)}, "
                    f"num_shards={len(dataset.shards)}, "
                    f"keys={format_keys(dataset.shards[0].keys())}"
                    f")"
                )
            elif isinstance(dataset, dict):
                if len(dataset) <= 3:
                    formatted_dataset = [
                        "  " + format_dataset(key, dataset[key])
                        for key in sorted(dataset.keys())
                    ]
                    formatted_dataset = "\n".join(formatted_dataset)
                    return (
                        f"  {key}=SafetensorsDict(" f"splits={{\n{formatted_dataset}}})"
                    )
                else:
                    first_key = first(sorted(dataset.keys()))
                    return (
                        f"  {key}=SafetensorsDict("
                        f"num_splits={len(dataset)}, "
                        f"splits={{{format_dataset(first_key, dataset[first_key])[2:]}}}, ... and {len(dataset) - 1} more)"
                    )
            else:
                return f"  {key}={dataset}"

        info = [f"Loaded dataset {self.__class__.__name__}("]
        for split in self.data.keys():
            if split not in self.split_names:
                continue
            info.append(format_dataset(split, self.data[split]))
        logger.info("\n".join(info))

    def inner_setup_examples(
        self,
        split: str,
        split_info: SplitInfo,
        elements: Iterable[Any],
        element_fn: SetupElementFunc,
        **kwargs: Any,
    ) -> SafetensorsDataset:
        output: dict[str, list[Tensor]] = dict()

        if kwargs:
            element_fn = functools.partial(element_fn, **kwargs)

        infos = OrderedDict()
        total = None if self.max_size == 0 else self.max_size
        for pos, item in enumerate(
            tq := tqdm(elements, desc=split_info.name + " " + split, total=total)
        ):
            if pos >= self.max_size > 0:
                break

            example = element_fn(split, split_info, item)
            if not example or isinstance(example, SkipExample):
                cause = "skipped"
                if isinstance(example, SkipExample):
                    cause = example.cause
                if cause not in infos:
                    infos[cause] = 1
                else:
                    infos[cause] += 1
                tq.set_postfix(infos)
                continue

            if isinstance(example, list) or isinstance(example, tuple):
                for example_elem in example:
                    append_to_map(output, example_elem)
            elif isinstance(example, dict):
                append_to_map(output, example)
            else:
                raise ValueError(type(example))

        def map_without_empty(m: dict):
            return {k: v for k, v in m.items() if len(v) > 0}

        output = map_without_empty(output)
        return SafetensorsDataset(output, preprocess=True)

    def multiprocessing_inner_setup_examples(
        self,
        split: str,
        split_info: SplitInfo,
        elements: Iterable[Any],
        element_fn: SetupElementFunc,
        **kwargs: Any,
    ) -> SafetensorsDataset:
        output: dict[str, list[Tensor]] = dict()

        if kwargs:
            element_fn = functools.partial(element_fn, **kwargs)

        infos = OrderedDict(skipped=0)
        total = None if self.max_size == 0 else self.max_size

        def try_len(a: Iterable[Any]) -> int | None:
            if isinstance(a, Sized):
                return total or len(a)
            return None

        def imap_generator(iterable: Iterable[Any]):
            for item in build_chunks(iterable, self.multiprocessing_chunk_size):
                yield item

        n_proc = self.multiprocessing_num_proc or cpu_count()
        with (
            logging_redirect_tqdm(),
            Pool(
                processes=n_proc,
            ) as pool,
            tqdm(
                desc=f"{split_info.name + ' ' + split} ({n_proc} processes)",
                total=try_len(elements),
            ) as tq,
        ):
            for pos, examples in enumerate(
                pool.imap(
                    functools.partial(
                        _multiprocessing_argument_unpack,
                        element_fn,
                        split,
                        split_info,
                    ),
                    imap_generator(elements),
                    chunksize=1,
                ),
            ):
                for example in examples:
                    tq.update(1)
                    if pos >= self.max_size > 0:
                        warnings.warn("Stopping multiprocessing tokenization ...")
                        break

                    if not example:
                        infos["skipped"] += 1
                        tq.set_postfix(infos)
                        continue

                    if isinstance(example, list) or isinstance(example, tuple):
                        for example_elem in example:
                            append_to_map(output, example_elem)
                    elif isinstance(example, dict):
                        append_to_map(output, example)
                    else:
                        raise ValueError(type(example))

        def map_without_empty(m: dict):
            return {k: v for k, v in m.items() if len(v) > 0}

        output = map_without_empty(output)
        return SafetensorsDataset(output, preprocess=True)

    def json_list_setup_examples(
        self, split: str, split_info: SplitInfo, json: list[Any]
    ) -> SafetensorsDataset:
        inner_fn = self.inner_setup_examples
        if self.use_multiprocessing and self.multiprocessing_num_proc >= 0:
            inner_fn = self.multiprocessing_inner_setup_examples
        return inner_fn(split, split_info, json, self.json_setup_example)

    @rank_zero_first
    def setup_dataset(self):
        for split in sorted(self.splits.keys()):
            if split not in self.split_names:
                continue

            split_info = self.splits[split]
            if isinstance(split_info, dict):
                self._setup_dict_split(split, split_info)
            else:
                self._setup_split(split, split_info)

    def setup_hpo(self):
        eval_spit = self.data.get("dev", self.data.get("eval", None))
        if eval_spit is None:
            raise ValueError(
                f"Cannot setup hpo, unknown name for eval/dev split: {self.data}"
            )
        self.data["test"] = eval_spit
        self.is_hpo = True

    def is_hpo_dataset(self):
        return self.is_hpo

    def _load_split_data(self, split: str, split_info: SplitInfo):
        file_path = self._split_path(split, split_info)
        if file_path.suffix == ".json":
            with open(self._split_path(split, split_info)) as f:
                return json.load(f)
        elif file_path.suffix == ".jsonl":

            def _iterator():
                with open(file_path) as f:
                    while line := f.readline():
                        yield json.loads(line)

            return _iterator()
        raise ValueError(
            f"Unknown file extension {file_path.suffix} for {file_path.as_posix()}"
        )

    def _setup_dict_split(self, split: str, split_infos: dict[str, SplitInfo]):
        datasets = OrderedDict()
        for key in sorted(split_infos.keys()):
            datasets[key] = self._setup_split_base(split, split_infos[key])
        self.data[split] = datasets

    def _setup_split_base(self, split: str, split_info: SplitInfo):

        if split_info.extension == ".safetensors":

            def exists_dataset(info: SplitInfo):
                return exists_safetensors(self._split_path(split, info, is_final=True))

        elif split_info.extension == ".datasets":

            def exists_dataset(info: SplitInfo):
                return os.path.exists(self._split_path(split, info, is_final=True))

        else:
            raise ValueError(split_info.extension)

        if self.force_setup or not exists_dataset(split_info):
            loaded_data = self._load_split_data(split, split_info)
            dataset = self._call_setup_examples(
                self.setup_examples,
                split,
                split_info,
                split_info.name_or_file,
                loaded_data,
            )
            if (
                self.use_shards
                and len(dataset) > self.shard_size
                and isinstance(dataset, (SafetensorsDataset, SafetensorsDict))
            ):
                dataset = dataset.shard(self.shard_size, preprocess_if_unprocessed=True)
            if not self.memory_only:
                self._save_split0(split, split_info, dataset)
                dataset = self._load_split0(split, split_info)
            return dataset
        else:
            return self._load_split0(split, split_info)

    @staticmethod
    def _call_setup_examples(
        func: SetupSplitFunc,
        split: str,
        split_info: SplitInfo,
        path: str,
        data: Any,
    ) -> SafetensorsDataset:
        func_args = (split, split_info, data)
        func_kwargs = {}
        spec = inspect.getfullargspec(func)
        if "path" in spec.args or "path" in spec.kwonlyargs:
            func_kwargs["path"] = path
        return func(*func_args, **func_kwargs)

    def _setup_split(self, split: str, split_info: SplitInfo):
        self.data[split] = self._setup_split_base(split, split_info)

    def _save_split0(self, split: str, split_info: SplitInfo, dataset: Any):
        split_path = self._split_path(split, split_info, is_final=True)
        if split_info.extension == ".safetensors":
            if not isinstance(
                dataset,
                (SafetensorsDataset, SafetensorsDict, ShardedSafetensorsDataset),
            ):
                raise ValueError
            dataset.save_to_file(split_path)
        elif split_info.extension == ".datasets":
            if DATASETS:
                dataset.save_to_disk(split_path)
            else:
                raise ImportError("datasets is unavailable")

    def _load_split0(self, split: str, split_info: SplitInfo):
        if self.memory_only:
            raise ValueError(f"Cannot load split {split} when {self.memory_only=}")
        if split_info.extension == ".safetensors":
            dataset = load_safetensors(
                self._split_path(split, split_info, is_final=True)
            )
        elif split_info.extension == ".datasets":
            from datasets.load import load_from_disk

            dataset = load_from_disk(self._split_path(split, split_info, is_final=True))
        else:
            raise ValueError(split_info.extension)
        return dataset

    def _split_path(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> Path:
        data_directory = self.data_path
        if not data_directory.exists():
            data_directory.mkdir(parents=True, exist_ok=False)
        return data_directory / self._split_filename(
            split, split_info, is_final=is_final
        )

    def _split_filename_additional_args(self, split: str, path: str) -> list[str]:
        return list()

    def _split_filename(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> str:
        if is_final:
            file_path = split_info.path
            filename = file_path.name[: -len(file_path.suffix)]

            if hasattr(self, "features") and getattr(self, "features") != 0:
                filename = filename + "_feat_" + self._feature_components()
            if hasattr(self, "tokenizer_config") and isinstance(
                self.tokenizer_config, PretrainedConfig
            ):
                tokenizer_name = self.tokenizer_config.model_type
                if tokenizer_name != "t5":
                    filename = filename + "_tk_" + tokenizer_name
            for arg in self._split_filename_additional_args(
                split, split_info.name_or_file
            ):
                filename = filename + arg
            if self._storage_prefix:
                filename = self._storage_prefix + filename
            elif self.storage_prefix:
                filename = self.storage_prefix + filename
            if split_info.extension == ".safetensors":
                filename = filename + split_info.extension
            return filename
        return split_info.name_or_file

    def _feature_components(self) -> str:
        features = getattr(self, "features")
        if isinstance(features, int):
            i = 0
            flags = []
            while 2**i <= features:
                if features & (1 << i):
                    flags.append(str(i))
                i += 1
            return ",".join(flags)
        elif isinstance(features, list):
            return ",".join(list(map(str, features)))
        else:
            raise ValueError(features)


class ProxyTrainingArgs:
    def __init__(self, args: TrainingArgs[THparams, TDataset], dataset_key: str):
        self.args = args
        self.dataset_key = dataset_key

    def __getattr__(self, item):
        if item == "dataset":
            return ProxyDataset(self.args.dataset, self.dataset_key)
        return getattr(self.args, item)


class ProxyDataset(BasicDataset):
    def __init__(self, dataset: BasicDataset, split_key: str):
        self.inner = dataset
        self.split_key = split_key

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return self.inner.__getattribute__(item)

    def __getitem__(self, item):
        return self.inner[item][self.split_key]


def is_proxy_dataset(dataset: Any, inner_class: Optional[type] = None):
    is_proxy = isinstance(dataset, ProxyDataset)
    if is_proxy and inner_class is not None:
        return isinstance(dataset.inner, inner_class)
    return is_proxy


class HuggingfaceDataset(BasicDataset):
    def __init__(self):
        if not DATASETS:
            raise ValueError(
                f'Using {self.__class__.__name__} requires installing the "datasets" library, '
                f'available via "pip install datasets"'
            )
        super().__init__()

    def _load_split_data(self, split: str, split_info: SplitInfo):
        if load is None:
            raise ImportError("datasets library is not available")
        return load.load_dataset(self.name, split=split_info.name_or_file)

    def _split_path(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> Path:
        data_directory = self.data_path
        if not data_directory.exists():
            data_directory.mkdir(parents=True, exist_ok=True)
        return data_directory / self._split_filename(
            split, split_info, is_final=is_final
        )

    def _split_filename(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> str:
        hf_split, hf_path = split, split_info.name_or_file
        if is_final:
            filepath = split_info.path
            filename = filepath.name + "_" + split
            for arg in self._split_filename_additional_args(
                split, split_info.name_or_file
            ):
                filename = filename + arg
            if split_info.extension == ".safetensors":
                return filename + ".safetensors"
            return filename
        return hf_path
