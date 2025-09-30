import dataclasses
import logging
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import Field, fields as dataclass_fields, is_dataclass
from typing import (
    ClassVar,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    ParamSpec,
    Protocol,
    runtime_checkable,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import torch
from safetensors_dataset import SafetensorsDataset, SafetensorsDict
from torch import Generator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from ricode.ml.training_basics import (
    BasicHparams,
    BasicMetrics,
    Batch,
    MetricsDict,
    TensorboardLogger,
)
from ricode.ml.training_utils import cached_property


TConfig = TypeVar("TConfig")
TModel = TypeVar("TModel", bound=PreTrainedModel)
_T_CovModel = TypeVar("_T_CovModel", bound=PreTrainedModel, covariant=True)
_T_ContModel = TypeVar("_T_ContModel", bound=PreTrainedModel, contravariant=True)
THparams = TypeVar("THparams", bound=BasicHparams)
_T_ContHparams = TypeVar("_T_ContHparams", bound=BasicHparams, contravariant=True)
TMetrics = TypeVar("TMetrics", bound=BasicMetrics)
_T_CovMetrics = TypeVar("_T_CovMetrics", bound=BasicMetrics, covariant=True)
TParamSpec = ParamSpec("TParamSpec")
_T_co = TypeVar("_T_co", covariant=True)

int_tuple: TypeAlias = tuple[int, ...]
float_tuple: TypeAlias = tuple[float, ...]
str_tuple: TypeAlias = tuple[str, ...]


class SupportsNext(Protocol[_T_co]):
    def __next__(self) -> _T_co: ...


class SupportsGetItemAndLength(Protocol[_T_co]):
    def __getitem__(self, item) -> _T_co: ...

    def __len__(self) -> int: ...


# Type alias for dataclass instances, copied from https://github.com/python/typeshed/blob/9f28171658b9ca6c32a7cb93fbb99fc92b17858b/stdlib/_typeshed/__init__.pyi#L349
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field]]


T = TypeVar("T", bound=DataclassInstance)


class GetItemIterator:
    def __init__(self, iterable: SupportsGetItemAndLength):
        self.iterable = iterable
        self.pos = 0

    def __next__(self):
        pos = self.pos
        self.pos += 1
        if pos >= len(self.iterable):
            raise StopIteration
        return self.iterable.__getitem__(pos)


class SupportsGetItemDataclass(DataclassInstance):
    def _dataclass_fields(self):
        if not is_dataclass(self.__class__):
            raise ValueError("Must be a dataclass instance")
        return dataclass_fields(self)

    def __getitem__(self, item: int | slice):
        if not isinstance(item, (int, slice)):
            raise ValueError(f"item must be a int | slice, got {type(item)}")
        fields = self._dataclass_fields()
        field_or_fields_in_question = fields[item]
        if isinstance(item, slice):
            return tuple(
                getattr(self, field.name) for field in field_or_fields_in_question
            )
        return getattr(self, field_or_fields_in_question.name)

    def __iter__(self):
        return GetItemIterator(self)

    def __len__(self):
        return len(self._dataclass_fields())


@runtime_checkable
class StepBasedTraining(Protocol):
    num_steps: int
    eval_every_n_steps: int
    patience: int = 0
    max_steps: int = 0


@runtime_checkable
class EpochBasedTraining(Protocol):
    num_epochs: int
    max_epochs: int
    eval_every_n_epochs: int = 1
    patience: int = 0


@runtime_checkable
class HasOptimizerArgs(Protocol):
    lr: float
    weight_decay: float
    lr_scheduler: str
    warmup_steps: int | float

    num_steps: int
    num_epochs: int
    max_epochs: int
    optimize_for: str
    patience: int = 0
    batch_size: int
    gradient_accumulation: int = 1
    scale_lr: bool


@runtime_checkable
class HasValidationInterval(Protocol):
    """
    Required in training to determine whether to use step based evaluation or per epoch.
    """

    validation_steps: int


@runtime_checkable
class HasDatasetProperties(Protocol):
    """
    Required for training
    """

    name: str
    file_path: Optional[str]

    def list_splits(self, logger: logging.Logger): ...

    def __getitem__(self, item: str) -> SafetensorsDataset | SafetensorsDict: ...


TDataset = TypeVar("TDataset", bound=HasDatasetProperties)


@dataclasses.dataclass(kw_only=True)
class TrainingArgs(Generic[THparams, TDataset]):
    local_device: str
    seed: int
    generator: Generator
    hparams: THparams
    dataset: TDataset
    train_logger: TensorboardLogger

    # flags
    use_fsdp: bool
    use_tqdm: bool

    # tracking progress for training
    epoch: int = 0
    start_step: int = 0
    train_steps: int = 0
    grad_steps: int = 0
    best_score: Optional[tuple[int, float | int]] = None
    score_history: Mapping[str, list[float | int | tuple[float | int, ...]]] = (
        dataclasses.field(default_factory=lambda: defaultdict(list))
    )
    scores_to_track: set[str] = dataclasses.field(default_factory=set)

    @property
    def rank(self):
        from ricode.ml.training_fsdp import distributed_rank

        return distributed_rank()

    @cached_property
    def world_size(self):
        from ricode.ml.training_fsdp import distributed_world_size

        return distributed_world_size()


# ModelInitProtocol = Callable[[TConfig], TModel]

_T_cont = TypeVar("_T_cont", contravariant=True)
_U_cont = TypeVar("_U_cont", contravariant=True)
_V_cont = TypeVar("_V_cont", contravariant=True)


class ModelInitProtocol(Protocol[_T_cont, _T_co]):
    def __call__(self, config: _T_cont) -> _T_co: ...


class OptimizerInitProtocol(Protocol[_T_cont, TDataset, THparams]):
    def __call__(
        self,
        model: _T_cont,
        args: TrainingArgs[THparams, TDataset],
        num_training_steps: int,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: ...


class ConfigInitProtocol(Protocol[_T_cont, _U_cont, _T_co]):
    def __call__(self, dataset: _T_cont, hparams: _U_cont) -> _T_co: ...


@runtime_checkable
class DataLoaderProtocol(Protocol[TDataset, THparams]):
    def __call__(
        self,
        # all general arguments that may be required for creating a dataloader
        args: TrainingArgs[THparams, TDataset],
        # the split we are creating the dataloader for
        split: str,
        # do we want to use the dataloader for training?
        #   this should enable shuffling and stuff like that
        train: bool,
    ) -> DataLoader: ...


@runtime_checkable
class ForwardBackwardProtocol(Protocol[TModel, THparams, TDataset]):
    def __call__(
        self,
        model: TModel,
        args: TrainingArgs[THparams, TDataset],
        batch: Batch,
        device: torch.device | str,
        compute_context: AbstractContextManager | None,
        logger: logging.Logger,
    ) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]: ...


_U = TypeVar("_U")
_V = TypeVar("_V")
_W_cov = TypeVar("_W_cov", covariant=True)


class EvaluateProtocol(Protocol[_T_cont, TDataset, THparams, _T_CovMetrics]):
    def __call__(
        self,
        model: _T_cont,
        args: TrainingArgs[THparams, TDataset],
        split: str,
        dataloader_fn: DataLoaderProtocol[TDataset, THparams],
    ) -> _T_CovMetrics | MetricsDict[_T_CovMetrics]: ...
