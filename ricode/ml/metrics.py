import dataclasses
import enum
import weakref
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
)

import torch
import typing_extensions
from torcheval.metrics import Metric, MulticlassConfusionMatrix
from torcheval.metrics.toolkit import sync_and_compute_collection
from transformers import PreTrainedTokenizerBase

from ricode.ml.training_fsdp import distributed_world_size
from ricode.ml.training_types import SupportsGetItemDataclass

TensorTuple: TypeAlias = tuple[torch.Tensor, torch.Tensor]

LabelType: TypeAlias = Union[int, str]
LabelGeneric = TypeVar("LabelGeneric", int, str)
Output: TypeAlias = tuple[int, Any, LabelType]
Outputs: TypeAlias = set[tuple[int, Any, LabelType]]
BatchedOutputs: TypeAlias = Sequence[set[tuple[Any, LabelType]]]

PositionalEntity: TypeAlias = tuple[tuple[int, int], LabelType]
ELEntity: TypeAlias = tuple[tuple[tuple[int, ...], int], LabelType]

NonPositionalEntity: TypeAlias = Union[tuple[tuple[int, ...] | str, LabelType]]
PositionalRelation: TypeAlias = tuple[
    tuple[tuple[int, int], str, tuple[int, int], str], LabelType
]
NonPositionalRelation: TypeAlias = tuple[
    tuple[tuple[int, ...] | str, str, tuple[int, ...] | str, str], LabelType
]

BOUNDARIES_ENTITY_TYPE = "<boundaries entity type>"
BOUNDARIES_RELATION_TYPE = "<boundaries relation type>"


@dataclasses.dataclass(frozen=True)
class TwoSpans(SupportsGetItemDataclass):
    head_position_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    head_type: str
    tail_position_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    tail_type: str


@dataclasses.dataclass(frozen=True)
class Span(SupportsGetItemDataclass):
    position_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    type: str

    def as_tuple(self):
        return tuple(self)

    def change_type(self, new_type: str) -> "Span":
        return Span(self.position_tokens_or_text, new_type)


@dataclasses.dataclass(frozen=True)
class TokenizedSpan(Span):
    position_tokens_or_text: str | tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class WordBoundarySpan(Span):
    position_tokens_or_text: tuple[int, int]
    type: str

    words: tuple[str, ...]

    def tokenized_span(
        self,
        tokenizer_ref: weakref.ReferenceType[PreTrainedTokenizerBase],
        decode_to_str: Optional[bool],
    ) -> TokenizedSpan:
        """
        Tokenize the span to get a string repr of it!
        """
        tokenizer = tokenizer_ref()
        if tokenizer is None:
            raise ValueError

        encodings = tokenizer(
            self.words[slice(*self.position_tokens_or_text)],
            is_split_into_words=True,
            add_special_tokens=False,
        )
        input_ids = tuple(encodings["input_ids"])
        if decode_to_str:
            return TokenizedSpan(tokenizer.decode(input_ids), self.type)
        return TokenizedSpan(input_ids, self.type)

    def tokenized_position(
        self,
        tokenizer: weakref.ReferenceType[PreTrainedTokenizerBase],
    ) -> tuple[int, int]:
        """
        What does happen here?

        Example:
            In NER and RE datasets, inputs are often defined as sequences of words,
            for example::

                >>> words = ["Alice", "and", "Bob", "are", "walking", "to", "Checkpoint", "Charlie"]

            and entities are then defined as sub-strings of the sequence of `words`::

                >>> entities = [{"start": 0, "end": 1, "type": "PER"}]

            where `start` is inclusive, while `end` is exclusive, i.e::

                >>> words[0:1] == ["Alice"]  # Alice is PER

            But NER models may only output
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ELSpan(Span):
    position: int


@dataclasses.dataclass(frozen=True)
class Relation(SupportsGetItemDataclass):
    two_spans: TwoSpans
    type: str

    def change_type(self, new_type: str) -> "Relation":
        return Relation(self.two_spans, new_type)


@dataclasses.dataclass(frozen=True)
class RelationWithProbability(Relation):
    probability: float

    def change_type(self, new_type: str) -> "RelationWithProbability":
        return RelationWithProbability(self.two_spans, new_type, self.probability)


class MultiMetric:
    def __init__(self, prefix: Optional[str] = None, /, **kwargs: Metric[torch.Tensor]):
        self.prefix = prefix
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @property
    def metrics(self) -> Mapping[str, Metric[torch.Tensor]]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, Metric)
        }

    def compute(self) -> Mapping[str, torch.Tensor]:
        if distributed_world_size() > 1:
            return self.sync_and_compute()

        prefix = self.prefix or ""
        return {prefix + key: metric.compute() for key, metric in self.metrics.items()}

    def __getattr__(self, item: str) -> Metric[torch.Tensor]:
        return self.metrics[item]

    def __getitem__(self, item: str) -> Metric[torch.Tensor]:
        return self.metrics[item]

    def __add__(self, other):
        raise NotImplementedError

    def update(self, input: Any, target: Any):
        for metric in self.metrics.values():
            metric.update(input, target)

    def sync_and_compute(self):
        prefix = self.prefix or ""
        return sync_and_compute_collection(
            {prefix + key: metric for key, metric in self.metrics.items()}
        )


class _NERScore(Metric[torch.Tensor], Generic[LabelGeneric]):
    return_index: ClassVar[int] = 0

    labels: Optional[Sequence[LabelGeneric]]
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        strict: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.average = average
        self.position_aware = position_aware
        self.labels = labels
        self.strict = strict
        self.num_labels = len(labels) if labels is not None else 0
        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fn", torch.tensor(0.0, device=self.device))
        elif average in {"macro", "none"}:
            self._add_state(
                "num_tp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fn", torch.zeros((self.num_labels,), device=self.device)
            )
        else:
            raise NotImplementedError(average)

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _ner_score_update(
            output,
            target,
            self.average,
            self.strict,
            self.position_aware,
            self.labels,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[self.return_index]

    def compute_with_precision_recall(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _f1_score_compute(self.num_tp, self.num_fp, self.num_fn, self.average)

    def compute_per_label(
        self,
    ) -> Mapping[
        LabelGeneric, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.average not in {"none", "macro"}:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        if self.labels is None:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        precision, recall, f1 = _f1_score_compute(
            self.num_tp, self.num_fp, self.num_fn, "none"
        )
        return {
            label: (
                precision[pos],
                recall[pos],
                f1[pos],
                self.num_fn[pos] + self.num_tp[pos] + self.num_fp[pos],
            )
            for pos, label in enumerate(self.labels)
        }

    def merge_state(self, metrics: "Iterable[_NERScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class ComputeMode(enum.Enum):
    COMPUTE_CONTAMINATED = 0
    COMPUTE_CLEAN = 1


class _ContaminationNERScore(_NERScore):

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        strict: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
        contaminated_entities: Optional[set[TokenizedSpan | WordBoundarySpan]] = None,
        compute_mode: ComputeMode = ComputeMode.COMPUTE_CLEAN,
    ):
        super().__init__(labels, strict, average, position_aware, device)

        self.contaminated_entities = contaminated_entities or set()
        self.compute_mode = compute_mode

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _ner_score_update(
            output,
            target,
            self.average,
            self.strict,
            self.position_aware,
            self.labels,
            self.device,
            self.contaminated_entities,
            self.compute_mode,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn


class NERF1Score(_NERScore, Generic[LabelGeneric]):
    return_index = 2


class NERPrecision(_NERScore):
    return_index = 0


class NERRecall(_NERScore):
    return_index = 1


class ContaminatedNERF1Score(_ContaminationNERScore, NERF1Score):
    pass


class ContaminatedNERPrecision(_ContaminationNERScore, NERPrecision):
    pass


class ContaminatedNERRecall(_ContaminationNERScore, NERRecall):
    pass


class _REScore(_NERScore):

    def __init__(
        self,
        labels: Optional[Sequence[LabelType]] = None,
        strict: bool = True,
        strict_relations: bool = True,
        average: Literal["micro", "macro", "none"] = "micro",
        position_aware: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(labels, strict, average, position_aware, device)
        self.strict_relations = strict_relations

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _re_score_update(
            output,
            target,
            self.average,
            self.position_aware,
            self.labels,
            self.strict,
            self.strict_relations,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def merge_state(self, metrics: "Iterable[_NERScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class REF1Score(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[2]


class REPrecision(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[0]


class RERecall(_REScore):
    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[1]


class _ELScore(Metric[torch.Tensor], Generic[LabelGeneric]):
    return_index: ClassVar[int] = 0

    labels: Optional[Sequence[LabelGeneric]]
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        labels: Optional[Sequence[LabelGeneric]] = None,
        average: Literal["micro", "macro", "none"] = "micro",
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.average = average
        self.labels = labels
        self.num_labels = len(labels) if labels is not None else 0
        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fn", torch.tensor(0.0, device=self.device))
        elif average in {"macro", "none"}:
            self._add_state(
                "num_tp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fp", torch.zeros((self.num_labels,), device=self.device)
            )
            self._add_state(
                "num_fn", torch.zeros((self.num_labels,), device=self.device)
            )
        else:
            raise NotImplementedError(average)

    def update(self, output: list[set[Any]], target: list[set[Any]]):
        num_tp, num_fp, num_fn = _el_score_update(
            output,
            target,
            self.average,
            self.labels,
            self.device,
        )

        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn

    def compute(self) -> torch.Tensor:
        return self.compute_with_precision_recall()[self.return_index]

    def compute_with_precision_recall(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _f1_score_compute(self.num_tp, self.num_fp, self.num_fn, self.average)

    def compute_per_label(
        self,
    ) -> Mapping[
        LabelGeneric, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.average not in {"none", "macro"}:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        if self.labels is None:
            raise ValueError(
                f"Average must be either 'none' or 'macro', not {self.average}"
            )
        precision, recall, f1 = _f1_score_compute(
            self.num_tp, self.num_fp, self.num_fn, "none"
        )
        return {
            label: (
                precision[pos],
                recall[pos],
                f1[pos],
                self.num_fn[pos] + self.num_tp[pos] + self.num_fp[pos],
            )
            for pos, label in enumerate(self.labels)
        }

    def merge_state(self, metrics: "Iterable[_ELScore]") -> typing_extensions.Self:
        for metric in metrics:
            self.num_tp += metric.num_tp
            self.num_fp += metric.num_fp
            self.num_fn += metric.num_fn
        return self


class ELF1Score(_ELScore, Generic[LabelGeneric]):
    return_index = 2


class ELPrecision(_ELScore):
    return_index = 0


class ELRecall(_ELScore):
    return_index = 1


class _MulticlassConfusionMatrix(MulticlassConfusionMatrix):

    def __init__(
        self,
        labels: Sequence[LabelType],
        position_aware: bool = False,
        *,
        normalize: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(len(labels) + 1, normalize=normalize, device=device)
        self.position_aware = position_aware
        self.labels = labels

    @torch.inference_mode()
    def update(self, input: torch.Tensor, target: torch.Tensor):
        if input.numel() == 0 and target.numel() == 0:
            return self
        return super().update(input, target)


class NERConfusionMatrix(_MulticlassConfusionMatrix):
    @torch.inference_mode()
    def update(self, output: BatchedOutputs, target: BatchedOutputs):  # type: ignore[override]
        super().update(
            *_ner_confusion_update(
                output, target, self.position_aware, self.labels, self.device
            )
        )


class REConfusionMatrix(_MulticlassConfusionMatrix):

    def __init__(
        self,
        labels: list[str],
        position_aware: bool = False,
        strict: bool = True,
        *,
        normalize: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(labels, position_aware, normalize=normalize, device=device)
        self.strict = strict

    @torch.inference_mode()
    def update(self, output: BatchedOutputs, target: BatchedOutputs):  # type: ignore[override]
        super().update(
            *_re_confusion_update(
                output,
                target,
                self.position_aware,
                self.labels,
                self.strict,
                self.device,
            )
        )


def _replace_nan_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.where(tensor.isnan(), 0, tensor)


def _f1_score_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    average: Literal["micro", "macro", "none"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision, recall, F1

    :param num_tp: TP
    :param num_fp: FP
    :param num_fn: FN
    :param average: either micro or macro
    :return:
    """
    precision = _replace_nan_with_zero(num_tp / (num_tp + num_fp))
    recall = _replace_nan_with_zero(num_tp / (num_tp + num_fn))
    f1 = _replace_nan_with_zero((2 * precision * recall) / (precision + recall))
    if average == "macro":
        precision = precision.mean(dim=0)
        recall = recall.mean(dim=0)
        f1 = f1.mean(dim=0)
    return precision, recall, f1


def _f1_score_flatten_batch(
    batched_elements: Sequence[set[tuple[Any, LabelType]]]
) -> set[tuple[int, Any, LabelType]]:
    """
    Remap a batch of elements to a single large set, makes computing TP, FP, FN easier

    Does not interfere with calculation, since we encode the position of the batch element inside each new element

    Args:
        batched_elements (list): The batched list of outputs or targets in the following format:
            Format: ```[{(info,type)}]``` where
            ``info`` uniquely identifies the prediction, whereas
            ``type`` is the type of the prediction, used for macro scores

    Returns:
        The non batched list of elements for TP, FP, FN calculation
    """
    output = set()
    for pos, elem in enumerate(batched_elements):
        for data, typ in elem:
            output.add((pos, data, typ))
    return output


def _f1_score_update(
    output: BatchedOutputs,
    target: BatchedOutputs,
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[int, int, int]:
    if len(output) != len(target):
        raise ValueError(len(output), len(target), "must be same length")

    outputs = _f1_score_flatten_batch(output)
    targets = _f1_score_flatten_batch(target)

    return _f1_score_update_flattened(outputs, targets, average, labels, device)


def _f1_score_update_flattened(
    outputs: set[tuple[int, Any, int | str]],
    targets: set[tuple[int, Any, int | str]],
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[int, int, int]:
    if average in {"macro", "none"} and labels is None:
        raise ValueError(labels, " cannot be None when average = ", average)

    if average == "micro":
        num_tp = len(outputs & targets)
        num_fp = len(outputs - targets)
        num_fn = len(targets - outputs)
        return num_tp, num_fp, num_fn
    elif average in {"macro", "none"}:
        if labels is None:
            raise ValueError(labels, " cannot be None when average = ", average)

        tp_fp_fn = torch.zeros((3, len(labels)), device=device)

        for pos, label in enumerate(labels):
            tp_fp_fn[0, pos] += len(
                {elem for elem in outputs if elem[-1] == label}
                & {elem for elem in targets if elem[-1] == label}
            )
            tp_fp_fn[1, pos] += len(
                {elem for elem in outputs if elem[-1] == label}
                - {elem for elem in targets if elem[-1] == label}
            )
            tp_fp_fn[2, pos] += len(
                {elem for elem in targets if elem[-1] == label}
                - {elem for elem in outputs if elem[-1] == label}
            )
        return tp_fp_fn[0], tp_fp_fn[1], tp_fp_fn[2]
    else:
        raise NotImplementedError(average)


def _re_score_update(
    output: list[set[Any]],
    target: list[set[Any]],
    average: Literal["micro", "macro", "none"],
    position_aware: bool,
    labels: Optional[Sequence[LabelType]],
    strict_entities: bool,
    strict_relations: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[int, int, int]:
    if len(output) != len(target):
        raise ValueError(len(output), len(target), "must be same length")
    if average in {"macro", "none"} and labels is None:
        raise ValueError(labels, " cannot be None when average = ", average)

    outputs = list(
        map(
            lambda x: _re_score_check_set(
                x, position_aware, strict_entities, strict_relations
            ),
            output,
        )
    )
    targets = list(
        map(
            lambda x: _re_score_check_set(
                x, position_aware, strict_entities, strict_relations
            ),
            target,
        )
    )

    return _f1_score_update(outputs, targets, average, labels, device)


def _is_tuple_of_two_ints(element: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(element, tuple)
        or not len(element) == 2
        or not all(isinstance(x, int) for x in element)
    ):
        raise ValueError(f"{name} must be a tuple of two ints, but is {element}")
    return element  # noqa


def _is_str(element: Any, name: str) -> str:
    if not isinstance(element, str):
        raise ValueError(f"{name} must be a str, but is {element}")
    return element


def _is_int(element: Any, name: str) -> int:
    if not isinstance(element, int):
        raise ValueError(f"{name} must be a int, but is {element}")
    return element


def _is_tuple_of_tokens_or_str(element: Any, name: str) -> tuple[int, ...] | str:
    if isinstance(element, str):
        return element
    elif not isinstance(element, tuple) or not all(isinstance(x, int) for x in element):
        raise ValueError(f"{name} must be a tuple of ints, but is {element}")
    return element  # noqa


def _re_score_check_element(
    element: Any,
    position_aware: bool,
    strict_entities: bool,
    strict_relations: bool,
) -> PositionalRelation | NonPositionalRelation:
    if position_aware:
        if (
            not isinstance(element, (tuple, Relation))
            or len(element) != 2
            or not isinstance(element[1], str)
        ):
            raise ValueError(
                element,
                "must be a tuple (((head_start, head_stop), head_typ, (tail_start, tail_stop), tail_typ), type)",
            )
        pos, typ = element
        if not isinstance(pos, (tuple, TwoSpans)) or len(pos) != 4:
            raise ValueError(
                element,
                "must be a tuple (((head_start, head_stop), head_typ, (tail_start, tail_stop), tail_typ), type)",
            )
        head_start, head_stop = _is_tuple_of_two_ints(pos[0], "(head_start, head_stop)")
        head_type = _is_str(pos[1], "head_type")
        tail_start, tail_stop = _is_tuple_of_two_ints(pos[2], "(tail_start, tail_stop)")
        tail_type = _is_str(pos[3], "tail_type")

        if not strict_relations:
            typ = BOUNDARIES_RELATION_TYPE
        if not strict_entities:
            head_type = tail_type = BOUNDARIES_ENTITY_TYPE
        return (
            (head_start, head_stop),
            head_type,
            (tail_start, tail_stop),
            tail_type,
        ), typ
    else:
        if (
            not isinstance(element, (tuple, Relation))
            or len(element) != 2
            or not isinstance(element[1], str)
        ):
            raise ValueError(
                element,
                "must be a tuple ([head_tokens, head_type, tail_tokens, tail_type], type)",
            )
        pos, typ = element[0], _is_str(element[1], "type")
        if not isinstance(pos, (tuple, TwoSpans)):
            raise ValueError(
                element,
                "must be a tuple ([head_tokens, head_type, tail_tokens, tail_type], type)",
            )
        if len(pos) != 4:
            raise ValueError(
                element,
                "must be a tuple ([head_tokens, head_type, tail_tokens, tail_type], type)",
            )
        head_tokens = _is_tuple_of_tokens_or_str(pos[0], "head_tokens")
        head_type = _is_str(pos[1], "head_type")
        tail_tokens = _is_tuple_of_tokens_or_str(pos[2], "tail_tokens")
        tail_type = _is_str(pos[3], "tail_type")

        if not strict_relations:
            typ = BOUNDARIES_RELATION_TYPE
        if not strict_entities:
            head_type = tail_type = BOUNDARIES_ENTITY_TYPE
        return (head_tokens, head_type, tail_tokens, tail_type), typ


def _re_score_check_set(
    elements: set[Any],
    position_aware: bool,
    strict_entities: bool,
    strict_relations: bool,
) -> set[PositionalRelation | NonPositionalRelation]:
    return {
        _re_score_check_element(
            element, position_aware, strict_entities, strict_relations
        )
        for element in elements
    }


def _el_score_update(
    output: list[set[Any]],
    target: list[set[Any]],
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[int, int, int]:
    outputs = list(
        map(
            lambda x: _el_score_check_set(
                x,
            ),
            output,
        )
    )
    targets = list(
        map(
            lambda x: _el_score_check_set(
                x,
            ),
            target,
        )
    )

    return _f1_score_update(outputs, targets, average, labels, device)


def _el_score_check_element(
    element: Any,
) -> ELEntity:
    if not isinstance(element, (tuple, ELSpan)) or len(element) != 3:
        raise ValueError(element, "must be a tuple ([tokens], type)")
    typ = _is_str(element[1], "type")
    tokens = _is_tuple_of_tokens_or_str(element[0], "[tokens]")
    pos = _is_int(element[2], "position")
    return (tokens, pos), typ


def _el_score_check_set(
    elements: set[Any],
) -> set[ELEntity]:
    outputs = {_el_score_check_element(element) for element in elements}
    return outputs


def _el_confusion_update(
    output: Sequence[set[Any]],
    target: Sequence[set[Any]],
    position_aware: bool,
    labels: Sequence[LabelType],
    device: torch.device,
) -> TensorTuple:
    outputs = list(map(lambda x: _ner_score_check_set(x, position_aware, True), output))
    targets = list(map(lambda x: _ner_score_check_set(x, position_aware, True), target))

    return _confusion_update(outputs, targets, labels, device)


def _ner_score_update(
    output: list[set[Any]],
    target: list[set[Any]],
    average: Literal["micro", "macro", "none"],
    strict: bool,
    position_aware: bool,
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
    contaminated_entities: Optional[set[TokenizedSpan | WordBoundarySpan]] = None,
    compute_mode: ComputeMode = ComputeMode.COMPUTE_CLEAN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[int, int, int]:
    if average in {"macro", "none"} and not strict:
        raise ValueError(f"Cannot calculate macro F1 for NER when {strict=}")

    outputs = list(
        map(
            lambda x: _ner_score_check_set(
                x, position_aware, strict, contaminated_entities, compute_mode
            ),
            output,
        )
    )
    targets = list(
        map(
            lambda x: _ner_score_check_set(
                x, position_aware, strict, contaminated_entities, compute_mode
            ),
            target,
        )
    )

    return _f1_score_update(outputs, targets, average, labels, device)


def _ner_score_check_element(
    element: Any,
    position_aware: bool,
    strict: bool,
) -> PositionalEntity | NonPositionalEntity:
    if position_aware:
        if not isinstance(element, (tuple, Span)) or len(element) != 2:
            raise ValueError(element, "must be a tuple ((start, stop), type)")
        typ = _is_str(element[1], "type")
        pos = _is_tuple_of_two_ints(element[0], "(start, stop)")

        if not strict:
            return pos, BOUNDARIES_ENTITY_TYPE
        return pos, typ
    else:
        if not isinstance(element, (tuple, Span)) or len(element) != 2:
            raise ValueError(element, "must be a tuple ([tokens], type)")
        typ = _is_str(element[1], "type")
        tokens = _is_tuple_of_tokens_or_str(element[0], "[tokens]")
        if not strict:
            return tokens, BOUNDARIES_ENTITY_TYPE
        return tokens, typ


def _ner_score_check_set(
    elements: set[Any],
    position_aware: bool,
    strict: bool,
    contaminated_entities: Optional[set[WordBoundarySpan | TokenizedSpan]] = None,
    compute_mode: ComputeMode = ComputeMode.COMPUTE_CLEAN,
) -> set[PositionalEntity | NonPositionalEntity]:
    outputs = {
        _ner_score_check_element(element, position_aware, strict)
        for element in elements
    }

    if contaminated_entities is not None and len(contaminated_entities) > 0:
        contaminated_entities = set(map(lambda x: x.as_tuple(), contaminated_entities))
        if position_aware:
            raise NotImplementedError
        else:
            filtered_outputs = {
                output
                for output in outputs
                if (
                    output not in contaminated_entities
                    if compute_mode == ComputeMode.COMPUTE_CLEAN
                    else output in contaminated_entities
                )
            }
            outputs = filtered_outputs
    return outputs


def _ner_confusion_update(
    output: Sequence[set[Any]],
    target: Sequence[set[Any]],
    position_aware: bool,
    labels: Sequence[LabelType],
    device: torch.device,
) -> TensorTuple:
    outputs = list(map(lambda x: _ner_score_check_set(x, position_aware, True), output))
    targets = list(map(lambda x: _ner_score_check_set(x, position_aware, True), target))

    return _confusion_update(outputs, targets, labels, device)


def _re_confusion_update(
    output: Sequence[set[Any]],
    target: Sequence[set[Any]],
    position_aware: bool,
    labels: Sequence[LabelType],
    strict: bool,
    device: torch.device,
) -> TensorTuple:
    outputs = list(
        map(lambda x: _re_score_check_set(x, position_aware, strict, True), output)
    )
    targets = list(
        map(lambda x: _re_score_check_set(x, position_aware, strict, True), target)
    )
    return _confusion_update(outputs, targets, labels, device)


def _confusion_set_to_tensor(
    outputs: Outputs,
    targets: Outputs,
    labels: Sequence[LabelType],
    device: torch.device,
) -> TensorTuple:
    mapping: dict[Output, int] = dict()
    outputs_and_targets = outputs | targets
    for elem in outputs_and_targets:
        mapping[elem] = len(mapping)

    conf_labels = torch.full((2, len(mapping)), len(labels), device=device)
    for elem in outputs_and_targets:
        pos = mapping[elem]
        if elem in outputs:
            conf_labels[0, pos] = labels.index(elem[2])
        if elem in targets:
            conf_labels[1, pos] = labels.index(elem[2])
    return conf_labels[0], conf_labels[1]


def _confusion_update(
    output: BatchedOutputs,
    target: BatchedOutputs,
    labels: Sequence[LabelType],
    device: torch.device,
) -> TensorTuple:
    if len(output) != len(target):
        raise ValueError(len(output), len(target), "must be same length")

    outputs = _f1_score_flatten_batch(output)
    targets = _f1_score_flatten_batch(target)

    return _confusion_set_to_tensor(outputs, targets, labels, device)
