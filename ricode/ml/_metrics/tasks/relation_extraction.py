import dataclasses
from typing import Any, Literal, Optional, Sequence, TypeAlias

import torch

from ricode.ml._metrics.functional import (
    _confusion_update,
    _f1_score_update,
    LabelType,
    TensorTuple,
)
from ricode.ml._metrics.tasks.ner import (
    _ner_score_check_element,
    NonPositionalEntity,
    PositionalEntity,
    Span,
)
from ricode.ml.training_types import SupportsGetItemDataclass

_BOUNDARIES_RELATION_TYPE = "<boundaries relation type>"

PositionalRelation: TypeAlias = tuple[
    PositionalEntity,
    PositionalEntity,
    LabelType,
]
NonPositionalRelation: TypeAlias = tuple[
    NonPositionalEntity, NonPositionalEntity, LabelType
]


@dataclasses.dataclass(frozen=True)
class TwoSpans(SupportsGetItemDataclass):
    head_tokens_or_text: tuple[int, ...] | str
    head_type: str
    head_position: None | tuple[int, int]
    tail_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    tail_type: str
    tail_position: None | tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Relation(SupportsGetItemDataclass):
    head: Span
    tail: Span
    type: str

    def change_type(self, new_type: str) -> "Relation":
        return Relation(self.head, self.tail, new_type)


@dataclasses.dataclass(frozen=True)
class RelationWithProbability(Relation):
    probability: float

    def change_type(self, new_type: str) -> "RelationWithProbability":
        return RelationWithProbability(self.head, self.tail, new_type, self.probability)


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


def _re_score_check_element(
    element: Any,
    position_aware: bool,
    strict_entities: bool,
    strict_relations: bool,
) -> PositionalRelation | NonPositionalRelation:
    if not isinstance(element, (tuple, Relation)) or len(element) != 3:
        raise ValueError(
            element,
            "must be a tuple (head, tail, type)",
        )
    head, tail, typ = element
    head_span = _ner_score_check_element(head, position_aware, strict_entities)
    tail_span = _ner_score_check_element(tail, position_aware, strict_entities)

    if not strict_relations:
        typ = _BOUNDARIES_RELATION_TYPE
    return (head_span, tail_span, typ)


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
