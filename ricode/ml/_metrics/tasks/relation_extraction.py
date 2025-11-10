import dataclasses
from typing import Any, Literal, Optional, Sequence, TypeAlias

import torch

from ricode.ml._metrics.functional import (
    _confusion_update,
    _f1_score_update,
    LabelType,
    TensorTuple,
)
from ricode.ml._metrics.tasks.ner import _BOUNDARIES_ENTITY_TYPE
from ricode.ml._metrics.utils import (
    _is_str,
    _is_tuple_of_tokens_or_str,
    _is_tuple_of_two_ints,
)
from ricode.ml.training_types import SupportsGetItemDataclass

_BOUNDARIES_RELATION_TYPE = "<boundaries relation type>"

PositionalRelation: TypeAlias = tuple[
    tuple[tuple[int, int], str, tuple[int, int], str], LabelType
]
NonPositionalRelation: TypeAlias = tuple[
    tuple[tuple[int, ...] | str, str, tuple[int, ...] | str, str], LabelType
]


@dataclasses.dataclass(frozen=True)
class TwoSpans(SupportsGetItemDataclass):
    head_position_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    head_type: str
    tail_position_tokens_or_text: tuple[int, int] | tuple[int, ...] | str
    tail_type: str


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
            typ = _BOUNDARIES_RELATION_TYPE
        if not strict_entities:
            head_type = tail_type = _BOUNDARIES_ENTITY_TYPE
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
            typ = _BOUNDARIES_RELATION_TYPE
        if not strict_entities:
            head_type = tail_type = _BOUNDARIES_ENTITY_TYPE
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
