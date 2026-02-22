import dataclasses
from typing import Any, Literal, Optional, Sequence, TypeAlias, Union

import torch

from ricode.ml._metrics.functional import (
    _confusion_update,
    _f1_score_update,
    LabelType,
    TensorTuple,
)
from ricode.ml._metrics.tasks.ner import _ner_score_check_set, Span
from ricode.ml._metrics.utils import (
    _is_str,
    _is_tuple_of_tokens_or_str,
    _is_tuple_of_two_ints,
)

ELEntity: TypeAlias = tuple[
    # position
    tuple[int, int],
    # tokens_or_text
    Union[tuple[int, ...], str],
    # el label type
    LabelType,
]


@dataclasses.dataclass(frozen=True)
class ELSpan(Span):
    position: int


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
    if not isinstance(element, (tuple, Span)) or len(element) != 3:
        raise ValueError(element, "must be a tuple ([tokens], type)")
    tokens = _is_tuple_of_tokens_or_str(element[0], "[tokens]")
    typ = _is_str(element[1], "type")
    pos = _is_tuple_of_two_ints(element[2], "position")
    return pos, tokens, typ


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
