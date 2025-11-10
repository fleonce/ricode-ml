import operator
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar

import torch
from torch import Tensor

from ricode.ml._metrics.functional import (
    _f1_score_flatten_batch,
    BatchedOutputs,
    LabelType,
)
from ricode.ml._metrics.tasks.ner import (
    _ner_score_check_set,
    ComputeMode,
    TokenizedSpan,
    WordBoundarySpan,
)


def _fuzzy_ner_score_update(
    output: list[set[Any]],
    target: list[set[Any]],
    average: Literal["micro", "macro", "none"],
    strict: bool,
    position_aware: bool,
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
    contaminated_entities: Optional[set[TokenizedSpan | WordBoundarySpan]] = None,
    compute_mode: ComputeMode = ComputeMode.COMPUTE_CLEAN,
) -> tuple[Tensor, Tensor, Tensor] | tuple[int, int, int]:
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

    return _fuzzy_f1_score_update(outputs, targets, average, labels, device)


def _fuzzy_f1_score_update(
    output: BatchedOutputs,
    target: BatchedOutputs,
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor] | tuple[int, int, int]:
    if len(output) != len(target):
        raise ValueError(len(output), len(target), "must be same length")

    outputs = _f1_score_flatten_batch(output)
    targets = _f1_score_flatten_batch(target)

    return _fuzzy_f1_score_update_flattened(outputs, targets, average, labels, device)


T = TypeVar("T")


def _fuzzy_compare(
    a: tuple[int, Any, LabelType, str], b: tuple[int, Any, LabelType, str]
) -> bool:
    # if the batch id does not match, we never match
    if a[0] != b[0]:
        return False
    if a == b:
        return True

    # element "a" is always from outputs
    # element "b" is always from targets
    # fuzzy match a to b => a may be larger than b?
    a_pos = a[1]
    b_pos = b[1]
    overlap = a_pos[0] <= b_pos[0] <= a_pos[1]
    if a_pos == b_pos:
        # if the positions match, return whether label is correct
        return a[2] == b[2]

    if not overlap:
        return False

    small_overlap = (
        max(
            abs(a_pos[0] - b_pos[0]),
            abs(a_pos[1] - b_pos[1]),
        )
        <= 3
    )

    if small_overlap:
        # still, we want to capture whether the predicted label is correct
        return a[2] == b[2]
    return False


def _fuzzy_set_size_operation(
    a: set[T],
    b: set[T],
    compare: Callable[[T, T], bool],
    op: Callable[[set[T], set[T]], set[T]],
) -> int:
    size = 0
    for x in a:
        match = False
        for y in b:
            # if comparison sais equal
            if compare(x, y):
                match = True
                break

        if op == operator.and_:
            size += match
        elif op == operator.sub:
            size += not match
        else:
            raise NotImplementedError(op)
    return size


def _fuzzy_f1_score_update_flattened(
    outputs: set[tuple[int, Any, LabelType]],
    targets: set[tuple[int, Any, LabelType]],
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor] | tuple[int, int, int]:
    if average in {"macro", "none"} and labels is None:
        raise ValueError(labels, " cannot be None when average = ", average)

    if average == "micro":
        num_tp = _fuzzy_set_size_operation(
            outputs, targets, _fuzzy_compare, operator.and_
        )
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
