from typing import Any, Literal, Optional, Sequence, TypeAlias, TypeVar, Union

import torch
from torch import Tensor

TensorTuple: TypeAlias = tuple[torch.Tensor, torch.Tensor]

LabelType: TypeAlias = Union[int, str]
LabelGeneric = TypeVar("LabelGeneric", int, str)
Output: TypeAlias = tuple[int, *tuple[Any, ...], LabelType]
Outputs: TypeAlias = set[Output]
BatchedOutputs: TypeAlias = Sequence[set[tuple[*tuple[Any, ...], LabelType]]]


def _replace_nan_with_zero(tensor: Tensor) -> Tensor:
    return torch.where(tensor.isnan(), 0, tensor)


def _f1_score_compute(
    num_tp: Tensor,
    num_fp: Tensor,
    num_fn: Tensor,
    average: Literal["micro", "macro", "none"],
) -> tuple[Tensor, Tensor, Tensor]:
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


def _f1_score_flatten_batch(batched_elements: BatchedOutputs) -> Outputs:
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
        for tup in elem:
            output.add((pos,) + tup)
    return output


def _f1_score_update(
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

    return _f1_score_update_flattened(outputs, targets, average, labels, device)


def _f1_score_update_flattened(
    outputs: Outputs,
    targets: Outputs,
    average: Literal["micro", "macro", "none"],
    labels: Optional[Sequence[LabelType]],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor] | tuple[int, int, int]:
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
            conf_labels[0, pos] = labels.index(elem[-1])
        if elem in targets:
            conf_labels[1, pos] = labels.index(elem[-1])
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
