import torch


def recall_at_k_compute(
    num_tp: torch.Tensor,
    num_target: torch.Tensor,
) -> torch.Tensor:
    return num_tp.div(num_target)


def recall_at_k_update(
    output: torch.Tensor,
    target: torch.Tensor,
    k: int,
    ignore_index: int | None = None,
    largest: bool = True,
):
    """
    Calculate the number of true positives for a combination of input and target

    Args:
        output: The labels of shape `(bs,num_labels)`
        target: The targets of shape `(bs,)`
        k: The number of top-k predictions to consider for recall@K
        ignore_index: an optional integer to mask out predictions from output and target

    Returns: A tuple `(num_tp,num_target)` containing the number of true positives and total targets in this update
    """

    if output.dim() != 2 or target.dim() != 1:
        raise ValueError(output.shape, target.shape)
    if output.size(1) < k:
        raise ValueError(
            f"Cannot calculate recall@{k} with a label space of {output.size(1)} elements"
        )

    if ignore_index is not None:
        target_mask = target != ignore_index

        output = output[target_mask]
        target = target[target_mask]

    output_at_topk_labels = torch.topk(
        output, k=k, dim=1, largest=largest, sorted=False
    )[1]

    num_tp = (output_at_topk_labels.eq(target.unsqueeze(1))).any(dim=1).sum()
    num_target = target.numel()
    return num_tp, num_target
