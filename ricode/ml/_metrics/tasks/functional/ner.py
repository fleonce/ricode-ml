import warnings
from typing import Mapping, Optional, Sequence

import torch

from ricode.ml._metrics import Span
from ricode.ml._preprocessing.bio import JsonEntity
from ricode.utils.types import raise_if_none


def batched_token_labels_to_spans(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    subtoken_mask: Optional[torch.Tensor],
    offset_mapping: Optional[torch.Tensor],
    original_text: str | None,
    id2label: Mapping[int, str],
    ignore_index: int = -100,
) -> Sequence[Sequence[Span]]:
    none_list = [None] * labels.size(0)

    return [
        token_labels_to_spans(
            *args,
            original_text=original_text,
            id2label=id2label,
            ignore_index=ignore_index,
        )
        for args in zip(
            input_ids.unbind(dim=0),
            labels.unbind(dim=0),
            subtoken_mask.unbind(dim=0) if subtoken_mask is not None else none_list,
            offset_mapping.unbind(dim=0) if offset_mapping is not None else none_list,
        )
    ]


def token_labels_to_spans(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    subtoken_mask: Optional[torch.Tensor],
    offset_mapping: Optional[torch.Tensor],
    original_text: str | None,
    id2label: Mapping[int, str],
    ignore_index: int = -100,
) -> Sequence[Span]:
    if labels.dim() != 1:
        raise ValueError(labels.shape)

    if (original_text is not None) ^ (offset_mapping is not None):
        raise ValueError(
            "original_text and offset_mapping must either both be None or supplied"
        )

    word_ids = (
        torch.arange(labels.size(0), device=labels.device)
        if subtoken_mask is None
        else torch.cumsum(~subtoken_mask, dim=0, dtype=torch.long)
    )

    if ignore_index is not None:
        label_mask = labels != ignore_index

        word_ids = word_ids[label_mask]
        input_ids = input_ids[label_mask]
        labels = labels[label_mask]
        if subtoken_mask is not None:
            subtoken_mask = subtoken_mask[label_mask]
        if offset_mapping is not None:
            offset_mapping = offset_mapping[label_mask]
    input_ids = input_ids.tolist()
    labels = labels.tolist()
    word_ids = word_ids.tolist()
    if subtoken_mask is not None:
        subtoken_mask = subtoken_mask.tolist()
    if offset_mapping is not None:
        offset_mapping = offset_mapping.tolist()

    spans = []

    def finish_entity(entity: JsonEntity):
        if entity["type"] is None:
            raise ValueError(
                f"{list(zip(input_ids, labels))}\n"
                f"Entity {entity} was parsed but no type was defined"
            )
        if entity["start"] >= entity["end"]:
            raise ValueError(
                f"{list(zip(input_ids, labels))}\n" f"Entity {entity} has zero length"
            )

        if original_text is not None:
            raise NotImplementedError(offset_mapping)
        else:
            span = Span(
                tokens_or_text=tuple(input_ids[entity["start"] : entity["end"]]),
                type=entity["type"],
                position=(word_ids[entity["start"]], word_ids[entity["end"]]),
            )
        spans.append(span)

    start_index = 0
    current_label: str | None = None
    num_begin_tags = 0
    index = 0
    for index, token in enumerate(input_ids):
        tag = labels[index]
        is_subtoken = bool(subtoken_mask[index] if subtoken_mask is not None else False)
        if is_subtoken:
            continue

        label = id2label[tag]

        if label == "O":
            if current_label:
                finish_entity(
                    {
                        "start": start_index,
                        "end": index,
                        "type": raise_if_none(current_label),
                    }
                )
                # reset the category
                current_label = None
        else:
            tag_category = label.split("-")[1]
            tag_type = label.split("-")[0]
            if tag_type == "B":
                # count how many entities are found, used as verification later
                num_begin_tags += 1
            if tag_type == "I" and not current_label:
                warnings.warn(f"Entity in sample {index} starts with a I- tag, beware")

            if current_label and (tag_category != current_label or tag_type == "B"):
                finish_entity(
                    {
                        "type": raise_if_none(current_label),
                        "start": start_index,
                        "end": index,
                    }
                )
                # reset the category
                current_label = None
            if not current_label:
                start_index = index
            current_label = tag_category
        index += 1

    # check for leftover entities
    if current_label:
        finish_entity(
            {
                "type": raise_if_none(current_label),
                "start": start_index,
                "end": index + 1,
            }
        )

    return spans
