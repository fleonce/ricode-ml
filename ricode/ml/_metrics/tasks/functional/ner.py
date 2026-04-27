import sys
import warnings
from typing import Literal, Mapping, Optional, Sequence

import torch

from ricode.ml._metrics import Span
from ricode.ml._metrics.tasks.ner import NamedEntity
from ricode.ml._preprocessing.bio import JsonEntity, JsonSample
from ricode.typing_utils.protocols import SupportsToList
from ricode.utils.mappings import inverse
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


def bio_labels_for_entity_types(entity_types: Sequence[str]):
    labels = (
        ["O"]
        + ["B-" + entity_type for entity_type in entity_types]
        + ["I-" + entity_type for entity_type in entity_types]
    )
    return labels


def bio_id2label(entity_types: Sequence[str]) -> Mapping[int, str]:
    return {i: k for i, k in enumerate(bio_labels_for_entity_types(entity_types))}


def spans_to_word_labels(
    sample: JsonSample, entity_types: Sequence[str]
) -> Sequence[str]:
    word_labels = ["O"] * len(sample["tokens"])

    assigned_labels = set()
    for entity in sorted(
        sample["entities"], key=lambda e: (e["start"], e["end"] - e["start"])
    ):
        if entity["type"] not in entity_types:
            raise ValueError(
                f"Unknown entity type {entity['type']} in {entity_types=!r}"
            )

        first = True
        for pos in range(entity["start"], entity["end"]):
            if pos in assigned_labels:
                raise ValueError(
                    f"Sample {sample} contains nested entities, cannot be represented by BIO tags"
                )
            assigned_labels.add(pos)
            # todo: B only if prev. is the same tag!
            word_labels[pos] = ("B-" if first else "I-") + entity["type"]
            first = False
    return word_labels


def word_labels_to_token_labels(
    word_labels: Sequence[str],
    word_ids: Sequence[int | None],
    entity_types: Sequence[str],
):
    label2id = inverse(bio_id2label(entity_types))
    o_label = label2id["O"]

    last_word_id = None
    token_labels = [o_label] * len(word_ids)
    for pos, word_id in enumerate(word_ids):
        is_subtoken = word_id == last_word_id
        last_word_id = word_id

        if word_id is None or word_id < 0:
            # is a special token
            token_labels[pos] = -100
            continue

        word_label = word_labels[word_id]
        if word_label == "O":
            continue

        label_entity_type = word_label.split("-")[1]
        if is_subtoken and word_label.startswith("B-"):
            # this token is a subtoken; assign the I- tag for the label
            token_label = label2id["I-" + label_entity_type]
        else:
            token_label = label2id[word_label]
        token_labels[pos] = token_label

        last_word_id = word_id
    return token_labels


def spans_to_token_labels(
    sample: JsonSample,
    word_ids: Sequence[int | None] | SupportsToList,
    entity_types: Sequence[str],
):
    word_labels = spans_to_word_labels(sample, entity_types)
    return word_labels_to_token_labels(word_labels, word_ids, entity_types)


def _replace_none_with_neginf(word_ids: Sequence[int | None]) -> Sequence[int]:
    return [w if w is not None else -sys.maxsize for w in word_ids]


def token_labels_to_word_labels(
    labels: Sequence[int],
    word_ids: Sequence[int | None],
    entity_types: Sequence[str],
):
    if len(labels) != len(word_ids):
        raise ValueError(
            f"labels and word_ids must be of the same size, got {len(labels)=} vs {len(word_ids)=}"
        )

    id2label = bio_id2label(entity_types)

    num_words = max(word_ids, key=lambda w: 0 if w is None else w)
    word_labels = ["O"] * (num_words + 1)
    last_word_id = None
    for label, word_id in zip(labels, word_ids):
        if word_id is None or word_id < 0:
            # special token, ignore
            continue

        is_subtoken = last_word_id == word_id
        last_word_id = word_id

        if is_subtoken:
            # subtoken, assign the label from the primary token for this word_id
            continue
        word_labels[word_id] = id2label[int(label)]
    return word_labels


def word_labels_to_spans(
    word_labels: Sequence[str],
    entity_types: Sequence[str],
    error_handling: Literal["ignore", "warning", "error"] = "error",
    return_frozen: bool = False,
) -> Sequence[JsonEntity] | Sequence[NamedEntity]:
    current_start = -1
    current_type = None

    result = []
    index = 0
    begin_tags = 0
    for index, word_label in enumerate(word_labels):
        if word_label == "O":
            if current_type:
                if current_start < 0:
                    raise ValueError(f"Invalid decoding state: {current_start=!r}")
                result.append(
                    JsonEntity(
                        start=current_start,
                        end=index,
                        type=raise_if_none(current_type),
                    )
                )
                # reset the category
                current_type = None
                current_start = -1
        else:
            tag_entity_type = word_label.split("-")[1]
            if tag_entity_type not in entity_types:
                raise ValueError(
                    f"Unknown entity type {tag_entity_type!r}, list of known entity types is {entity_types!r}"
                )

            tag_type = word_label.split("-")[0]
            if tag_type == "B":
                # count how many entities are found, used as verification later
                begin_tags += 1
            if tag_type == "I" and current_type is None:
                if error_handling == "error":
                    raise ValueError(
                        f"Invalid sequence of word labels, starting entity with an inside tag {word_label!r}: {word_labels}"
                    )
                elif error_handling == "warning":
                    warnings.warn(
                        f"Invalid sequence of word labels, starting entity with an inside tag {word_label!r}: {word_labels}",
                        stacklevel=2,
                    )

            if current_type is not None and (
                tag_entity_type != current_type or tag_type == "B"
            ):
                if current_start < 0:
                    raise ValueError(f"Invalid decoding state: {current_start=!r}")
                result.append(
                    JsonEntity(
                        start=current_start,
                        end=index,
                        type=raise_if_none(current_type),
                    )
                )
                # reset the category
                current_type = None
            if current_type is None:
                current_start = index
            current_type = tag_entity_type

    # check for leftover entities
    if current_type is not None:
        if current_start < 0:
            raise ValueError(f"Invalid decoding state: {current_start=!r}")
        result.append(
            JsonEntity(
                start=current_start,
                end=index + 1,
                type=raise_if_none(current_type),
            )
        )

    if return_frozen:
        result = [NamedEntity(**elem) for elem in result]

    return result


def token_labels_to_spans(
    labels: torch.Tensor | Sequence[int],
    word_ids: torch.Tensor | Sequence[int],
    entity_types: Sequence[str],
    error_handling: Literal["ignore", "warning", "error"] = "error",
    return_frozen: bool = False,
) -> Sequence[JsonEntity]:
    if not isinstance(labels, list) and hasattr(labels, "tolist"):
        labels = labels.tolist()
    if not isinstance(word_ids, list) and hasattr(word_ids, "tolist"):
        word_ids = word_ids.tolist()

    word_labels = token_labels_to_word_labels(
        labels,
        word_ids,
        entity_types,
    )

    spans = word_labels_to_spans(
        word_labels, entity_types, error_handling, return_frozen
    )
    return spans


def spans_to_named_entities(
    spans: Sequence[JsonEntity], words: Sequence[str]
) -> Sequence[NamedEntity]:
    return [
        NamedEntity(
            span["start"],
            span["end"],
            span["type"],
            " ".join(words[span["start"] : span["end"]]),
        )
        for span in spans
    ]


def named_entities_to_spans(
    named_entities: Sequence[NamedEntity],
) -> Sequence[JsonEntity]:
    return [
        JsonEntity(
            start=entity.start,
            end=entity.end,
            type=entity.type,
        )
        for entity in named_entities
    ]
