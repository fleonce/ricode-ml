import warnings
from typing import Mapping, Optional, Sequence, TypedDict

from tqdm import tqdm

from ricode.utils.types import raise_if_none


class JsonEntity(TypedDict):
    start: int
    end: int
    type: str


class BIOSample(TypedDict):
    tokens: Sequence[str]
    tags: Sequence[str] | Sequence[int]


class JsonSample(TypedDict):
    tokens: Sequence[str]
    entities: Sequence[JsonEntity]


def bio_to_json(
    samples: Sequence[BIOSample],
    progress: bool = False,
    desc: str | None = None,
    id_to_tag: Optional[Mapping[int, str]] = None,
    strict: bool = False,
) -> tuple[Sequence[JsonSample], set[str]]:
    result_samples = []
    result_tags = set()

    for sample in tqdm(samples, disable=not progress, desc=desc):
        json_sample, sample_tags = sample_bio_to_json(sample, id_to_tag)
        result_samples.append(json_sample)
        result_tags.update(sample_tags)
    return result_samples, result_tags


def sample_bio_to_json(
    sample: BIOSample,
    id_to_tag: Optional[Mapping[int, str]],
    strict: bool = False,
) -> tuple[JsonSample, set[str]]:
    """
    Converts a single BIO-tagged sample to the equivalent JSON variant

    Args:
        sample (BIOSample): The sample, tagged in BIO format
        id_to_tag: An optional mapping to convert an integer tag id to a string
        strict (bool): Whether to raise errors on data misalignment or just to skip entities during parsing
    Returns:
        A tuple containing the converted sample and a set of identified entity types
    """
    tags: set[str] = set()

    current_entity = []
    current_category = None
    start_index = 0
    entities = []
    num_begin_tags = 0

    def _raise_or_warn(msg: str):
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)

    def finish_entity(entity: JsonEntity):
        if entity["type"] is None:
            raise ValueError(
                f"{list(zip(sample['tokens'], sample['tags']))}\n"
                f"Entity {entity} was parsed but no type was defined"
            )
        if entity["start"] >= entity["end"]:
            raise ValueError(
                f"{list(zip(sample['tokens'], sample['tags']))}\n"
                f"Entity {entity} has zero length"
            )

        entities.append(entity)
        # record that we found a new entity type
        tags.add(entity["type"])

    index = 0
    for token, tag in zip(sample["tokens"], sample["tags"]):
        if isinstance(tag, int):
            if id_to_tag is None:
                raise ValueError("tags are ints but no mapping is provided")
            tag = id_to_tag[tag]

        if tag == "O":
            if current_entity:
                finish_entity(
                    {
                        "start": start_index,
                        "end": index,
                        "type": raise_if_none(current_category),
                    }
                )
                # reset the category
                current_category = None
            current_entity = []
        else:
            tag_category = tag.split("-")[1]
            tag_type = tag.split("-")[0]
            if tag_type == "B":
                # count how many entities are found, used as verification later
                num_begin_tags += 1
            if tag_type == "I" and not current_category:
                _raise_or_warn(f"Entity in sample {index} starts with a I- tag, beware")

            if current_entity and (tag_category != current_category or tag_type == "B"):
                finish_entity(
                    {
                        "type": raise_if_none(current_category),
                        "start": start_index,
                        "end": index,
                    }
                )
                # reset the category
                current_category = None
            if not current_entity:
                start_index = index
            current_entity.append(token)
            current_category = tag_category
        index += 1

    # check for leftover entities
    if current_entity:
        finish_entity(
            {
                "type": raise_if_none(current_category),
                "start": start_index,
                "end": len(sample["tokens"]),
            }
        )

    if len(entities) != num_begin_tags:
        _raise_or_warn(
            f"{list(zip(sample['tokens'], sample['tags']))}\n"
            f"Parsed {entities=} but expected {num_begin_tags} entities"
        )

    # why like this? makes the type checker happy :)
    sample = JsonSample(tokens=sample["tokens"], entities=entities)
    return sample, tags
