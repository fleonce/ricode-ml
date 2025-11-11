import dataclasses
import enum
import weakref
from typing import Any, Literal, Optional, Sequence, TypeAlias, Union

import torch
from transformers import PreTrainedTokenizerBase

from ricode.ml._metrics.functional import (
    _confusion_update,
    _f1_score_update,
    LabelType,
    TensorTuple,
)
from ricode.ml._metrics.utils import (
    _is_str,
    _is_tuple_of_tokens_or_str,
    _is_tuple_of_two_ints,
)
from ricode.ml.training_types import SupportsGetItemDataclass

PositionalEntity: TypeAlias = Union[
    tuple[
        # (begin, end)
        tuple[int, int],
        # type
        LabelType,
        # tokens
        tuple[int, ...],
    ],
    tuple[
        # (begin, end)
        tuple[int, int],
        # type
        LabelType,
        # token_str
        str,
    ],
]

ELEntity: TypeAlias = tuple[tuple[tuple[int, ...], int], LabelType]

NonPositionalEntity: TypeAlias = Union[tuple[tuple[int, ...] | str, LabelType]]

_BOUNDARIES_ENTITY_TYPE = "<boundaries entity type>"


@dataclasses.dataclass(frozen=True)
class Span(SupportsGetItemDataclass):
    # [0] = tokens_or_text
    tokens_or_text: tuple[int, ...] | str
    # [1] = type
    type: str
    # [2] = Optional[position]
    position: None | tuple[int, int]

    def as_tuple(self):
        return tuple(self)

    def change_type(self, new_type: str) -> "Span":
        return Span(self.tokens_or_text, new_type, self.position)


# TODO: what do we do with these two classes?
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


class ComputeMode(enum.Enum):
    COMPUTE_CONTAMINATED = 0
    COMPUTE_CLEAN = 1


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
        if not isinstance(element, (tuple, Span)) or len(element) < 3:
            raise ValueError(
                f"Element must be a tuple ((start, stop), type, [tokens]), got {element!r}"
            )
        tokens = _is_tuple_of_tokens_or_str(element[0], "[tokens]")
        typ = _is_str(element[1], "type")
        pos = _is_tuple_of_two_ints(element[2], "(start, stop)")
    else:
        if not isinstance(element, (tuple, Span)) or len(element) != 3:
            raise ValueError(element, "must be a tuple ([tokens], type, None)")
        typ = _is_str(element[1], "type")
        tokens = _is_tuple_of_tokens_or_str(element[0], "[tokens]")
        pos = None

    if not strict:
        return pos, tokens, _BOUNDARIES_ENTITY_TYPE
    return pos, tokens, typ


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
