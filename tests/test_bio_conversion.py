import unittest

import torch

from ricode.ml._metrics.tasks.functional.ner import (
    spans_to_token_labels,
    spans_to_word_labels,
    token_labels_to_spans,
    token_labels_to_word_labels,
    word_labels_to_spans,
    word_labels_to_token_labels,
)
from ricode.ml._preprocessing.bio import JsonEntity, JsonSample
from ricode.ml._preprocessing.tokenization import tokenize_batch
from tools import foreach
from transformers import AutoTokenizer

ENTITY_TYPES = ["PER", "LOC"]
DEFAULT_TOKENIZER_NAMES = (
    "google-t5/t5-small",
    "answerdotai/ModernBERT-base",
)

DEFAULT_SAMPLES = (
    JsonSample(
        tokens=["Bob", "is", "going", "to", "San", "Marzano"],
        entities=[
            JsonEntity(start=0, end=1, type=ENTITY_TYPES[0]),
            JsonEntity(start=4, end=6, type=ENTITY_TYPES[1]),
        ],
    ),
    JsonSample(
        tokens=["Bob", "is", "going", "to", "San", "Marzano"],
        entities=[
            JsonEntity(start=0, end=1, type=ENTITY_TYPES[0]),
            JsonEntity(start=1, end=6, type=ENTITY_TYPES[1]),
        ],
    ),
)
OVERLAPPING_SAMPLES = (
    JsonSample(
        tokens=["Bob", "is", "going", "to", "San", "Marzano"],
        entities=[
            JsonEntity(start=0, end=1, type=ENTITY_TYPES[0]),
            JsonEntity(start=0, end=6, type=ENTITY_TYPES[1]),
        ],
    ),
)


class BioConversionTestCase(unittest.TestCase):

    # def verify_sample_and_decoding(self, sample: JsonSample, word_ids: torch.Tensor, ):
    @foreach(sample=OVERLAPPING_SAMPLES)
    def test_nested_entities(self, sample: JsonSample):
        with self.assertRaisesRegex(ValueError, "nested entities"):
            spans_to_word_labels(sample, ENTITY_TYPES)

    @foreach(
        sample=DEFAULT_SAMPLES,
        tokenizer_name_or_path=DEFAULT_TOKENIZER_NAMES,
        use_tokenize_fn=(
            True,
            False,
        ),
    )
    def test_word_level_to_token_level(
        self, sample: JsonSample, tokenizer_name_or_path: str, use_tokenize_fn: bool
    ):
        orig_word_labels = spans_to_word_labels(sample, ENTITY_TYPES)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if use_tokenize_fn:
            encoding = tokenize_batch(
                {"tokens": [sample["tokens"]]},
                tokenizer=tokenizer,
                key="tokens",
                input_is_pretokenized=True,
                return_word_ids=True,
                return_subtoken_mask=False,
                return_special_token_mask=False,
                word_id_dtype=None,
                return_dtype="native",
            )
            word_ids = encoding["word_ids"][0]
        else:
            encoding = tokenizer(sample["tokens"], is_split_into_words=True)
            word_ids = encoding.word_ids()

        token_labels = word_labels_to_token_labels(
            orig_word_labels,
            word_ids,
            ENTITY_TYPES,
        )

        word_labels = token_labels_to_word_labels(
            token_labels,
            word_ids,
            ENTITY_TYPES,
        )

        self.assertEqual(word_labels, orig_word_labels)

    @foreach(sample=DEFAULT_SAMPLES)
    def test_json_to_word_level_tags(self, sample: JsonSample):
        word_level_tags = spans_to_word_labels(sample, ENTITY_TYPES)
        spans = word_labels_to_spans(word_level_tags, ENTITY_TYPES, "error")
        self.assertEqual(spans, sample["entities"])

    @foreach(tokenizer_name_or_path=DEFAULT_TOKENIZER_NAMES)
    def test_simple_bio_conversion(self, tokenizer_name_or_path: str):
        sample = JsonSample(
            tokens=["Bob", "is", "going", "to", "San", "Marzano"],
            entities=[
                JsonEntity(start=0, end=1, type=ENTITY_TYPES[0]),
                JsonEntity(start=4, end=6, type=ENTITY_TYPES[1]),
            ],
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        encoding = tokenize_batch(
            {"tokens": [sample["tokens"]]},
            tokenizer=tokenizer,
            key="tokens",
            input_is_pretokenized=True,
            return_word_ids=True,
            return_subtoken_mask=True,
            word_id_dtype=torch.long,
        )

        word_ids = encoding["word_ids"][0]

        token_labels = spans_to_token_labels(
            sample,
            word_ids,
            ENTITY_TYPES,
        )

        word_labels = token_labels_to_word_labels(token_labels, word_ids, ENTITY_TYPES)
        self.assertEqual(word_labels, ["B-PER", "O", "O", "O", "B-LOC", "I-LOC"])

        spans = token_labels_to_spans(
            token_labels,
            word_ids,
            ENTITY_TYPES,
            error_handling="error",
        )

        self.assertEqual(
            sample["entities"],
            spans,
            f"{word_ids=!r} {token_labels=!r} {word_labels=!r}",
        )
