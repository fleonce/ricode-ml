import functools
import math
from collections import OrderedDict
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

__all__ = [
    "dtype_for_tokenizer",
    "reduce_to_num_tokens",
    "tokenize_batch",
]

NUM_BYTES_TO_DTYPE = {
    1: torch.uint8,
    2: torch.uint16,
    4: torch.uint32,
    8: torch.uint64,
}


def dtype_for_tokenizer(tokenizer: PreTrainedTokenizerBase):
    return _dtype_for_tokenizer(tokenizer.name_or_path, len(tokenizer))


@functools.lru_cache
def _dtype_for_tokenizer(tokenizer_name_or_path: str, size: int):
    num_bytes = math.ceil(math.log2(size) / 8)
    assert num_bytes > 1, (num_bytes, size, math.log2(size) / 8)

    rounded_bytes = 2 ** math.ceil(math.log2(num_bytes))
    if rounded_bytes < 1 or rounded_bytes > 8:
        raise ValueError(
            f"Tokenizer {tokenizer_name_or_path} requires more than {max(NUM_BYTES_TO_DTYPE.keys())} ({num_bytes}) to represent a single token (total size={size})"
        )

    return NUM_BYTES_TO_DTYPE[rounded_bytes]


def _sample_to_tensor(sample: np.ndarray | list, dtype: torch.dtype):
    if isinstance(sample, np.ndarray):
        tensor = torch.from_numpy(sample).to(dtype)
    else:
        try:
            tensor = torch.tensor(sample, dtype=dtype)
        except RuntimeError:
            raise RuntimeError(f"Failed to convert {sample!r} to {dtype=!r}") from None
    return tensor


def reduce_to_num_tokens(
    batch: MutableMapping[str, list[Any]],
) -> int:
    return sum(map(len, batch["tokens"]))


def tokenize_pretokenized_batch(
    batch: MutableMapping[str, list[Any]],
    /,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int = 0,
    key: str = "content",
):
    return tokenize_batch(batch, tokenizer, min_length, key, True)


def word_ids_as_tensor(word_ids: list[int | None], dtype: torch.dtype) -> torch.Tensor:
    smallest_int = torch.iinfo(dtype).min
    word_ids = [w if w is not None else smallest_int for w in word_ids]
    return torch.tensor(word_ids, dtype=dtype)


def tokenize_batch(
    batch: MutableMapping[str, list[Any]],
    /,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int = 0,
    key: str = "content",
    input_is_pretokenized: bool = False,
    return_word_ids: bool = False,
    word_id_dtype: torch.dtype | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
):
    if not tokenizer_kwargs:
        tokenizer_kwargs = {}
    if not isinstance(tokenizer_kwargs, MutableMapping):
        tokenizer_kwargs = OrderedDict(tokenizer_kwargs)

    if input_is_pretokenized:
        tokenizer_kwargs["is_split_into_words"] = True

    tokens_dtype = dtype_for_tokenizer(tokenizer)
    word_id_dtype = word_id_dtype or tokens_dtype

    tokenizer_input = batch[key]
    if input_is_pretokenized and "modernbert" in tokenizer.name_or_path.lower():
        tokenizer_input = [
            [
                token if pos == 0 or token.startswith(" ") else " " + token
                for pos, token in enumerate(tokens)
            ]
            for tokens in tokenizer_input
        ]

    tokenizer_output = tokenizer(
        tokenizer_input,
        truncation=True,
        **tokenizer_kwargs,
    )
    samples = tokenizer_output["input_ids"]

    output = {"tokens": []}
    if return_word_ids:
        output["word_ids"] = []

    for i in range(len(samples)):
        skip_sample = min_length and len(samples[i] < min_length)
        if skip_sample:
            continue

        output["tokens"].append(_sample_to_tensor(samples[i], tokens_dtype))
        if return_word_ids:
            word_ids = tokenizer_output.word_ids(i)
            output["word_ids"].append(word_ids_as_tensor(word_ids, word_id_dtype))

    return output
