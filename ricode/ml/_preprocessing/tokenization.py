import functools
import math
from typing import Any, MutableMapping

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
    tokens_dtype = dtype_for_tokenizer(tokenizer)
    samples = tokenizer(batch[key], truncation=True, is_split_into_words=True)[
        "input_ids"
    ]
    tokens = [
        _sample_to_tensor(sample, tokens_dtype)
        for sample in samples
        if not min_length or len(sample) >= min_length > 0
    ]
    return {"tokens": tokens}


def tokenize_batch(
    batch: MutableMapping[str, list[Any]],
    /,
    tokenizer: PreTrainedTokenizerBase,
    min_length: int = 0,
    key: str = "content",
):
    tokens_dtype = dtype_for_tokenizer(tokenizer)
    samples = tokenizer(batch[key], truncation=True)["input_ids"]
    tokens = [
        _sample_to_tensor(sample, tokens_dtype)
        for sample in samples
        if not min_length or len(sample) >= min_length > 0
    ]
    return {"tokens": tokens}
