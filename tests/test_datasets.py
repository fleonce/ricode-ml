import unittest

import datasets

from ricode.ml._preprocessing.map_huggingface import map_huggingface_dataset
from ricode.ml._preprocessing.tokenization import reduce_to_num_tokens, tokenize_batch
from transformers import AutoTokenizer


class DatasetsTestCase(unittest.TestCase):

    def test_select(self):
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        ds = datasets.load_dataset("tner/ontonotes5", split="train")
        dataset = map_huggingface_dataset(
            ds,
            tokenize_batch,
            column_names=["tokens"],
            mode="to-intermediate",
            fn_kwargs={
                "key": "tokens",
                "input_is_pretokenized": True,
                "tokenizer": tokenizer,
            },
            num_proc=1,
        )
        smaller_ds = dataset.select(range(1000))
        num_tokens = smaller_ds.reduce(
            reduce_to_num_tokens,
            column_names=["tokens"],
        )
        print(num_tokens)
