import os.path
from pathlib import Path
from typing import ClassVar, Generator

from ricode.ml._metrics.tasks.functional.ner import word_sample_to_spans_sample
from ricode.ml._preprocessing.bio import JsonSample
from ricode.utils.json_files import iterate_write_with_tempfile, save_json_file_type
from ...text import TextDataset
from ...utils import DownloadInfo, DownloadOption


class OntoNotes5(TextDataset):
    data_dir = "ontonotes5"
    splits = ["dev", "test", "train"]
    required_packages = ["datasets"]

    tags: ClassVar[list[str]] = [
        "O",
        "B-CARDINAL",
        "B-DATE",
        "I-DATE",
        "B-PERSON",
        "I-PERSON",
        "B-NORP",
        "B-GPE",
        "I-GPE",
        "B-LAW",
        "I-LAW",
        "B-ORG",
        "I-ORG",
        "B-PERCENT",
        "I-PERCENT",
        "B-ORDINAL",
        "B-MONEY",
        "I-MONEY",
        "B-WORK_OF_ART",
        "I-WORK_OF_ART",
        "B-FAC",
        "B-TIME",
        "I-CARDINAL",
        "B-LOC",
        "B-QUANTITY",
        "I-QUANTITY",
        "I-NORP",
        "I-LOC",
        "B-PRODUCT",
        "I-TIME",
        "B-EVENT",
        "I-EVENT",
        "I-FAC",
        "B-LANGUAGE",
        "I-PRODUCT",
        "I-ORDINAL",
        "I-LANGUAGE",
    ]
    entity_types: ClassVar[list[str]] = list(
        sorted(set((tag.split("-")[1] for tag in tags if tag != "O")))
    )

    name_mapping = {
        "test": "test",
        "train": "train",
        "validation": "dev",
    }

    def __init__(
        self,
        root: str | Path,
        # true = always, None = if missing, false = never
        download: DownloadOption = DownloadOption.IF_MISSING_OR_OUT_OF_DATE,
    ):
        super().__init__(root, download)

    def download(self) -> DownloadInfo:
        from datasets import load_dataset

        dataset_dict = load_dataset(
            "tner/ontonotes5", revision="9ff08f45db75a23e9eebb47e29549ca4e1fbaf53"
        )

        def reformat_split(split: str) -> Generator[JsonSample, None, None]:
            for entry in dataset_dict[split]:
                yield word_sample_to_spans_sample(
                    {
                        "tokens": entry["tokens"],
                        "tags": [str(self.tags[tag]) for tag in entry["tags"]],
                    },
                    self.entity_types,
                    error_handling="warning",
                )

        data_files = {}
        for hf_split_name, split_name in self.name_mapping.items():
            split_file = f"{self.data_dir}_{split_name}.jsonl"
            data_files[split_name] = [split_file]

            iterate_write_with_tempfile(
                reformat_split,
                os.path.join(self.base_dir, split_file),
                fn_kwargs={"split": hf_split_name},
                desc=f"{self.__class__.__name__} ({split_name})",
                total=len(dataset_dict[hf_split_name]),
            )

        types_file = f"{self.data_dir}_types.json5"
        save_json_file_type(
            {
                "entities": {
                    tag: {"short": tag, "verbose": tag} for tag in self.entity_types
                },
                "type": "ner",
            },
            os.path.join(self.base_dir, types_file),
        )

        return DownloadInfo(
            [types_file],
            data_files,
        )
