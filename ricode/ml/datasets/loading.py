import os.path
from typing import Mapping

from .._preprocessing.data_files import DataFile
from .._preprocessing.huggingface_parquet import load_datasets_dataset, load_parquet

from .dataset import DataFileDataset, DatasetDict

__all__ = [
    "load_parquet",
    "load_datasets_dataset",
    "load_from_file",
]


def load_from_file(file_path: str) -> DataFileDataset:
    if not os.path.exists(file_path):
        if os.path.isabs(file_path):
            raise ValueError(f"File {file_path} does not exist")
        else:
            raise ValueError(
                f"File {file_path} does not exist in directory {os.getcwd()}"
            )

    return DataFileDataset([DataFile(file_path, "file", None)])


def load_dict_from_files(file_paths: Mapping[str, str]) -> DatasetDict:
    return DatasetDict(
        {split: load_from_file(file_path) for split, file_path in file_paths.items()}
    )
