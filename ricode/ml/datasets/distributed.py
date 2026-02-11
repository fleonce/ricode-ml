import json
import os
from collections import OrderedDict
from typing import Any, Mapping

from tqdm import tqdm

from ricode.ml.datasets.concatenated import ConcatenatedDataset
from ricode.ml.datasets.cumulative import CumulativeDataset


class DistributedDataset:

    @classmethod
    def from_preprocessed(cls, name_or_path: str | os.PathLike):
        with open(os.path.join(name_or_path, "dataset_info.json")) as f:
            metadata = json.load(f)
        return cls(name_or_path, metadata)

    def __init__(self, name_or_path: str, metadata: Mapping[str, Any]):
        super().__init__()
        self.name_or_path = name_or_path
        self.dataset_type = metadata.get("type", None)
        if self.dataset_type is None:
            raise ValueError(metadata)

        self.world_size = metadata.get("world_size", 1)
        self.batch_size = metadata.get("batch_size", 1000)
        self.data_files = metadata.get("data_files", [])
        if not self.data_files:
            raise ValueError(metadata)

        if self.dataset_type == "cumulative":
            datasets = [
                CumulativeDataset.from_preprocessed(data_file)
                for data_file in tqdm(self.data_files, desc="Loading shards")
            ]
        else:
            raise NotImplementedError(self.dataset_type)

        self.dataset = ConcatenatedDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int):
        if not isinstance(item, int):
            raise ValueError

        return self.dataset[item]


class DistributedDatasetDict(OrderedDict[str, DistributedDataset]):
    @classmethod
    def from_preprocessed(cls, name_or_path: str | os.PathLike):
        raise NotImplementedError
        with open(os.path.join(name_or_path, "dataset_info.json")) as f:
            metadata = json.load(f)
        return cls(metadata)
