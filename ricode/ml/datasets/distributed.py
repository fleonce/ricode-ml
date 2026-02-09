import json
import os
from collections import OrderedDict
from typing import Any, Mapping

from tqdm import tqdm

from ricode.ml.datasets.concatenated import ConcatenatedDataset, cumulative_sum
from ricode.ml.datasets.cumulative import CumulativeDataset


class DistributedDataset:

    @classmethod
    def from_preprocessed(cls, name_or_path: str | os.PathLike):
        with open(os.path.join(name_or_path, "dataset_info.json")) as f:
            metadata = json.load(f)
        return cls(name_or_path, metadata)

    def __init__(self, name_or_path: str, metadata: Mapping[str, Any]):
        super().__init__()
        self.dataset_type = metadata.get("type", None)
        if self.dataset_type is None:
            raise ValueError(metadata)

        self.world_size = metadata.get("world_size", 1)
        self.batch_size = metadata.get("batch_size", 1000)
        self.data_files = metadata.get("data_files", [])
        if not self.data_files:
            raise ValueError(metadata)

        if self.dataset_type == "distributed_cumulative":
            data_files = [
                f"data{i}" for i in range(len(self.data_files) // self.world_size)
            ]
            datasets_per_data = {data_file: [] for data_file in data_files}

            progress_bar = tqdm(
                desc="Loading shards",
                total=len(self.data_files),
            )

            for data_file in datasets_per_data.keys():
                for j in range(self.world_size):
                    dataset = CumulativeDataset.from_preprocessed(
                        os.path.join(name_or_path, f"{data_file}_rank{j}")
                    )

                    datasets_per_data[data_file].append(dataset)
                    progress_bar.update()

            self.subsplit_sizes = [
                sum(map(len, datasets_per_data[data_file])) for data_file in data_files
            ]
            self.cumulative_subsplit_sizes = cumulative_sum(self.subsplit_sizes)
            datasets = [
                subsplit
                for data_file in data_files
                for subsplit in datasets_per_data[data_file]
            ]
        elif self.dataset_type == "cumulative":
            datasets = [
                CumulativeDataset.from_preprocessed(
                    os.path.join(name_or_path, data_file)
                )
                for data_file in self.data_files
            ]
            self.subsplit_sizes = None
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
