import json
import os
from collections import OrderedDict
from typing import Any, Literal, Mapping, Optional, Sequence

import attrs
from safetensors_dataset import load_safetensors
from tqdm import tqdm

from ricode.ml.datasets.concatenated import ConcatenatedDataset
from ricode.ml.datasets.cumulative import CumulativeDataset


def is_dataset(disk_folder: str | os.PathLike) -> bool:
    if not os.path.isdir(disk_folder):
        return False

    dataset_config = os.path.join(disk_folder, "dataset_info.json")
    if not os.path.exists(dataset_config):
        return False

    with open(dataset_config, "r") as f:
        dataset_info = json.load(f)
    del f
    return dataset_info["type"] != "dict"


def is_dataset_dict(disk_folder: str | os.PathLike) -> bool:
    if not os.path.isdir(disk_folder):
        return False

    dataset_info_file = os.path.join(disk_folder, "dataset_info.json")
    if not os.path.exists(dataset_info_file):
        return False

    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)
    del f
    return dataset_info["type"] == "dict"


def load_from_disk(
    disk_folder: str | os.PathLike, /, lazy: bool = False
) -> "Dataset | DatasetDict":
    if not os.path.isdir(disk_folder):
        raise ValueError(f"{disk_folder!r} is not a directory")

    dataset_info_file = os.path.join(disk_folder, "dataset_info.json")
    if not os.path.exists(dataset_info_file):
        raise ValueError(f"{dataset_info_file!r} does not exist, cannot load dataset")

    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)
    del f

    if dataset_info["type"] == "dict":
        dataset_dict = DatasetDict()
        for split in dataset_info["splits"]:
            dataset_dict[split] = load_from_disk(
                os.path.join(disk_folder, split), lazy=lazy
            )
        return dataset_dict
    else:
        dataset = Dataset(
            disk_folder,
            dataset_info,
            lazy=lazy,
        )
        return dataset


@attrs.define(hash=True)
class DataFile:
    name_or_path: str
    dataset_type: Literal["huggingface", "flattened", "safetensors", "file"] = "file"


class LazyField:
    def __get__(self, instance, owner):
        raise NotImplementedError


class Dataset:
    def __init__(
        self, name_or_path: str, metadata: Mapping[str, Any], lazy: bool = False
    ):
        super().__init__()
        self.name_or_path = name_or_path
        self.lazy = lazy
        self.dataset_type = metadata.get("type", None)
        if self.dataset_type is None:
            raise ValueError(metadata)

        self.world_size = metadata.get("world_size", 1)
        self.batch_size = metadata.get("batch_size", 1000)
        self.data_files = metadata.get("data_files", [])
        if not self.data_files:
            raise ValueError(metadata)

        if self.dataset_type == "flattened":
            if not self.lazy:
                datasets = [
                    CumulativeDataset.from_preprocessed(
                        os.path.join(name_or_path, data_file)
                    )
                    for data_file in tqdm(
                        self.data_files,
                        desc="Loading shards",
                        disable=len(self.data_files) < 10,
                    )
                ]
            else:
                datasets = None
        elif self.dataset_type == "safetensors":
            if not self.lazy:
                datasets = [
                    load_safetensors(
                        os.path.join(name_or_path, data_file, "tensors.safetensors")
                    )
                    for data_file in tqdm(
                        self.data_files,
                        desc="Loading shards",
                        disable=len(self.data_files) < 10,
                    )
                ]
            else:
                datasets = None
        else:
            raise NotImplementedError(self.dataset_type)

        self._data_files = [
            DataFile(os.path.join(name_or_path, data_file), "flattened")
            for data_file in self.data_files
        ]

        if datasets is None and self.lazy:
            self.dataset = None
        elif datasets is None:
            raise ValueError(datasets)
        elif len(datasets) > 1:
            self.dataset = ConcatenatedDataset(datasets)
        else:
            self.dataset = datasets[0]

    def __repr__(self):
        num_shards = (
            1
            if not isinstance(self.dataset, ConcatenatedDataset)
            else len(self.dataset.datasets)
        )
        return f"Dataset(samples={len(self)}, shards={num_shards})"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int):
        if not isinstance(item, int):
            raise ValueError

        return self.dataset[item]

    def map_to_disk(
        self,
        fn,
        column_names: Sequence[str],
        save_path: Optional[str] = None,
        batch_size: int = 1000,
        drop_last: bool = False,
        desc: Optional[str] = None,
        num_proc: int = 1,
        workers_per_file: int = 1,
        fn_kwargs: Optional[Mapping[str, Any]] = None,
        multiprocessing_mode: Literal["process", "threads"] = "process",
        return_dataset_type: Literal["flattened", "safetensors"] = "flattened",
        return_mapped: Literal["lazy", "in-memory"] = "in-memory",
    ) -> "Dataset":
        from ricode.ml._preprocessing.map_files import map_files

        return map_files(
            self._data_files,
            fn,
            column_names,
            "to-disk",
            save_path,
            batch_size,
            drop_last,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            return_dataset_type,
            return_mapped,
        )

    def reduce(
        self,
        fn,
        column_names: Sequence[str],
        batch_size: int = 1000,
        desc: Optional[str] = None,
        num_proc: int = 1,
        workers_per_file: int = 1,
        fn_kwargs: Optional[Mapping[str, Any]] = None,
        multiprocessing_mode: Literal["process", "threads"] = "process",
        reduction: Literal["sum", "mean", "min", "max"] = "sum",
    ):
        from ricode.ml._preprocessing.reduce_files import reduce_files

        return reduce_files(
            self._data_files,
            fn,
            column_names,
            batch_size,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            reduction,
        )


class DatasetDict(OrderedDict[str, Dataset]):

    def _to_dict_of_files(self) -> OrderedDict[str, Sequence[str]]:
        dict_of_data_files = OrderedDict()
        for split, dataset in self.items():
            dict_of_data_files[split] = dataset._data_files
        return dict_of_data_files

    def map_to_disk(
        self,
        fn,
        column_names: Sequence[str],
        save_path: Optional[str] = None,
        batch_size: int = 1000,
        drop_last: bool = False,
        desc: Optional[str] = None,
        num_proc: int = 1,
        workers_per_file: int = 1,
        fn_kwargs: Optional[Mapping[str, Any]] = None,
        multiprocessing_mode: Literal["process", "threads"] = "process",
        return_dataset_type: Literal["flattened", "safetensors"] = "flattened",
        return_mapped: Literal["lazy", "in-memory"] = "in-memory",
    ) -> "DatasetDict":
        from ricode.ml._preprocessing.map_files import map_dict_of_files

        return map_dict_of_files(
            self._to_dict_of_files(),
            fn,
            column_names,
            "to-disk",
            save_path,
            batch_size,
            drop_last,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            return_dataset_type,
            return_mapped,
        )

    def reduce(
        self,
        fn,
        column_names: Sequence[str],
        batch_size: int = 1000,
        desc: Optional[str] = None,
        num_proc: int = 1,
        workers_per_file: int = 1,
        fn_kwargs: Optional[Mapping[str, Any]] = None,
        multiprocessing_mode: Literal["process", "threads"] = "process",
        reduction: Literal["sum", "mean", "min", "max"] = "sum",
    ):
        from ricode.ml._preprocessing.reduce_files import reduce_dict_of_files

        return reduce_dict_of_files(
            self._to_dict_of_files(),
            fn,
            column_names,
            batch_size,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            reduction,
        )
