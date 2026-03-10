import json
import os
from collections import OrderedDict
from typing import Any, Literal, Mapping, Optional, Sequence

import attrs
from safetensors_dataset import load_safetensors
from tqdm import tqdm

from ricode.ml._preprocessing.data_files import DataFile, ViewDataFile
from ricode.ml._preprocessing.utils import estimate_data_file_size
from ricode.ml.datasets.concatenated import ConcatenatedDataset, cumulative_sum
from ricode.ml.datasets.cumulative import CumulativeDataset
from ricode.ml.training_utils import cached_property


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
    elif dataset_info["type"] == "view":
        data_files = [
            ViewDataFile(
                name_or_path=data_file["name_or_path"],
                dataset_type=data_file["dataset_type"],
                data=None,
                indices=data_file["indices"],
            )
            for data_file in dataset_info["data_files"]
        ]
        return SelectView(None, dataset_info["indices"], data_files)
    else:
        dataset = Dataset(
            disk_folder,
            dataset_info,
            lazy=lazy,
        )
        return dataset


class LazyField:
    def __get__(self, instance, owner):
        raise NotImplementedError


class _Dataset:
    def __getattr__(self, item):
        if item in {"_data_files", "dataset"}:
            raise NotImplementedError(
                self.__class__.__name__ + "." + item + " is not defined"
            )
        return self.__getattribute__(item)

    def __len__(self):
        raise NotImplementedError(self.__class__.__name__ + ".__len__")

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
        return self.map(
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

    def map(
        self,
        fn,
        column_names: Sequence[str],
        mode: Literal["to-disk", "to-intermediate", "to-memory"] = "to-disk",
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
            mode,
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
        progress_bar: bool = True,
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
            progress_bar,
        )

    def select(self, indices: Sequence[int]):
        indices = list(indices)
        max_index = max(indices)
        min_index = min(indices)
        if min_index < 0 or max_index >= len(self):
            raise ValueError(f"Indices must be between 0 and {len(self)}")

        if self.lazy:
            data_files = self._data_files
            data_file_sizes = list(map(estimate_data_file_size, data_files))
            cumulative_data_file_sizes = cumulative_sum(data_file_sizes)
        elif isinstance(self.dataset, ConcatenatedDataset):
            cumulative_data_file_sizes = self.dataset.cumulative_sizes
        else:
            cumulative_data_file_sizes = [len(self.dataset)]

        if len(cumulative_data_file_sizes) == 1:
            return [ViewDataFile.view_from(self._data_files[0], indices)]

        # zi = zero inclusive, cumulative sum starting at zero
        zi_cumulative_data_file_sizes = [0] + cumulative_data_file_sizes

        select_data_files = {}
        for index in indices:
            # the split / data_file this index goes "into"
            index_split = 0
            for split_index, split_end_index in enumerate(cumulative_data_file_sizes):
                if index < split_end_index:
                    break
                index_split = split_index

            assert cumulative_data_file_sizes[index_split] > index
            assert zi_cumulative_data_file_sizes[index_split] <= index
            # we first filter out those data_files that are not matched by the indices given
            if index_split not in select_data_files:
                select_data_files[index_split] = []

            index_in_split = index - zi_cumulative_data_file_sizes[index_split]
            select_data_files[index_split].append(index_in_split)

        data_files = [
            ViewDataFile.view_from(
                self._data_files[split_index], select_data_files[split_index]
            )
            for split_index in sorted(select_data_files.keys())
        ]

        return SelectView(self, indices, data_files)


@attrs.define
class DataFileDataset(_Dataset):
    _data_files: Sequence[DataFile] = attrs.field(alias="data_files")

    @cached_property
    def data_file_sizes(self) -> Sequence[int]:
        return [estimate_data_file_size(data_file) for data_file in self._data_files]

    @cached_property
    def lazy(self):
        return True

    def __len__(self):
        return sum(self.data_file_sizes)


class Dataset(_Dataset):
    @staticmethod
    def from_json(file_path: str):
        return DataFileDataset([DataFile(file_path, "file", None)])

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
        return self.map(
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

    def map(
        self,
        fn,
        column_names: Sequence[str],
        mode: Literal["to-disk", "to-intermediate", "to-memory"] = "to-disk",
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
            mode,
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
        progress_bar: bool = True,
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
            progress_bar,
        )

    def save_to_disk(self, save_path: str):
        for name, split in self.items():
            split.save_to_disk(os.path.join(save_path, name))
        self.save_metadata_to_disk(save_path)

    # save metadata to disk
    def save_metadata_to_disk(self, save_path):
        with open(os.path.join(save_path, "dataset_info.json"), "w") as json_f:
            json.dump({"type": "dict", "splits": list(self.keys())}, json_f)
        del json_f


@attrs.define
class SelectView(_Dataset):
    dataset: Dataset
    indices: Sequence[int]
    _data_files: Sequence[ViewDataFile]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.dataset[index]

    def save_to_disk(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        dataset_info = os.path.join(save_path, "dataset_info.json")
        with open(dataset_info, "w") as f:
            json.dump(
                {
                    "type": "view",
                    "data_files": [
                        {
                            "name_or_path": data_file.name_or_path,
                            "dataset_type": data_file.dataset_type,
                            "indices": data_file.indices,
                        }
                        for data_file in self._data_files
                    ],
                    "indices": self.indices,
                },
                f,
            )
