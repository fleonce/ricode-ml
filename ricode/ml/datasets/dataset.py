import json
import os
from collections import OrderedDict
from typing import Any, Literal, Mapping, MutableMapping, Optional, Sequence

import attrs
from safetensors_dataset import load_safetensors
from tqdm import tqdm

from ricode.ml._preprocessing.data_files import DataFile, ViewDataFile
from ricode.ml._preprocessing.lazy import LazyMapping
from ricode.ml._preprocessing.utils import (
    estimate_data_file_size,
    expected_keys_in_data_file,
)
from ricode.ml.datasets.concatenated import ConcatenatedDataset, cumulative_sum
from ricode.ml.datasets.cumulative import CumulativeDataset
from ricode.ml.training_utils import cached_property
from ricode.utils.imports import is_pyarrow_available
from ricode.utils.mappings import inverse


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
    disk_folder: str | os.PathLike,
    /,
    lazy: bool | None = None,
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
                os.path.join(disk_folder, split),
                lazy=lazy,
            )
        return dataset_dict

    return _load_from_disk(disk_folder, dataset_info, lazy)


def _load_from_disk(
    disk_folder: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    lazy: bool | None = None,
):
    if metadata["type"] in {"view", "select_view"}:
        data_files = [
            ViewDataFile(
                name_or_path=data_file["name_or_path"],
                dataset_type=data_file["dataset_type"],
                data=None,
                indices=data_file.get("indices", None),
                renamed_fields=None,
            )
            for data_file in metadata["data_files"]
        ]
        inner_dataset = _load_from_disk(disk_folder, metadata["dataset"], lazy)
        return SelectView(inner_dataset, metadata["indices"], data_files)
    elif metadata["type"] == "rename_view":
        inner_dataset = _load_from_disk(disk_folder, metadata["dataset"], lazy)
        return RenameView(inner_dataset, metadata["renamed_fields"])
    elif metadata["type"] in {"flattened", "safetensors"}:
        return Dataset(
            disk_folder,
            metadata,
            False if lazy is None else lazy,  # make sure its a bool!
        )
    elif metadata["type"] in {"data_files", "parquet"}:
        if lazy is None:
            dataset_class = (
                DataFileDataset if metadata["type"] == "data_files" else ParquetDataset
            )
        else:
            dataset_class = DataFileDataset if lazy else ParquetDataset

        data_files = [
            DataFile(
                name_or_path=data_file["name_or_path"],
                dataset_type=data_file["dataset_type"],
                data=None,
            )
            for data_file in metadata["data_files"]
        ]
        return dataset_class(data_files)
    else:
        raise NotImplementedError(metadata["type"])


class LazyField:
    def __get__(self, instance, owner):
        raise NotImplementedError


class _Dataset:
    def __getattr__(self, item):
        if item in {"_data_files", "dataset", "lazy"}:
            raise NotImplementedError(
                self.__class__.__name__ + "." + item + " is not defined",
            )
        return self.__getattribute__(item)

    def __getitem__(self, item):
        if self.lazy:
            raise ValueError(
                f"This dataset ({self.__class__.__name__}) has been loaded lazily, its data cannot be accessed yet."
            )
        raise NotImplementedError(
            self.__class__.__name__ + ".__getitem__",
        )

    def __lazy_len__(self):
        return self.cached_length

    @cached_property
    def cached_length(self):
        return sum(map(estimate_data_file_size, self._data_files))

    def __len__(self):
        raise NotImplementedError(self.__class__.__name__ + ".__len__()")

    def keys(self):
        if not self.lazy:
            return self.dataset.keys()
        return expected_keys_in_data_file(self._data_files[0])

    def save_metadata(self):
        raise NotImplementedError(self.__class__.__name__ + ".save_metadata()")

    def save_to_disk(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_metadata = self.save_metadata()

        dataset_info = os.path.join(save_path, "dataset_info.json")
        with open(dataset_info, "w") as f:
            json.dump(save_metadata, f)

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
        progress_bar: bool = True,
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
            progress_bar,
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
        progress_bar: bool = True,
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
            progress_bar,
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
            return SelectView(
                self,
                indices,
                [ViewDataFile.view_from(self._data_files[0], indices)],
            )

        # zi = zero inclusive, cumulative sum starting at zero
        zi_cumulative_data_file_sizes = [0] + cumulative_data_file_sizes

        select_data_files = {}
        for index in indices:
            # the split / data_file this index goes "into"
            index_split = 0
            for split_index, split_end_index in enumerate(cumulative_data_file_sizes):
                index_split = split_index
                if index < split_end_index:
                    break

            assert cumulative_data_file_sizes[index_split] > index
            assert zi_cumulative_data_file_sizes[index_split] <= index
            # we first filter out those data_files that are not matched by the indices given
            if index_split not in select_data_files:
                select_data_files[index_split] = []

            index_in_split = index - zi_cumulative_data_file_sizes[index_split]
            select_data_files[index_split].append(index_in_split)

        data_files = [
            ViewDataFile.view_from(
                self._data_files[split_index],
                select_data_files[split_index],
            )
            for split_index in sorted(select_data_files.keys())
        ]

        return SelectView(self, indices, data_files)

    def rename_fields(self, field_names: Mapping[str, str]):
        if all(k == v for k, v in field_names.items()):
            return self

        return RenameView(
            self,
            field_names,
        )

    def rename_field(self, field_name: str, new_field_name: str) -> "RenameView":
        return self.rename_fields({field_name: new_field_name})


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

    def save_metadata(self):
        return {
            "type": "data_files",
            "data_files": [
                {
                    "name_or_path": data_file.name_or_path,
                    "dataset_type": data_file.dataset_type,
                }
                for data_file in self._data_files
            ],
        }


class Dataset(_Dataset):
    @staticmethod
    def from_json(file_path: str):
        return DataFileDataset([DataFile(file_path, "file", None)])

    def __init__(
        self,
        name_or_path: str,
        metadata: Mapping[str, Any],
        lazy: bool = False,
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
                        os.path.join(name_or_path, data_file),
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
                        os.path.join(name_or_path, data_file, "tensors.safetensors"),
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
    _data_files: Sequence[ViewDataFile] = attrs.field(alias="data_files")

    @property
    def lazy(self):
        return self.dataset is None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.dataset[index]

    def save_metadata(self) -> Mapping[str, Any]:
        return {
            "type": "view",
            "indices": self.indices,
            "data_files": [
                {
                    "name_or_path": data_file.name_or_path,
                    "dataset_type": data_file.dataset_type,
                    "indices": data_file.indices,
                }
                for data_file in self._data_files
            ],
            "dataset": self.dataset.save_metadata(),
        }


@attrs.define
class RenameView(_Dataset):
    dataset: Dataset
    renamed_fields: Mapping[str, str]

    @property
    def _data_files(self):
        return [
            ViewDataFile(
                data_file.name_or_path,
                data_file.dataset_type,
                data_file.data,
                (
                    None
                    if not isinstance(data_file, ViewDataFile)
                    else data_file.indices
                ),
                self.renamed_fields,
            )
            for data_file in self.dataset._data_files
        ]

    @property
    def lazy(self):
        return self.dataset.lazy

    def __len__(self):
        if self.lazy:
            return self.__lazy_len__()
        if self.dataset is None:
            raise ValueError(f"{self.dataset=}")
        return len(self.dataset)

    def __getitem__(self, item):
        result = self.dataset[item]
        return {
            (self.renamed_fields[key] if key in self.renamed_fields else key): value
            for key, value in result.items()
        }

    def keys(self):
        keys = super().keys()
        return [
            # ignore missing keys and use key as a backup
            self.renamed_fields.get(key, key)
            for key in keys
        ]

    def rename_fields(self, field_names: Mapping[str, str]):
        for field_name in field_names.keys():
            if field_name not in self.keys():
                raise ValueError(
                    f"Invalid field name {field_name}, expected one of {self.keys()}"
                )

        updated_renamed_fields = dict(self.renamed_fields)
        inverted_renamed_fields = inverse(updated_renamed_fields)

        # field_name was the target of a rename before
        for field_name, new_field_name in field_names.items():
            if field_name in inverted_renamed_fields:
                updated_renamed_fields[inverted_renamed_fields[field_name]] = (
                    new_field_name
                )
            else:
                updated_renamed_fields[field_name] = new_field_name

        return RenameView(
            self.dataset,
            updated_renamed_fields,
        )

    def save_metadata(self) -> Mapping[str, Any]:
        return {
            "type": "rename_view",
            "renamed_fields": self.renamed_fields,
            "dataset": self.dataset.save_metadata(),
        }


if is_pyarrow_available():
    import pyarrow
    import pyarrow.parquet as pq

    @attrs.define
    class ParquetDataset(_Dataset):
        _data_files: Sequence[DataFile] = attrs.field(alias="data_files")
        tables: MutableMapping[int, pyarrow.Table] = attrs.field(
            init=False, factory=OrderedDict
        )

        @cached_property
        def cached_lengths(self):
            return list(map(estimate_data_file_size, self._data_files))

        @cached_property
        def cached_length(self):
            return sum(self.cached_lengths)

        @cached_property
        def cumulative_cached_lengths(self):
            return cumulative_sum(self.cached_lengths)

        def __getitem__(self, item):
            split_index = 0
            for split_index, split_end_index in enumerate(
                self.cumulative_cached_lengths
            ):
                if item < split_end_index:
                    break
            if split_index not in self.tables:
                self.tables[split_index] = table = pq.read_table(
                    self._data_files[split_index].name_or_path
                )
            else:
                table = self.tables[split_index]

            if split_index > 0:
                item -= self.cumulative_cached_lengths[split_index - 1]

            column_names = table.column_names
            return LazyMapping(
                {column_name: table[column_name][item] for column_name in column_names}
            )

        def save_metadata(self) -> Mapping[str, Any]:
            return {
                "type": "parquet",
                "dataset": self.dataset.save_metadata(),
            }

else:
    ParquetDataset = None
