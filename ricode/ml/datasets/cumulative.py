import math
import os
import pathlib
import pickle
import typing
from collections import OrderedDict
from typing import (
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
)

import numpy as np
import safetensors.torch
import torch
from more_itertools import first
from more_itertools.recipes import flatten

_cpu_device = torch.device("cpu")

T = TypeVar("T")
PY_BINSIZE = 10000
_OBJECTS_FILENAME = "objects.bin"


def dtype_for_list(values: Sequence[float] | Sequence[int]):
    if isinstance(values[0], float):
        return torch.float
    elif max(values) > 2**31 - 1:
        return torch.int64
    return torch.int32


def _separate_into_bins(
    m: Mapping[str, T]
) -> MutableMapping[str, MutableMapping[int, T]]:
    bins = OrderedDict()
    for key in sorted(m.keys()):
        value = m[key]
        name, index = key.split(".", maxsplit=1)
        if name not in bins:
            bins[name] = OrderedDict()
        bins[name][int(index)] = value
    return bins


def _flatten_bins(m: Mapping[str, Mapping[int, T]]) -> dict[str, T]:
    out = OrderedDict()
    for key in sorted(m.keys()):
        for index in sorted(m[key].keys()):
            out[f"{key}.{index}"] = m[key][index]
    return out


class FlattenedDataset:
    @classmethod
    def from_preprocessed(cls, name_or_path: str | os.PathLike):
        tensors = safetensors.torch.load_file(
            os.path.join(name_or_path, "tensors.safetensors")
        )
        objects = None
        objects_file = os.path.join(name_or_path, _OBJECTS_FILENAME)
        if os.path.exists(objects_file):
            with open(objects_file, "rb") as objects_f:
                objects = pickle.load(objects_f)
        return cls(tensors, objects)

    @classmethod
    def new_empty(cls, binsize_hints=None):
        return cls(None, None, None, binsize_hints)

    bins: MutableMapping[str, MutableMapping[int, torch.Tensor]]
    py_bins: MutableMapping[str, MutableMapping[int, MutableSequence[Any]]]
    # binsize: int
    cumulative_lengths: MutableMapping[str, list[int]]
    # track the names of the scalar field values
    scalars: set[str]

    def __init__(
        self,
        tensors: Optional[MutableMapping[str, torch.Tensor]] = None,
        objects: Optional[MutableMapping[str, MutableSequence[Any]]] = None,
        cumulative_lengths=None,
        binsizes=None,
        binsize_hints=None,
        max_binsize=2**23,
    ):
        if tensors is None:
            tensors = {}
        if objects is None:
            objects = {}
        if cumulative_lengths is None:
            if len(tensors) > 0:
                cumulative_lengths = {
                    key[: -len("_cumulative_length")]: tensors.pop(key).tolist()
                    for key in set(tensors.keys())
                    if key.endswith("_cumulative_length")
                }
            else:
                cumulative_lengths = {}
        if binsizes is None:
            binsizes = {}

        self.tensors = tensors
        self.cumulative_lengths = cumulative_lengths
        self.keys = set(cumulative_lengths.keys())

        self.bins = _separate_into_bins(tensors)
        for name, ts in self.bins.items():
            binsizes[name] = len(first(self.bins.values()))
        self.py_bins = _separate_into_bins(objects)
        self.py_keys = self.keys - set(self.bins.keys())

        self.max_binsize = max_binsize
        self.binsizes = binsizes
        self.binsize_hints = binsize_hints or {}

    @property
    def device(self) -> torch.device:
        return first(first(self.bins.values()).values()).device

    def dtype_of_key(self, key: str):
        return first(self.bins[key].values()).dtype

    def shape_of_key(self, key: str):
        return first(self.bins[key].values()).shape[1:]

    def binsize_of_key(self, key: str) -> int:
        return self.binsizes[key]

    @typing.overload
    def to(self, device: int | str | torch.device) -> "FlattenedDataset": ...

    @typing.overload
    def to(self, key: str, dtype: torch.dtype) -> "FlattenedDataset": ...

    def to(
        self,
        device_or_key: int | str | torch.device | torch.dtype,
        dtype: Optional[torch.dtype] = None,
    ):
        if dtype is not None:
            assert isinstance(device_or_key, str)
            assert isinstance(dtype, torch.dtype)
            return FlattenedDataset(
                {
                    key: (
                        value if not key.startswith(device_or_key) else value.to(dtype)
                    )
                    for key, value in self.tensors.items()
                },
                _flatten_bins(self.py_bins),
                self.cumulative_lengths,
                self.binsizes,
                self.binsize_hints,
            )
        else:
            return FlattenedDataset(
                {key: value.to(device_or_key) for key, value in self.tensors.items()},
                _flatten_bins(self.py_bins),
                self.cumulative_lengths,
                self.binsizes,
                self.binsize_hints,
            )

    def _is_py(self, key: str):
        return key in self.py_keys

    def _key_len(self, key: str):
        # return the length of the specified key in terms of elements
        return len(self.cumulative_lengths[key]) - 1

    def __len__(self):
        # cumulative_lengths always contains a 0 element, so the real size is one less!
        # furthermore, every key should have the same length!
        return len(first(self.cumulative_lengths.values())) - 1

    def __getitem__(self, item: int) -> Mapping[str, torch.Tensor]:
        if item >= len(self) or item < 0:
            raise IndexError(item)

        result = {}
        for key in self.keys:
            if self._is_py(key):
                item = self.__py_getitem__for_key(key, item)
            else:
                item = self.__getitem__for_key(key, item)
            result[key] = item
        return result

    def __py_getitem__for_key(self, key: str, item: int) -> Sequence[Any]:
        bins = self.py_bins[key]
        binsize = PY_BINSIZE  # hardcoded for now
        cumulative_lengths = self.cumulative_lengths[key]

        start, end = cumulative_lengths[item], cumulative_lengths[item + 1]
        bin_start, bin_end = start // binsize, end // binsize
        if bin_start not in bins:
            raise IndexError(bin_start, bins.keys())

        if bin_start == bin_end:
            # quick: we found an element inside a single bin
            value = bins[bin_start][start % binsize : end % binsize]
        else:
            parts = [
                # the first part
                bins[bin_start][start % binsize :],
            ]

            if bin_start + 1 < bin_end:
                for bin_intermediate in range(bin_start + 1, bin_end):
                    parts.append(bins[bin_intermediate])
            parts.append(bins[bin_end][: end % binsize])
            value = flatten(parts)

        assert len(value) == (end - start)
        return value

    def __getitem__for_key(self, key: str, item: int) -> torch.Tensor:
        bins = self.bins[key]
        binsize = self.binsizes[key]
        cumulative_lengths = self.cumulative_lengths[key]

        start, end = cumulative_lengths[item], cumulative_lengths[item + 1]
        bin_start, bin_end = start // binsize, end // binsize
        if bin_start not in bins:
            raise IndexError(bin_start, bins.keys())

        if bin_start == bin_end:
            # quick: we found an element inside a single bin
            tensor = bins[bin_start][start % binsize : end % binsize]
        else:
            parts = [
                # the first part
                bins[bin_start][start % binsize :],
                # the last part
                bins[bin_end][: end % binsize],
            ]

            if bin_start + 1 < bin_end:
                # iterate all intermediate bins and collect the full tensor
                for index, bin in enumerate(range(bin_start + 1, bin_end)):
                    parts.insert(index + 1, bins[bin])

            tensor = torch.concatenate(parts, dim=0)

        assert tensor.size(0) == (end - start)
        return tensor

    def _new_bin(
        self,
        key: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        shape: Optional[Sequence[int | torch.SymInt]] = None,
        binsize: Optional[int] = None,
    ):
        # todo: save "finished" bins to disk already!
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype_of_key(key)
        if shape is None:
            shape = self.shape_of_key(key)
        if binsize is None:
            binsize = self.binsizes[key]

        new_bin = torch.zeros((binsize, *shape), device=device, dtype=dtype)
        index = len(self.bins[key])
        self.bins[key][index] = new_bin
        self.tensors[key + "." + str(index)] = new_bin

    def _new_py_bin(
        self,
        key: str,
    ):
        index = len(self.py_bins[key])
        new_bin = []
        self.py_bins[key][index] = new_bin

    def append(
        self,
        tensors: Mapping[
            str,
            torch.Tensor
            | np.ndarray[int]
            | np.ndarray[float]
            | Sequence[int]
            | Sequence[float]
            | Sequence[Any],
        ],
    ):
        if len(self.cumulative_lengths) == 0:
            # this dataset is empty, set up the cumulative lengths list
            self.keys.update(set(tensors.keys()))
            for key in self.keys:
                self.cumulative_lengths[key] = [0]
                if (
                    not isinstance(tensors[key], torch.Tensor)
                    and isinstance(tensors[key], Sequence)
                    and not isinstance(tensors[key], np.ndarray)
                    and not isinstance(tensors[key][0], (int, float))
                ):
                    # this is a pure py object, handle differently
                    self.py_bins[key] = {}
                    self.py_keys.add(key)
                else:
                    self.bins[key] = {}

        pre_len = len(self)
        keys = set(tensors.keys()) | self.keys
        for key in keys:
            if key not in tensors:
                raise ValueError(f"Missing key {key!r}")
            elif key not in self.keys:
                raise ValueError(
                    f"Cannot introduce new keys after first iteration, got {key!r}, have {self.keys!r}"
                )

            if self._is_py(key):
                self._py_append(key, tensors[key])
            else:
                self._append(key, tensors[key])

        for key in tensors.keys():
            assert pre_len + 1 == self._key_len(key)

    def _num_bins(self, key: str):
        return len(self.bins[key])

    def _extend(
        self,
        key: str,
        tensor_or_list: (
            torch.Tensor | Sequence[int] | Sequence[float] | Sequence[torch.Tensor]
        ),
    ):
        raise NotImplementedError

    def _py_append(self, key: str, l: Sequence[Any]):
        this_sequence_length = len(l)

        if self._key_len(key) == 0:
            # we insert the first element for this key
            self._new_py_bin(key)

        binsize = PY_BINSIZE
        cumulative_lengths = self.cumulative_lengths[key]
        bins = self.py_bins[key]
        num_bins = len(bins)

        active_position = cumulative_lengths[-1] % binsize
        new_position = active_position + this_sequence_length
        must_split = new_position >= binsize
        last_bin = bins[num_bins - 1]
        if not must_split:
            last_bin[active_position : active_position + this_sequence_length] = l
        else:
            items_in_last_bin = binsize - (new_position % binsize)
            last_bin[items_in_last_bin:] = l[:items_in_last_bin]
            remainder = l[items_in_last_bin:]
            while len(remainder) > 0:
                self._new_py_bin(key)
                last_bin = bins[len(bins) - 1]
                if len(remainder) >= binsize:
                    last_bin[:] = l[:binsize]
                else:
                    last_bin[: len(remainder)] = l
        cumulative_lengths.append(cumulative_lengths[-1] + this_sequence_length)

    def _append(
        self, key: str, tensor_or_list: torch.Tensor | Sequence[int] | Sequence[float]
    ):
        this_sequence_length = len(tensor_or_list)

        if isinstance(tensor_or_list, Sequence):
            tensor = torch.tensor(tensor_or_list)
        else:
            tensor = tensor_or_list

        if self._key_len(key) == 0:
            # we insert the first element for this key, determine the bin size
            binsize = self.binsizes.get(key, None)
            if binsize is None:
                tensor_shape = tensor.shape[1:]
                num_inner_elements = math.prod(tensor_shape)
                hint = self.binsize_hints.get(key, 2**14)
                binsize = hint // num_inner_elements
                self.binsizes[key] = binsize

            # create a new bin with everything we know!
            self._new_bin(key, tensor.device, tensor.dtype, tensor.shape[1:], binsize)
        else:
            if tensor.shape[1:] != self.shape_of_key(key):
                raise ValueError(tensor.shape[1:], self.shape_of_key(key))
            if tensor.dtype != self.dtype_of_key(key):
                raise ValueError(tensor.dtype, self.dtype_of_key(key))
            if tensor.device != self.device:
                raise ValueError(tensor.device, self.device)

        binsize = self.binsizes[key]
        if this_sequence_length >= binsize:
            raise NotImplementedError

        cumulative_lengths = self.cumulative_lengths[key]
        bins = self.bins[key]
        num_bins = len(bins)

        active_position = cumulative_lengths[-1] % binsize
        new_position = active_position + this_sequence_length
        must_split = new_position >= binsize
        last_bin = bins[num_bins - 1]
        if not must_split:
            last_bin[active_position : active_position + this_sequence_length] = tensor
            if (new_position % binsize) == 0:
                # appending the tensor exactly reached the end of the current bin, advance to the next bin
                self._new_bin(key)
        else:
            sequence_length_next_bin = new_position % binsize
            sequence_length_this_bin = this_sequence_length - sequence_length_next_bin
            last_bin[active_position : active_position + sequence_length_this_bin] = (
                tensor[:sequence_length_this_bin]
            )
            # we reached the end of the bin in the middle of the current latent, split it up :)
            self._new_bin(key)
            # todo: if the bin is smaller than the introduced element, handle copying in a loop
            last_bin = bins[len(bins) - 1]
            # ... and copy the rest of the data!
            last_bin[:sequence_length_next_bin] = tensor[sequence_length_this_bin:]

        cumulative_lengths.append(cumulative_lengths[-1] + this_sequence_length)

    def save_to_disk(self, foldername: str | pathlib.Path):
        tensors = _flatten_bins(self.bins)

        for key, cumulative_lengths in self.cumulative_lengths.items():
            tensors[key + "_cumulative_length"] = torch.tensor(cumulative_lengths)

        if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)

        safetensors.torch.save_file(
            tensors, os.path.join(foldername, "tensors.safetensors"), None
        )

        if len(self.py_bins) > 0:
            objects = _flatten_bins(self.py_bins)
            with open(os.path.join(foldername, _OBJECTS_FILENAME), "wb") as objects_f:
                pickle.dump(objects, objects_f)


class FlattenedDatasetDict(dict[str, FlattenedDataset]):
    @classmethod
    def from_pretrained(cls, name_or_path: str | os.PathLike):
        return cls(
            {
                split: FlattenedDataset.from_preprocessed(
                    os.path.join(name_or_path, split)
                )
                for split in {"eval", "test", "train"}
            }
        )


CumulativeDataset: TypeAlias = FlattenedDataset
