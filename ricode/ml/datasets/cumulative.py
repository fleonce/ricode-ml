import math
import os
import pathlib
import typing
from collections import OrderedDict
from typing import Mapping, MutableMapping, Optional, Sequence

import numpy as np
import safetensors.torch
import torch
from more_itertools import first


_cpu_device = torch.device("cpu")


def dtype_for_list(values: Sequence[float] | Sequence[int]):
    if isinstance(values[0], float):
        return torch.float
    elif max(values) > 2**31 - 1:
        return torch.int64
    return torch.int32


class CumulativeDataset:
    @classmethod
    def from_preprocessed(cls, name_or_path: str | os.PathLike):
        tensors = safetensors.torch.load_file(
            os.path.join(name_or_path, "tensors.safetensors")
        )
        return cls(tensors, None)

    @classmethod
    def new_empty(cls, binsize_hints=None):
        return cls(None, None, None, binsize_hints)

    bins: MutableMapping[str, MutableMapping[int, torch.Tensor]]
    # binsize: int
    cumulative_lengths: MutableMapping[str, list[int]]

    def __init__(
        self,
        tensors: Optional[MutableMapping[str, torch.Tensor]],
        cumulative_lengths=None,
        binsizes=None,
        binsize_hints=None,
        max_binsize=2**23,
    ):
        if tensors is None:
            tensors = {}
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

        bins = {}
        for key in sorted(tensors.keys()):
            tensor = tensors[key]
            name, index = key.split(".", maxsplit=1)
            if name not in bins:
                bins[name] = {}
            bins[name][int(index)] = tensor
            if name not in binsizes:
                binsizes[name] = int(tensor.size(0))
        self.bins = bins
        self.max_binsize = max_binsize
        self.binsizes = binsizes
        self.binsize_hints = binsize_hints

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
    def to(self, device: int | str | torch.device) -> "CumulativeDataset": ...

    @typing.overload
    def to(self, key: str, dtype: torch.dtype) -> "CumulativeDataset": ...

    def to(
        self,
        device_or_key: int | str | torch.device | torch.dtype,
        dtype: Optional[torch.dtype] = None,
    ):
        if dtype is not None:
            assert isinstance(device_or_key, str)
            assert isinstance(dtype, torch.dtype)
            return CumulativeDataset(
                {
                    key: (
                        value if not key.startswith(device_or_key) else value.to(dtype)
                    )
                    for key, value in self.tensors.items()
                },
                self.cumulative_lengths,
                self.binsizes,
                self.binsize_hints,
            )
        else:
            return CumulativeDataset(
                {key: value.to(device_or_key) for key, value in self.tensors.items()},
                self.cumulative_lengths,
                self.binsizes,
                self.binsize_hints,
            )

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
            result[key] = self.__getitem__for_key(key, item)
        return result

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

    def append(
        self,
        tensors: Mapping[
            str,
            torch.Tensor
            | np.ndarray[int]
            | np.ndarray[float]
            | Sequence[int]
            | Sequence[float],
        ],
    ):
        if len(self.cumulative_lengths) == 0:
            # this dataset is empty, set up the cumulative lengths list
            self.keys.update(set(tensors.keys()))
            for key in self.keys:
                self.cumulative_lengths[key] = [0]
                self.bins[key] = {}

        pre_len = len(self)
        keys = set(tensors.keys()) | self.keys
        for key in keys:
            if key not in tensors:
                raise ValueError
            elif key not in self.keys:
                raise ValueError

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
        tensors = OrderedDict(
            {
                key + "." + str(index): tensor
                for key, bins in self.bins.items()
                for index, tensor in bins.items()
            }
        )

        for key, cumulative_lengths in self.cumulative_lengths.items():
            tensors[key + "_cumulative_length"] = torch.tensor(cumulative_lengths)

        if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)

        safetensors.torch.save_file(
            tensors, os.path.join(foldername, "tensors.safetensors"), None
        )


class FlattenedDatasetDict(dict[str, CumulativeDataset]):
    @classmethod
    def from_pretrained(cls, name_or_path: str | os.PathLike):
        return cls(
            {
                split: CumulativeDataset.from_preprocessed(
                    os.path.join(name_or_path, split)
                )
                for split in {"eval", "test", "train"}
            }
        )
