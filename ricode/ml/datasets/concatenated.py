from typing import Any, Mapping, Sequence

import attrs

from ricode.ml.training_types import SupportsGetItemAndLength
from ricode.ml.training_utils import cached_property


@attrs.define
class ConcatenatedDatasetOld:
    datasets: list[SupportsGetItemAndLength[Any]]

    @cached_property
    def cumulative_lengths(self) -> list[int]:
        lengths = []
        total_length = 0
        lengths.append(total_length)
        for dataset in self.datasets:
            total_length += len(dataset)
            lengths.append(total_length)
        return lengths

    def __getitem__(self, item: int):
        if item < 0 or item >= self.cumulative_lengths[-1]:
            raise IndexError(item, self.cumulative_lengths)

        index = 0
        prev_cum_len = 0
        for index, cum_len in enumerate(self.cumulative_lengths[1:]):
            if item < cum_len:
                break
            prev_cum_len = cum_len
        assert item - prev_cum_len >= 0
        assert item - prev_cum_len < len(self.datasets[index])
        return self.datasets[index][item - prev_cum_len]

    def __len__(self):
        return self.cumulative_lengths[-1]


class ConcatenatedDataset:
    @classmethod
    def from_mapping(cls, m: Mapping[Any, SupportsGetItemAndLength[Any]]):
        keys = list(sorted(m.keys()))
        datasets = [m[k] for k in keys]
        return cls(datasets)

    def __init__(self, datasets: Sequence[SupportsGetItemAndLength[Any]]):
        self.datasets = datasets
        self.sizes = list(map(lambda d: len(d), self.datasets))
        self.size = sum(self.sizes)

    @property
    def individual_sizes(self):
        return self.sizes

    @cached_property
    def cumulative_sizes(self):
        sizes = self.sizes
        cumulative_sizes = list(sizes)

        for i in range(1, len(self.datasets)):
            cumulative_sizes[i] = cumulative_sizes[i - 1] + sizes[i]
        return cumulative_sizes

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if item >= self.cumulative_sizes[-1]:
            raise IndexError(item, self.cumulative_sizes)

        index = 0
        max_index = len(self.datasets)
        while item >= self.cumulative_sizes[index] and index < max_index:
            index += 1

        if index >= max_index:
            raise IndexError(item, index, max_index, self.cumulative_sizes)
        if index == 0:
            return self.datasets[index][item]
        offset = self.cumulative_sizes[index - 1]
        return self.datasets[index][item - offset]
