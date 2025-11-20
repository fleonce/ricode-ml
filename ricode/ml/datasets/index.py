from typing import Any

import attrs

from ricode.ml.training_types import SupportsGetItemAndLength


def _set_to_list(a: set[int]) -> list[int]:
    return list(a)


@attrs.define
class IndexDataset:
    indices: list[int] = attrs.field(converter=_set_to_list)
    dataset: SupportsGetItemAndLength[Any]

    def __getitem__(self, item: int):
        index = self.indices[item]
        return self.dataset[index]

    def __len__(self):
        return len(self.indices)
