from typing import Any, Mapping, MutableMapping

import attrs

from ricode.ml.training_types import SupportsGetItem


@attrs.define
class CombinedDataset:
    datasets: list[SupportsGetItem[Mapping[str, Any]]]

    def __getitem__(self, item) -> MutableMapping[str, Any]:
        result = {}
        for dataset in self.datasets:
            for key, value in dataset[item].items():
                if key in result:
                    raise ValueError
                result[key] = value
        return result
