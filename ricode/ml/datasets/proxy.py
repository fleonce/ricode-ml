from typing import Any, Optional

from ricode.ml.training_datasets import BasicDataset
from ricode.ml.training_types import TDataset, THparams, TrainingArgs


class ProxyTrainingArgs:
    def __init__(self, args: TrainingArgs[THparams, TDataset], dataset_key: str):
        self.args = args
        self.dataset_key = dataset_key

    def __getattr__(self, item):
        if item == "dataset":
            return ProxyDataset(self.args.dataset, self.dataset_key)
        return getattr(self.args, item)


class ProxyDataset(BasicDataset):
    def __init__(self, dataset: BasicDataset, split_key: str):
        self.inner = dataset
        self.split_key = split_key

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return self.inner.__getattribute__(item)

    def __getitem__(self, item):
        return self.inner[item][self.split_key]


def is_proxy_dataset(dataset: Any, inner_class: Optional[type] = None):
    is_proxy = isinstance(dataset, ProxyDataset)
    if is_proxy and inner_class is not None:
        return isinstance(dataset.inner, inner_class)
    return is_proxy
