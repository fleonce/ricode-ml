from typing import Mapping, MutableMapping, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def inverse(mapping: Mapping[T1, T2]) -> MutableMapping[T2, T1]:
    return {v: k for k, v in mapping.items()}
