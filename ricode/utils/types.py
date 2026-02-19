from typing import NoReturn, TypeVar

_T = TypeVar("_T", bound=None)


def raise_if_none(t: _T | None) -> _T | NoReturn:
    if t is None:
        raise ValueError()
    return t
