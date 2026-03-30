from typing import Generator, NoReturn, Protocol, runtime_checkable, TypeVar

_T = TypeVar("_T", bound=None)


def raise_if_none(t: _T | None) -> _T | NoReturn:
    if t is None:
        raise ValueError()
    return t


@runtime_checkable
class ReturnsGeneratorProtocol(Protocol[_T]):
    def __call__(self, **kwargs) -> Generator[_T]: ...
