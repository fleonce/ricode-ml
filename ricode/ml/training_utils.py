import contextlib
import functools
import subprocess
import sys
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
)

import torch
from torch.nn import Module
from tqdm.contrib import DummyTqdmFile

_not_found = object()
_attribute_error = object()


T = TypeVar("T")
U = TypeVar("U")


@functools.lru_cache(1)
def get_commit_hash():
    return (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
        .stdout.decode("utf-8")
        .replace("\n", "")
    )


@functools.lru_cache(1)
def is_clean_working_tree():
    return len(get_working_tree_diff()) == 0


def get_working_tree_diff():
    return subprocess.run(
        ["git", "diff", "--no-color"], capture_output=True
    ).stdout.decode("utf-8")[:-1]


def move_to_device(state_dict: Any, device: torch.device | str) -> Any:
    if isinstance(state_dict, Mapping):
        return {key: move_to_device(value, device) for key, value in state_dict.items()}
    elif isinstance(state_dict, Sequence):
        init_class: type = list
        if isinstance(state_dict, tuple):
            init_class = tuple
        return init_class(move_to_device(value, device) for value in state_dict)
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    else:
        return state_dict


class CachedPropertySetter(Exception):
    pass


if not TYPE_CHECKING:

    class cached_property(property):  # noqa
        fget: Callable[[Any], Any]
        fset: Callable[[Any, Any], None]

        def __init__(
            self,
            fget: Callable[[Any], Any],
            fset: Callable[[Any, Any], None] | None = None,
            fdel: Callable[[Any], None] | None = None,
            doc: str | None = None,
        ):
            super().__init__(fget, fset, fdel, doc)
            self.__name__ = fget.__name__
            self.__doc__ = doc

        def setter(self, __fset: Callable[[Any, Any], None]) -> "cached_property":
            return self.__class__(self.fget, __fset, self.fdel, self.__doc__)

        def deleter(self, __fdel):
            return self.__class__(self.fget, self.fset, __fdel, self.__doc__)

        def __set__(self, __instance: Any, __value: Any) -> None:
            cached = False
            if self.fset is not None and self.fset is not Ellipsis:
                try:
                    self.fset(__instance, __value)
                    cached = True
                except CachedPropertySetter:
                    pass
            if not cached:
                if __instance is not None:
                    if not hasattr(__instance, "__dict__"):
                        raise ValueError(
                            f"{__instance} must have a __dict__ attribute to be used with cached properties"
                        )
                    __instance.__dict__[self.__name__] = __value

        def __get__(self, __instance: Any, __owner: type | None = None) -> Any:
            if __instance is None:
                return self  # type: ignore
            elif not hasattr(__instance, "__dict__"):
                raise ValueError(
                    f"{__instance} must have a __dict__ attribute to be used with cached properties"
                )
            elif (
                result := __instance.__dict__.get(self.__name__, _not_found)
            ) is _attribute_error:
                raise AttributeError
            elif result is not _not_found:
                return result

            try:
                result = self.fget(__instance)
                __instance.__dict__[self.__name__] = result
            except AttributeError:
                __instance.__dict__[self.__name__] = _attribute_error
                raise
            return result

else:
    cached_property = property


def build_chunks(
    iterable: Iterable[T], chunk_size: int
) -> Generator[Sequence[T], None, None]:
    reached_stopiteration = False
    iterator = iter(iterable)
    while not reached_stopiteration:
        chunk: list[T] = list()
        while len(chunk) < chunk_size:
            try:
                chunk.append(next(iterator))
            except StopIteration:
                reached_stopiteration = True
                break
        if len(chunk) > 0:
            yield chunk


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def estimate_model_parameters(func: Callable[[], Module]) -> int:
    with torch.device("meta"):
        model = func()
        return sum(map(lambda p: p.numel(), model.parameters()))


def map_if_not_none(a: T, func: Callable[[T], U]) -> U | None:
    if a is not None:
        return func(a)
    return None
