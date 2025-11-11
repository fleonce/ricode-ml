from typing import Any


def _is_tuple_of_two_ints(element: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(element, tuple)
        or not len(element) == 2
        or not all(isinstance(x, int) for x in element)
    ):
        raise ValueError(f"{name} must be a tuple of two ints, but is {element!r}")
    return element  # noqa


def _is_none(element: Any, name: str) -> None:
    if element is not None:
        raise ValueError(f"{name} must be None, but got {element!r}")
    return None


def _is_str(element: Any, name: str) -> str:
    if not isinstance(element, str):
        raise ValueError(f"{name} must be a str, but is {element!r}")
    return element


def _is_int(element: Any, name: str) -> int:
    if not isinstance(element, int):
        raise ValueError(f"{name} must be a int, but is {element!r}")
    return element


def _is_tuple_of_tokens_or_str(element: Any, name: str) -> tuple[int, ...] | str:
    if isinstance(element, str):
        return element
    elif not isinstance(element, tuple) or not all(isinstance(x, int) for x in element):
        raise ValueError(f"{name} must be a tuple of ints, but is {element!r}")
    return element  # noqa
