# https://github.com/fleonce/with-argparse/blob/main/test/tools/foreach.py
import itertools
import unittest
from functools import wraps
from typing import Any, Iterable


def foreach(zipped: bool = False, **setup: Iterable[Any]):
    """
    Apply the decorated function with all combination of the provided keyword arguments (cartesian product)

    Uses the unittest subTest functionality to provide additional context

    Args:
        zipped: Perform a zip instead of a cartesian product
        **setup: A keyword pair of argument name and its possible values

    Returns:
        the decorated function
    """
    if len(setup) == 0:
        raise TypeError("Empty foreach configuration")

    def inner(func):
        @wraps(func)
        def wrapper(self: unittest.TestCase, **kwargs):
            combinations = []
            for arg, values in setup.items():
                values = [(arg, value) for value in values]
                combinations.append(values)
            combinations = (
                zip(*combinations) if zipped else itertools.product(*combinations)
            )

            for combination in combinations:
                combination_kwargs = {arg: value for arg, value in combination}

                if not hasattr(func, "_foreach"):
                    with self.subTest(**combination_kwargs, **kwargs):
                        func(self, **combination_kwargs, **kwargs)
                else:
                    func(self, **combination_kwargs, **kwargs)

        setattr(wrapper, "_foreach", True)
        return wrapper

    return inner
