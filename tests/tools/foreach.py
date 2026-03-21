# https://github.com/fleonce/with-argparse/blob/main/test/tools/foreach.py
import itertools
import unittest
from functools import wraps
from typing import Any, Iterable


def foreach(**setup: Iterable[Any]):
    """
    Apply the decorated function with all combination of the provided keyword arguments (cartesian product)

    Uses the unittest subTest functionality to provide additional context

    Args:
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
            combinations = itertools.product(*combinations)

            for combination in combinations:
                combination_kwargs = {arg: value for arg, value in combination}

                with self.subTest(**combination_kwargs):
                    func(self, **combination_kwargs, **kwargs)

        return wrapper

    return inner
