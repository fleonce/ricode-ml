import functools
import unittest
from unittest import TextTestRunner

from tools.unittest_result import SubtestCountingTestResult


def run_tests():
    verbosity = 4

    main_func = functools.partial(
        unittest.main,
        testRunner=TextTestRunner(
            resultclass=SubtestCountingTestResult, verbosity=verbosity
        ),
        verbosity=verbosity,
    )

    main_func(None)


if __name__ == "__main__":
    run_tests()
