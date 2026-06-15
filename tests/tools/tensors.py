import unittest

import torch
from torch import Tensor


def assert_tensor_equal(
    first: Tensor,
    second: Tensor,
):
    assert torch.equal(first, second), f"{first!r} != {second!r}"


def test_dataset_equal(self: unittest.TestCase, dataset, key, comparison):
    for i in range(len(dataset)):
        self.assertEqual(len(dataset[i][key]), len(comparison[i]))
        value_t = type(comparison[i])
        if value_t == list:
            value_t = type(comparison[i][0])
        comparison_value = comparison[i]
        if value_t in (bool, int, float):
            comparison_value = torch.tensor(comparison[i])

        if not isinstance(comparison_value, Tensor):
            self.assertEqual(dataset[i][key], comparison[i], value_t)
        else:
            self.assertTrue(torch.equal(dataset[i][key], comparison_value))
