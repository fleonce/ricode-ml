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
        self.assertTrue(torch.equal(dataset[i][key], comparison[i]))
