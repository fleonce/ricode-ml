import torch
from torch import Tensor


def assert_tensor_equal(
    first: Tensor,
    second: Tensor,
):
    assert torch.equal(first, second), f"{first!r} != {second!r}"
