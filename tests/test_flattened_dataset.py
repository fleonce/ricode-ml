import unittest
from tempfile import TemporaryDirectory
from typing import Sequence

import torch

from ricode.ml.datasets.cumulative import FlattenedDataset
from tools import foreach

PROBLEM_SIZE = 1000
DEFAULT_DISCRETE_BOUNDS = (torch.iinfo(torch.int).min, torch.iinfo(torch.int).max)
DEFAULT_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.int8,
)
DEFAULT_PROBLEM_SIZES = (
    (1000, 100),
    (
        100,
        100,
        100,
    ),
)
DEFAULT_DEVICES = (torch.device("cpu"),)
RANDOM_SEED = 42


def _bounds_for_dtype(dtype: torch.dtype):
    if dtype.is_floating_point:
        raise ValueError("Only for integer types")
    info = torch.iinfo(dtype)
    return info.min, info.max


def craft_random_ndim_data(
    problem_size: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str | int | None,
    varying_first_dim_size: bool,
    seed: int = RANDOM_SEED,
) -> Sequence[torch.Tensor]:
    if len(problem_size) == 0:
        raise ValueError("Empty problem size")

    generator = torch.Generator().manual_seed(seed)
    if not varying_first_dim_size:
        if dtype.is_floating_point:
            # use randn
            data = torch.randn(
                problem_size, generator=generator, dtype=dtype, device=device
            ).unbind(dim=0)
        else:
            discrete_bounds = _bounds_for_dtype(dtype)
            data = torch.randint(
                discrete_bounds[0],
                discrete_bounds[1],
                problem_size,
                generator=generator,
                dtype=dtype,
                device=device,
            ).unbind(dim=0)
    else:
        if dtype.is_floating_point:
            data = [
                torch.randn(
                    problem_size[1:], generator=generator, dtype=dtype, device=device
                )
                for _ in range(problem_size[0])
            ]
        else:
            discrete_bounds = _bounds_for_dtype(dtype)
            data = [
                torch.randint(
                    discrete_bounds[0],
                    discrete_bounds[1],
                    problem_size[1:],
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(problem_size[0])
            ]
    return data


class FlattenedDatasetTestCase(unittest.TestCase):
    @foreach(
        problem_size=DEFAULT_PROBLEM_SIZES,
        dtype=DEFAULT_DTYPES,
        varying_sequence_length=(True, False),
        device=DEFAULT_DEVICES,
    )
    def test_basic_load_save_1d(
        self,
        problem_size: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        varying_sequence_length: bool,
    ):
        data = craft_random_ndim_data(
            problem_size,
            dtype,
            device,
            varying_sequence_length,
        )

        dataset = FlattenedDataset.new_empty()
        with TemporaryDirectory() as tempdir:
            for tensor in data:
                dataset.append(
                    {
                        "tensor": tensor,
                    }
                )
            self.assertEqual(len(dataset), len(data))

            dataset.save_to_disk(tempdir)
            loaded_dataset = FlattenedDataset.from_preprocessed(tempdir)

            self.assertEqual(len(loaded_dataset), len(dataset))
            for i in range(len(dataset)):
                self.assertTrue(
                    torch.equal(
                        dataset[i]["tensor"],
                        loaded_dataset[i]["tensor"],
                    ),
                    "Expected tensors to match exactly, difference detected",
                )

                self.assertTrue(
                    torch.equal(loaded_dataset[i]["tensor"], data[i]),
                    "Expected tensors to match exactly, difference detected",
                )
