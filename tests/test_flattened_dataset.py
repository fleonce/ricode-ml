import math
import string
import unittest
from random import Random
from tempfile import TemporaryDirectory
from typing import Literal, Sequence

import torch
from more_itertools import batched

from ricode.ml.datasets.cumulative import FlattenedDataset
from tools import foreach
from tools.tensors import test_dataset_equal

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
# first element in each tuple is the number of samples,
# the rest are the inner dimensions of the data, i.e., sequence length, etc.
DEFAULT_PROBLEM_SIZES = (
    (1000, 100),
    (
        100,
        100,
        100,
    ),
)
PROBLEM_SIZES_2D = (
    (1000, 100),
    (1000, 1),
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


def craft_random_data(
    problem_size: Sequence[int],
    dtype: Literal[int, bool, float, str],
    varying_first_dim_size: bool = False,
):
    """


    Args:
        problem_size: A non-empty sequence defining the shape of the data: `(n_data,...)`
        dtype: The type of the data
        varying_first_dim_size: Whether or not the first inner dimension should be of varying length

    Returns: A random sequence of data
    """
    randomness_source = Random(RANDOM_SEED)
    if problem_size[1] == 1 and varying_first_dim_size:
        varying_first_dim_size = False

    if varying_first_dim_size:
        inner_sizes = randomness_source.choices(
            list(range(1, problem_size[1])), k=problem_size[0]
        )

        if dtype == int:
            data = [
                list(randomness_source.choices(range(100000), k=inner))
                for inner in inner_sizes
            ]
        elif dtype == bool:
            data = [
                list(randomness_source.choices([True, False], k=inner))
                for inner in inner_sizes
            ]
        elif dtype == float:
            data = [
                [randomness_source.random() * 2 - 1 for _ in range(inner)]
                for inner in inner_sizes
            ]
        elif dtype == str:
            data = [
                randomness_source.choices(string.ascii_letters, k=inner)
                for inner in inner_sizes
            ]
        else:
            raise NotImplementedError(dtype)
    else:
        if dtype == int:
            data = [
                list(randomness_source.choices(range(100000), k=problem_size[1]))
                for _ in range(problem_size[0])
            ]
        elif dtype == bool:
            data = [
                list(randomness_source.choices([True, False], k=problem_size[1]))
                for _ in range(problem_size[0])
            ]
        elif dtype == float:
            data = [
                [randomness_source.random() * 2 - 1 for _ in range(problem_size[1])]
                for _ in range(problem_size[0])
            ]
        elif dtype == str:
            data = [
                randomness_source.choices(string.ascii_letters, k=problem_size[1])
                for _ in range(problem_size[0])
            ]
        else:
            raise NotImplementedError(dtype)
    return data


def craft_structured_data(problem_size: Sequence[int], varying_first_dim_size: bool):
    if varying_first_dim_size:
        raise NotImplementedError(varying_first_dim_size)
    else:
        n_elements = math.prod(problem_size)
        data = []
        for slice in batched(range(n_elements), n=problem_size[1]):
            data.append(list(slice))
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
            test_dataset_equal(self, dataset, "tensor", data)

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

    @foreach(
        problem_size=DEFAULT_PROBLEM_SIZES,
        dtype=DEFAULT_DTYPES,
        varying_sequence_length=(True, False),
        device=DEFAULT_DEVICES,
    )
    def test_insert_longer_than_binsize(
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

        dataset = FlattenedDataset.new_empty({"tensor": problem_size[0] // 2})
        for tensor in data:
            dataset.append({"tensor": tensor})

        test_dataset_equal(self, dataset, "tensor", data)

    @foreach(problem_size=PROBLEM_SIZES_2D)
    def test_insert_structured_data(self, problem_size: Sequence[int] = (1000, 1000)):
        data = craft_structured_data(problem_size, varying_first_dim_size=False)
        self.assertEqual(len(data), problem_size[0])

        dataset = FlattenedDataset.new_empty({"tensor": max(problem_size[1] // 2, 2)})
        for tensor in data:
            dataset.append({"tensor": tensor})
            self.assertFalse(dataset._is_py("tensor"))

        n_elements = math.prod(problem_size)
        expected_total_value = (n_elements * (n_elements - 1)) // 2
        i = 0
        for pos in range(len(dataset)):
            i += sum(dataset[pos]["tensor"])
            self.assertEqual(len(dataset[pos]["tensor"]), problem_size[1])
        self.assertEqual(len(dataset), problem_size[0], problem_size)
        self.assertEqual(i, expected_total_value)
        test_dataset_equal(self, dataset, "tensor", data)

    @foreach(
        problem_size=PROBLEM_SIZES_2D,
        varying_sequence_length=(True, False),
        dtype=(int, float, bool, str),
    )
    def test_insert_py_data(
        self, problem_size: Sequence[int], varying_sequence_length: bool, dtype: type
    ):
        data = craft_random_data(problem_size, dtype, varying_sequence_length)

        dataset = FlattenedDataset.new_empty()
        for element in data:
            dataset.append({"tensor": element})

        test_dataset_equal(self, dataset, "tensor", data)
