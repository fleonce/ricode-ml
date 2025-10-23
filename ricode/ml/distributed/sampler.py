import math
from typing import Iterator, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()

        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.shuffle = shuffle
        assert weights_tensor.numel() == self.total_size

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g.manual_seed(self.seed + self.epoch)
        else:
            # deterministically shuffle based on seed
            g.manual_seed(self.seed)

        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        )
        indices = rand_tensor.tolist()

        if self.drop_last:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        yield from iter(indices)

    def __len__(self) -> int:
        return self.num_samples
