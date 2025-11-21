import sys
import time
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)

import torch
from more_itertools.recipes import flatten
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from ricode.ml.datasets.concatenated import ConcatenatedDataset
from ricode.ml.training_basics import BasicHparams, Batch
from ricode.ml.training_types import (
    HasDatasetProperties,
    SupportsGetItemAndLength,
    SupportsNext,
    TrainingArgs,
)

_T_dataset = TypeVar("_T_dataset", bound=HasDatasetProperties)
_T_hparams = TypeVar("_T_hparams", bound=BasicHparams)


CollateFunction: TypeAlias = Callable[[list[dict[str, torch.Tensor]]], Batch]


class ProfilingDataLoaderIterator:
    @staticmethod
    def profile_dataloader(dataloader: DataLoader):
        return ProfilingDataLoaderIterator(dataloader)

    def __init__(self, dataloader: DataLoader):
        self.iterator = iter(dataloader)
        self.step_time = 0.0

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        t_start = time.perf_counter_ns()
        batch = next(self.iterator)
        t_end = time.perf_counter_ns()
        t_diff = t_end - t_start

        self.step_time = t_diff / 1e9
        if not isinstance(batch, Batch):
            raise ValueError("DataLoader does not return Batch")
        return batch


def create_primary_dataloader(
    # all general arguments that may be required for creating a dataloader
    args: TrainingArgs[_T_hparams, _T_dataset],
    # the split we are creating the dataloader for
    split: str,
    # do we want to use the dataloader for training?
    #   this should enable shuffling and stuff like that
    train: bool,
) -> DataLoader:
    return setup_dataloader(
        args,
        split,
        train,
        None,
    )


def setup_dataloader(
    args: TrainingArgs[_T_hparams, _T_dataset],
    # the split we are creating the dataloader for
    split: str,
    # do we want to use the dataloader for training?
    #   this should enable shuffling and stuff like that
    train: bool,
    collate_fn: Optional[CollateFunction],
    # enable memory pinning?
    # when we have lots of data, moving from pinned memory->gpu can speed up the transfer to the gpu
    pin_memory: bool = False,
    batch_size: Optional[int | Callable[[str], int]] = None,
    num_workers: int = 0,
    dataset: Optional[torch.utils.data.Dataset[Any]] = None,
    batch_sampler: Optional[Sampler[list[int]]] = None,
) -> DataLoader:
    if dataset is None:
        if not isinstance(args.dataset, HasDatasetProperties):
            raise ValueError(
                f"Dataset must be a HasDatasetProperties protocol instance, got {type(args.dataset)}"
            )
        dataset = args.dataset[split]

    if collate_fn is None:
        collate_fn = SequencePaddingDataCollator(0)

    if batch_size is None:
        batch_size = args.hparams.batch_size
    elif not isinstance(batch_size, int):
        batch_size = batch_size(split)

    do_shuffle = train is True
    if isinstance(dataset, IterableDataset):
        do_shuffle = False

    dataloader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "dataset": dataset,
        "shuffle": do_shuffle,
        "generator": args.generator,
        "drop_last": train is True and args.job_config.parallelize.dp_mode == "fsdp",
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if args.job_config.parallelize.dp_mode != "none":
        shuffle = dataloader_kwargs.pop("shuffle")
        dataloader_kwargs.pop("drop_last")
        sampler: DistributedSampler
        sampler = DistributedSampler(
            dataset,
            args.world_size,
            args.rank,
            shuffle,
            args.seed,
            drop_last=True,
        )
        if shuffle:
            sampler.set_epoch(args.epoch)

        dataloader_kwargs.update({"sampler": sampler})

    if batch_sampler is not None:
        if args.job_config.parallelize.dp_mode != "none":
            raise NotImplementedError

        dataloader_kwargs.pop("batch_size")
        dataloader_kwargs.pop("shuffle")
        dataloader_kwargs["batch_sampler"] = batch_sampler
    return DataLoader(**dataloader_kwargs)


setup_default_dataloader = create_primary_dataloader


class MergingCollateFunc(Protocol):
    def __call__(self, *batches: Batch) -> Batch: ...


def try_len(dataloader: DataLoader):
    try:
        return len(dataloader)
    except Exception:
        return sys.maxsize


class MergingDataloader(DataLoader):
    dataloaders: Sequence[DataLoader]
    collate_func: Optional[MergingCollateFunc]
    iters: tuple[SupportsNext[Batch], ...]

    def __init__(
        self,
        dataloaders: Sequence[DataLoader],
        collate_func: Optional[MergingCollateFunc] = None,
        infinite_non_primary_dataloaders: bool = False,
    ):
        self.step = 0
        self.dataloaders = dataloaders
        self.collate_func = collate_func
        self.infinite_non_primary_dataloaders = infinite_non_primary_dataloaders
        self.iters = ()

        if (
            len(set(try_len(dataloader) for dataloader in dataloaders)) != 1
            and infinite_non_primary_dataloaders
        ):
            warnings.warn(
                f"Treating dataloaders[0] as the primary dataloader, "
                f"as the dataloaders specified have different lengths: "
                f"{tuple((try_len(dataloader) for dataloader in dataloaders))}"
            )

    def __iter__(self):
        self.step = 0
        self.iters = tuple(iter(loader) for loader in self.dataloaders)
        return self

    def __len__(self):
        if self.infinite_non_primary_dataloaders:
            return try_len(self.dataloaders[0])
        return min(try_len(loader) for loader in self.dataloaders)

    def _reinitialize_pos(self, pos: int):
        new_iter = iter(self.dataloaders[pos])
        self.iters = self.iters[:pos] + (new_iter,) + self.iters[pos + 1 :]
        return new_iter

    def _get_next(self):
        batches = list()
        for pos, iterator in enumerate(self.iters):
            try:
                batches.append(next(iterator))
            except StopIteration:
                if pos == 0 or not self.infinite_non_primary_dataloaders:
                    raise StopIteration
                elif self.infinite_non_primary_dataloaders:
                    iterator = self._reinitialize_pos(pos)
                    try:
                        batches.append(next(iterator))
                    except StopIteration as si:
                        raise ValueError(
                            f"Got StopIteration at the first new next(...) call for pos {pos}"
                        ) from si
        return batches

    def __next__(self) -> Batch:
        try:
            batches = self._get_next()
            self.step += 1
        except StopIteration:
            raise StopIteration
        if self.collate_func:
            return self.collate_func(*batches)

        batch = Batch()
        for elem in batches:
            batch.update(elem)
        return batch


class InfiniteDataLoader:
    iter: Optional[SupportsNext[Batch]]

    def __init__(self, inner_dataloader: DataLoader):
        self.inner = inner_dataloader
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.inner)
        return self

    def __next__(self):
        try:
            if self.iter is None:
                raise ValueError("iter is None")
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.inner)
            return next(self)

    def __len__(self):
        return sys.maxsize


class JoinedBatchSampler(Sampler[list[int]]):
    def __init__(self, samplers: Iterable[BatchSampler]):
        super().__init__()
        self.samplers = samplers

    def __iter__(self):
        for samples in zip(*self.samplers):
            yield flatten(samples)
        pass

    def __len__(self):
        return min(len(sampler) for sampler in self.samplers)


def join_datasets(
    datasets: Sequence[SupportsGetItemAndLength[Any]],
    batch_sizes: Sequence[int],
    shuffle: bool,
    generator: Optional[torch.Generator],
    drop_last: bool,
):
    concatenated_dataset = ConcatenatedDataset(datasets)

    if shuffle:
        samplers = [RandomSampler(dataset, generator=generator) for dataset in datasets]
    else:
        samplers = [SequentialSampler(dataset) for dataset in datasets]

    batch_samplers = [
        BatchSampler(sampler, batch_size, drop_last)
        for sampler, batch_size in zip(samplers, batch_sizes)
    ]
    batch_sampler = JoinedBatchSampler(batch_samplers)
    return concatenated_dataset, batch_sampler


class SequencePaddingDataCollator:
    def __init__(
        self,
        padding: float,
        key_order: Optional[list[str]] = None,
        pad_token: Optional[int] = None,
        attn_mask_dtype: Optional[torch.dtype] = torch.long,
        keys_with_tokens: Optional[set[str]] = None,
        allow_partial_batch_elements: bool = False,
    ):
        self.padding = padding
        self.key_order = {key: pos for pos, key in enumerate(key_order or list())}
        self.pad_token = pad_token
        self.attn_mask_dtype = attn_mask_dtype
        self.keys_with_tokens = keys_with_tokens or {"input_ids", "labels"}
        self.allow_partial_batch_elements = allow_partial_batch_elements

    def __call__(self, inp: list[dict[str, torch.Tensor]]) -> Batch:
        bs = len(inp)
        batch = Batch()
        keys = set(flatten(elem.keys() for elem in inp))
        if self.key_order:
            keys = list(keys)
            keys.sort(key=lambda x: self.key_order.get(x, len(self.key_order)))
        for k in keys:
            if self.allow_partial_batch_elements:
                tensors = [inp[i][k] for i in range(bs) if k in inp[i]]
            else:
                tensors = [inp[i][k] for i in range(bs)]
            batch[k] = self._pad_key(k, tensors, batch, self._pad_token_for_key(k))

        if (
            self.pad_token is not None
            and "attention_mask" not in keys
            and "input_ids" in keys
        ):
            batch["attention_mask"] = (
                batch["input_ids"].ne(self.pad_token).to(self.attn_mask_dtype)
            )
        return batch

    def _pad_token_for_key(self, key: str):
        if key in self.keys_with_tokens and self.pad_token is not None:
            return self.pad_token
        return self.padding

    def _pad_key(
        self, key: str, tensors: list[torch.Tensor], batch: Batch, padding: float
    ) -> torch.Tensor:
        if tensors[0].dim() == 0 or all(tensor.numel() == 1 for tensor in tensors):
            return torch.stack(tensors, dim=0)
        elif tensors[0].dim() >= 1:
            try:
                return pad_sequence(tensors, batch_first=True, padding_value=padding)
            except RuntimeError as rt:
                shapes = list(map(lambda t: t.shape, tensors))
                raise ValueError(
                    f"Could not pad sequence of tensors for key {key}, shapes are: {shapes}"
                ) from rt
        else:
            raise ValueError(key, tensors[0].shape)
