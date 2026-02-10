import functools
import json
import multiprocessing
import os
import time
from collections import OrderedDict
from queue import Empty
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeAlias,
    Union,
)

import attrs
import more_itertools
import multiprocess.queues

import torch
from more_itertools import first
from safetensors_dataset import load_safetensors
from tqdm import tqdm

from ricode.ml.datasets.cumulative import CumulativeDataset
from ricode.ml.datasets.distributed import DistributedDataset
from ricode.utils.imports import is_datasets_available, is_pyarrow_available

_map_return_t: TypeAlias = Mapping[
    str, Union[torch.Tensor, Sequence[int], Sequence[float], Sequence[torch.Tensor]]
]
P = ParamSpec("P")


class MapFunction(Protocol[P]):
    def __call__(
        self, batch: MutableMapping[str, Sequence[Any]], /, **kwargs: P.kwargs
    ) -> _map_return_t:
        pass


@attrs.define
class DataFile:
    name_or_path: str
    dataset_type: Literal["huggingface", "flattened", "safetensors", "file"] = "file"


class LazyBatch(Mapping[str, Any]):
    def __init__(self, data: MutableMapping[str, Any]):
        self.data = data
        self.keys_to_format = set(data.keys())

    def __getitem__(self, key, /):
        value = self.data[key]
        if key in self.keys_to_format:
            self.data[key] = value = tuple(element.as_py() for element in value)
            self.keys_to_format.remove(key)
        return value

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def _rank_based_distribution(iterable: Iterable[Any], rank: int, world_size: int):
    for idx, data in enumerate(iterable):
        rank_idx = idx % world_size
        # 0->0, 1->1, 2->0, 3->1, 4->0, ...
        if rank_idx == rank:
            yield data


def _rank_based_batched(
    iterable: Sequence[Any], n: int, strict: bool, rank: int, world_size: int
):
    return _rank_based_distribution(
        more_itertools.batched(iterable, n, strict=strict), rank, world_size
    )


# list of dicts to batch/dict of lists
def _lod_to_batch(lod, column_names) -> MutableMapping[str, Sequence[Any]]:
    return OrderedDict(
        {
            column_name: [lod[pos][column_name] for pos in range(len(lod))]
            for column_name in column_names
        }
    )


def _batches_of_data(
    data_file: DataFile,
    column_names: Sequence[str],
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> Generator[LazyBatch | MutableMapping[str, Sequence[Any]], None, None]:
    if world_size > 1:
        batched = functools.partial(
            _rank_based_batched, rank=rank, world_size=world_size
        )
    else:
        batched = more_itertools.batched

    if data_file.dataset_type == "file":
        if data_file.name_or_path.endswith(".parquet"):
            if not is_pyarrow_available():
                raise ValueError("pyarrow is not installed")
            import pyarrow.parquet as pq

            table = pq.read_table(data_file.name_or_path)

            for batch in zip(
                *[
                    batched(table[column_name], n=batch_size, strict=False)
                    for column_name in column_names
                ]
            ):
                yield LazyBatch(
                    {
                        column_name: batch[pos]
                        for pos, column_name in enumerate(column_names)
                    }
                )
        elif data_file.name_or_path.endswith(".jsonl"):
            with open(data_file.name_or_path) as jsonl_file:
                for line_batch in batched(jsonl_file):
                    lod = [json.loads(line) for line in line_batch]
                    yield _lod_to_batch(lod, column_names)
        elif data_file.name_or_path.endswith(".json"):
            with open(data_file.name_or_path) as json_file:
                line_jsons = json.load(json_file)

            for lod in batched(line_jsons):
                yield _lod_to_batch(lod, column_names)
        else:
            raise NotImplementedError("Unknown extension of file " + repr(data_file))
    elif data_file.dataset_type == "flattened":
        with open(os.path.join(data_file.name_or_path, "dataset_info.json")):
            dataset = DistributedDataset.from_preprocessed(data_file.name_or_path)

        for indices in batched(range(len(dataset)), batch_size):
            yield _lod_to_batch(
                [dataset[index] for index in indices],
                column_names,
            )
    elif data_file.dataset_type == "safetensors":
        dataset = load_safetensors(data_file.name_or_path)

        for indices in batched(range(len(dataset)), batch_size):
            yield _lod_to_batch(
                [dataset[index] for index in indices],
                column_names,
            )
    elif data_file.dataset_type == "huggingface":
        if not is_datasets_available():
            raise ValueError("datasets is not installed")

        from datasets import load_dataset

        dataset = load_dataset(data_file.name_or_path)
        size = len(dataset)
        for indices in batched(range(size), batch_size):
            yield dataset[indices[0] : indices[-1]]
    else:
        raise NotImplementedError(data_file.dataset_type)


def _map_data_file(
    fn: MapFunction[P] | Callable[[MutableMapping[str, Sequence[Any]]], _map_return_t],
    data_file: DataFile,
    out_file: str,
    column_names: Sequence[str],
    fn_kwargs: Optional[MutableMapping[str, Any]],
    batch_size: int = 1000,
    drop_last: bool = False,
    mode: Literal["to-disk", "to-intermediate", "to-memory"] = "to-disk",
    dataset_type: Literal["flattened", "safetensors"] = "flattened",
    rank: int = 0,
    world_size: int = 1,
) -> Generator[tuple[int, int], None, None]:
    dataset = None

    for batch in _batches_of_data(
        data_file, column_names, batch_size, rank, world_size
    ):
        this_batch_size = len(first(batch.values()))
        if this_batch_size < batch_size and drop_last:
            # todo: is continue the right choice here?
            #   probably not, since out_queue misses the status update
            continue
        if True:
            yield this_batch_size, 0
            continue

        result = fn(batch, **fn_kwargs)

        if dataset is None:
            if dataset_type == "flattened":
                dataset = CumulativeDataset.new_empty(
                    {column_name: 2**16 for column_name in result.keys()}
                )
            else:
                raise NotImplementedError(dataset_type)

        this_result_size = len(first(result.values()))
        for sample in range(this_result_size):
            dataset.append({key: values[sample] for key, values in result.items()})
        del result, batch
        yield this_batch_size, max(this_batch_size - this_result_size, 0)

    return
    dataset.save_to_disk(out_file)


def _map_worker(
    fn,
    in_queue: Union[multiprocess.queues.Queue, multiprocessing.Queue],
    out_queue: Union[multiprocess.queues.Queue, multiprocessing.Queue],
    fn_kwargs: Optional[Mapping[str, Any]],
    column_names: Sequence[str],
    batch_size: int,
    drop_last: bool,
    mode: Literal["to-disk", "to-intermediate", "to-memory"],
    dataset_type: Literal["flattened", "safetensors"],
):
    while work := in_queue.get():
        data_file, out_file, rank, world_size = work
        for status in _map_data_file(
            fn,
            data_file,
            out_file,
            column_names,
            fn_kwargs,
            batch_size,
            drop_last,
            mode,
            dataset_type,
            rank,
            world_size,
        ):
            out_queue.put(status)
        if rank == 0:
            out_queue.put(data_file)

    # report this process has finished
    out_queue.put(None)


def _estimate_size(data_files: Sequence[DataFile]):
    def _estimate_item(data_file: DataFile):
        if data_file.dataset_type == "file":
            if data_file.name_or_path.endswith(".parquet"):
                if not is_pyarrow_available():
                    raise ValueError("pyarrow is not installed")

                import pyarrow.parquet as pq

                with pq.ParquetFile(data_file.name_or_path) as file:
                    return file.metadata.num_rows
            elif data_file.name_or_path.endswith(".jsonl"):
                return 0
            elif data_file.name_or_path.endswith(".json"):
                with open(data_file.name_or_path) as json_file:
                    return len(json.load(json_file))
        return 0

    return sum(map(_estimate_item, data_files)) or None


def map_to_disk(
    data_files: Sequence[str] | Sequence[DataFile],
    fn: MapFunction[P] | Callable[[MutableMapping[str, Sequence[Any]]], _map_return_t],
    column_names: Sequence[str],
    save_path: Optional[str] = None,
    batch_size: int = 1000,
    drop_last: bool = False,
    desc: Optional[str] = None,
    num_proc: int = 1,
    workers_per_file: int = 1,
    fn_kwargs: Optional[Mapping[str, Any]] = None,
    multiprocessing_mode: Literal["process", "threads"] = "process",
) -> Union[None]:
    return map_files(
        data_files,
        fn,
        column_names,
        "to-disk",
        save_path,
        batch_size,
        drop_last,
        desc,
        num_proc,
        workers_per_file,
        fn_kwargs,
        multiprocessing_mode,
    )


def map_files(
    data_files: Sequence[str] | Sequence[DataFile],
    fn: MapFunction[P] | Callable[[MutableMapping[str, Sequence[Any]]], _map_return_t],
    column_names: Sequence[str],
    mode: Literal["to-disk", "to-intermediate", "to-memory"] = "to-disk",
    save_path: Optional[str] = None,
    batch_size: int = 1000,
    drop_last: bool = False,
    desc: Optional[str] = None,
    num_proc: int = 1,
    workers_per_file: int = 1,
    fn_kwargs: Optional[Mapping[str, Any]] = None,
    multiprocessing_mode: Literal["process", "threads"] = "process",
    return_dataset_type: Literal["flattened", "safetensors"] = "flattened",
) -> Union[
    # to-disk -> None
    None,
    # unsupported at the moment:
    # to-intermediate -> ???
    # to-memory -> ???
]:
    if not isinstance(data_files[0], DataFile):
        data_files = [DataFile(str(data_file)) for data_file in data_files]

    if desc is None:
        desc = f"map({len(data_files)} files)"

    total_size = _estimate_size(data_files)
    if num_proc > 1 and num_proc % workers_per_file != 0:
        raise ValueError(f"{num_proc=} must be divisible by {workers_per_file=}")

    progress_postfix = OrderedDict(
        completed_files=0,
        waiting_processes=0,
        lost=0,
    )

    progress_bar = tqdm(
        desc=desc,
        total=total_size,
        unit=" samples",
        postfix=progress_postfix,
    )

    if num_proc == 1:
        dataset_dirs = [
            os.path.join(save_path, f"data{i}") for i in range(len(data_files))
        ]
        for data_file, dataset_dir in zip(data_files, dataset_dirs):
            for num_processed, num_lost in _map_data_file(
                fn,
                data_file,
                dataset_dir,
                column_names,
                fn_kwargs,
                batch_size,
                drop_last,
                mode,
                return_dataset_type,
            ):
                progress_postfix["lost"] += num_lost
                progress_bar.set_postfix(progress_postfix, refresh=False)
                progress_bar.update(num_processed)
        with open(os.path.join(save_path, "dataset_info.json"), "w") as info_f:
            json.dump(
                {
                    "type": "cumulative",
                    "data_files": list(map(os.path.dirname, dataset_dirs)),
                    "batch_size": fn_kwargs["batch_size"],
                },
                info_f,
            )
        del info_f
    else:
        dataset_dirs = [
            os.path.join(save_path, f"data{i}_rank{rank}")
            for i in range(len(data_files))
            for rank in range(workers_per_file)
        ]

        with (
            progress_bar,
            (
                multiprocessing.Pool(num_proc)
                if multiprocessing_mode == "threads"
                else multiprocess.pool.Pool(num_proc)
            ) as pool,
            (
                multiprocessing.Manager()
                if multiprocessing_mode == "threads"
                else multiprocess.managers.SyncManager()
            ) as manager,
        ):
            in_queue = manager.Queue()
            out_queue = manager.Queue()

            for data_file, dataset_dir in zip(data_files, dataset_dirs):
                for worker_id in range(workers_per_file):
                    in_queue.put(
                        (
                            data_file,
                            dataset_dir,
                            worker_id,
                            workers_per_file,
                        )
                    )
            for _ in range(num_proc):
                # tell the processes that there are no more files
                in_queue.put(None)

            args = (
                fn,
                in_queue,
                out_queue,
                fn_kwargs,
                column_names,
                batch_size,
                drop_last,
                mode,
                return_dataset_type,
            )

            async_results = [
                pool.apply_async(_map_worker, args) for _ in range(num_proc)
            ]

            check_t = time.time()
            progress = 0
            try:
                while True:
                    try:
                        status = out_queue.get(timeout=0.05)
                        if status is None:
                            # thread/process is done and has no more work
                            progress_postfix["waiting_processes"] += 1
                            progress_bar.set_postfix(progress_postfix)
                        elif isinstance(status, tuple):
                            # thread/process has completed X amount of samples
                            advanced, lost = status
                            progress += advanced
                            progress_postfix["lost"] += lost
                            progress_bar.set_postfix(progress_postfix)
                            progress_bar.update(advanced)
                        elif isinstance(status, str):
                            # thread/process has completed a data_file
                            progress_postfix["completed_files"] += 1
                            progress_bar.set_postfix(progress_postfix)
                        else:
                            raise NotImplementedError(status)
                        jobs_done = all(
                            [async_result.ready() for async_result in async_results]
                        )
                        if progress >= total_size and jobs_done:
                            break
                    except Empty:
                        pass

                    if check_t + 1 < time.time():
                        check_t = time.time()
                        try:
                            [
                                async_result.get(timeout=0.05)
                                for async_result in async_results
                            ]
                        except (
                            TimeoutError,
                            multiprocess.context.TimeoutError,
                        ):
                            pass
            finally:
                [async_result.get(timeout=0.05) for async_result in async_results]

        with open(os.path.join(save_path, "dataset_info.json"), "w") as info_f:
            json.dump(
                {
                    "type": "distributed_cumulative",
                    "data_files": list(map(os.path.dirname, dataset_dirs)),
                    "world_size": workers_per_file,
                    "batch_size": batch_size,
                },
                info_f,
            )
        del info_f
