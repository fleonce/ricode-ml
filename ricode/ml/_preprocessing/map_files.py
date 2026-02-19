import faulthandler
import functools
import json
import multiprocessing
import os
import signal
import sys
import time
import typing
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

import more_itertools
import multiprocess.managers
import multiprocess.queues

import torch
from more_itertools import first
from safetensors_dataset import load_safetensors, SafetensorsDataset
from tqdm import tqdm

from ricode.ml.datasets.cumulative import CumulativeDataset
from ricode.ml.datasets.dataset import DataFile, Dataset, DatasetDict, load_from_disk
from ricode.utils.imports import is_datasets_available, is_pyarrow_available
from ricode.utils.tempfiles import TemporaryDirectory

_map_return_t: TypeAlias = Mapping[
    str, Union[torch.Tensor, Sequence[int], Sequence[float], Sequence[torch.Tensor]]
]
P = ParamSpec("P")


class MapFunction(Protocol[P]):
    def __call__(
        self, batch: MutableMapping[str, Sequence[Any]], /, **kwargs: P.kwargs
    ) -> _map_return_t:
        pass


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
            _rank_based_batched, strict=False, rank=rank, world_size=world_size
        )
    else:
        batched = more_itertools.batched

    if data_file.dataset_type == "file":
        if data_file.name_or_path.endswith(".parquet"):
            if not is_pyarrow_available():
                raise ValueError("pyarrow is not installed")

            import pyarrow.parquet as pq

            table = pq.read_table(data_file.name_or_path)

            if not column_names:
                column_names = table.column_names

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
            del table
        elif data_file.name_or_path.endswith(".jsonl"):
            with open(data_file.name_or_path) as jsonl_file:
                for line_batch in batched(jsonl_file, batch_size):
                    lod = [json.loads(line) for line in line_batch]

                    if not column_names:
                        column_names = list(lod[0].keys())

                    yield _lod_to_batch(lod, column_names)
            del jsonl_file
        elif data_file.name_or_path.endswith(".json"):
            with open(data_file.name_or_path) as json_file:
                line_jsons = json.load(json_file)
            del json_file

            for lod in batched(line_jsons, batch_size):
                if not column_names:
                    column_names = list(lod[0].keys())

                yield _lod_to_batch(lod, column_names)
        else:
            raise NotImplementedError("Unknown extension of file " + repr(data_file))
    elif data_file.dataset_type == "flattened":
        dataset = CumulativeDataset.from_preprocessed(data_file.name_or_path)

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
    # derive initialization to when we know the keys of the result object
    dataset = None

    for batch in _batches_of_data(
        data_file, column_names, batch_size, rank, world_size
    ):
        this_batch_size = len(first(batch.values()))
        if this_batch_size < batch_size and drop_last:
            # todo: is continue the right choice here?
            #   probably not, since out_queue misses the status update
            continue

        result = fn(batch, **fn_kwargs)
        result_batch_size = len(first(result.values()))

        if dataset_type == "flattened":
            if dataset is None:
                dataset = CumulativeDataset.new_empty(
                    {column_name: 2**20 for column_name in result.keys()}
                )
            for sample in range(result_batch_size):
                dataset.append({key: values[sample] for key, values in result.items()})
            del result, batch

        elif dataset_type == "safetensors":
            if dataset is None:
                dataset = {key: [] for key in result.keys()}

            for key, values in result.items():
                dataset[key].extend(values)

        yield this_batch_size, max(this_batch_size - result_batch_size, 0)

    if dataset_type == "flattened":
        dataset.save_to_disk(out_file)
    elif dataset_type == "safetensors":
        if dataset is None:
            dataset = {}
        dataset = SafetensorsDataset.from_dict(dataset, preprocess=True)

        os.makedirs(out_file, exist_ok=True)
        dataset.save_to_file(os.path.join(out_file, "tensors.safetensors"))


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
    _check_faulthandler()

    while (work := in_queue.get()) is not None:
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
        del data_file, out_file, rank, world_size

    # report this process has finished
    out_queue.put(None)
    return 0


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


def unbatched_function(
    fn: Callable[
        [MutableMapping[str, Any]],
        Mapping[str, torch.Tensor | Sequence[int] | Sequence[float]],
    ]
) -> Callable[[MutableMapping[str, Sequence[Any]]], _map_return_t]:
    def batched_function(batch: MutableMapping[str, Sequence[Any]], **kwargs):
        batch_size = len(first(batch.values()))

        result = OrderedDict()
        for index in range(batch_size):
            _result = fn(
                {key: values[index] for key, values in batch.items()}, **kwargs
            )
            if not _result:
                continue
            for key, value in _result.items():
                if key not in result:
                    result[key] = [value]
                else:
                    result[key].append(value)
        return result

    return batched_function


def map_dict_of_files(
    dict_of_data_files: typing.OrderedDict[
        str, Sequence[str] | Sequence[DataFile] | str
    ],
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
    return_mapped: Literal[False, "lazy", "in-memory"] = "in-memory",
) -> "Optional[DatasetDict]":
    dict_of_data_files = OrderedDict(
        [(k, [v] if isinstance(v, str) else v) for k, v in dict_of_data_files.items()]
    )

    if mode == "to-intermediate" and save_path is None:
        with TemporaryDirectory("exit") as temp_save_path:
            save_path = temp_save_path

    return_dataset = DatasetDict()
    if save_path is None:
        raise ValueError(f"A save path must be supplied when {mode=!r}")

    for split, data_files in dict_of_data_files.items():
        split_path = os.path.join(save_path, split)

        result = map_files(
            data_files,
            fn,
            column_names,
            mode,
            split_path,
            batch_size,
            drop_last,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            return_dataset_type,
            return_mapped,
        )
        if result is not None:
            return_dataset[split] = result

    with open(os.path.join(save_path, "dataset_info.json"), "w") as json_f:
        json.dump({"type": "dict", "splits": list(dict_of_data_files.keys())}, json_f)
    del json_f

    if return_mapped:
        return return_dataset
    return None


_faulthandler_registered = False


def _check_faulthandler():
    global _faulthandler_registered
    if not _faulthandler_registered:
        _faulthandler_registered = True
        faulthandler.register(signal.SIGUSR1, sys.stderr, True, False)


def map_files(
    data_files: str | Sequence[str] | Sequence[DataFile],
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
    return_mapped: Literal[False, "lazy", "in-memory"] = "in-memory",
) -> Union[
    # return_mapped == False
    None,
    # return_mapped == "in-memory", "lazy"
    "Dataset",
]:
    _check_faulthandler()

    if fn_kwargs is None:
        fn_kwargs = OrderedDict()

    if isinstance(data_files, str):
        data_files = [data_files]
    if not isinstance(data_files[0], DataFile):
        data_files = [DataFile(str(data_file)) for data_file in data_files]

    if mode in {"to-memory"}:
        raise NotImplementedError("currently, all steps must be saved on-disk")
    if mode == "to-intermediate" and save_path is None:
        with TemporaryDirectory("exit") as temp_save_path:
            save_path = temp_save_path

    if desc is None:
        desc = f"map({len(data_files)} files)"

    total_size = _estimate_size(data_files)
    if num_proc > 1 and num_proc % workers_per_file != 0:
        raise ValueError(f"{num_proc=} must be divisible by {workers_per_file=}")

    progress_postfix = OrderedDict(
        completed_files=0,
        waiting_processes=0,
        lost=0,
        result_ready=0,
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

            for data_file in data_files:
                for worker_id in range(workers_per_file):
                    in_queue.put(
                        (
                            data_file,
                            dataset_dirs[
                                data_files.index(data_file) * workers_per_file
                                + worker_id
                            ],
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
                finished_processes = 0
                while True:
                    try:
                        status = out_queue.get(timeout=0.05)
                        if status is None:
                            # thread/process is done and has no more work
                            finished_processes += 1
                            progress_postfix["waiting_processes"] = finished_processes
                        elif isinstance(status, tuple):
                            # thread/process has completed X amount of samples
                            advanced, lost = status
                            progress += advanced
                            progress_postfix["lost"] += lost
                            progress_bar.update(advanced)
                        elif isinstance(status, DataFile):
                            # thread/process has completed a data_file
                            progress_postfix["completed_files"] += 1
                        else:
                            raise NotImplementedError(status)
                        jobs_ready = sum(
                            async_result.ready() for async_result in async_results
                        )
                        progress_postfix["result_ready"] = jobs_ready
                        progress_bar.set_postfix(progress_postfix)
                        if finished_processes >= num_proc:
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
                            multiprocessing.context.TimeoutError,
                            multiprocess.context.TimeoutError,
                        ):
                            pass
            finally:
                [async_result.get(timeout=0.05) for async_result in async_results]
            pool.close()
            pool.join()

    with open(os.path.join(save_path, "dataset_info.json"), "w") as info_f:
        json.dump(
            {
                "type": return_dataset_type,
                "data_files": list(map(os.path.basename, dataset_dirs)),
                "world_size": workers_per_file,
                "batch_size": batch_size,
            },
            info_f,
        )
    del info_f

    if return_mapped:
        return load_from_disk(save_path, lazy=(return_mapped == "lazy"))
    return None
