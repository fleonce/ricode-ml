import multiprocessing
import operator
import sys
import time
import typing
from collections import OrderedDict
from queue import Empty
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

import attrs
import multiprocess.managers
import multiprocess.queues
from more_itertools import first
from tqdm import tqdm

from ricode.ml._preprocessing.map_files import (
    _batches_of_data,
    _check_faulthandler,
    _estimate_size,
)
from ricode.ml.datasets.dataset import DataFile

ReductionFunction = Union[
    Callable[[MutableMapping[str, Sequence[Any]]], Union[int, float]]
]
ReductionLiteral: TypeAlias = Literal["sum", "mean", "min", "max"]


@attrs.define
class ReductionResult:
    data_file: DataFile
    rank: int
    value: int | float


def _reduce_default(reduction: ReductionLiteral):
    if reduction in {"sum", "mean"}:
        return 0
    elif reduction == "min":
        return sys.maxsize
    elif reduction == "max":
        return -sys.maxsize
    else:
        raise NotImplementedError(reduction)


def _reduce_data_file(
    fn: ReductionFunction,
    data_file: DataFile,
    column_names: Sequence[str],
    fn_kwargs: Optional[MutableMapping[str, Any]],
    batch_size: int,
    rank: int,
    world_size: int,
    reduce_fn: Callable[[int | float, int | float], int | float],
    reduction: Literal["sum", "mean", "min", "max"],
) -> Generator[tuple[int, int | float], None, None]:
    reduction_result = _reduce_default(reduction)
    for batch in _batches_of_data(
        data_file, column_names, batch_size, rank, world_size
    ):
        this_batch_size = len(first(batch.values()))

        result = fn(batch, **fn_kwargs)
        reduction_result = reduce_fn(reduction_result or 0, result)

        yield this_batch_size, reduction_result
    return None


def _reduce_worker(
    fn,
    in_queue: Union[multiprocess.queues.Queue, multiprocessing.Queue],
    out_queue: Union[multiprocess.queues.Queue, multiprocessing.Queue],
    fn_kwargs: Optional[Mapping[str, Any]],
    column_names: Sequence[str],
    batch_size: int,
    reduce_fn,
    reduction: ReductionLiteral,
):
    _check_faulthandler()

    while (work := in_queue.get()) is not None:
        data_file, rank, world_size = work
        result = _reduce_default(reduction)
        for status, result in _reduce_data_file(
            fn,
            data_file,
            column_names,
            fn_kwargs,
            batch_size,
            rank,
            world_size,
            reduce_fn,
            reduction,
        ):
            out_queue.put((status, ReductionResult(data_file, rank, result)))
        out_queue.put(ReductionResult(data_file, rank, result))
        del data_file, rank, world_size

    # report this process has finished
    out_queue.put(None)
    return 0


def reduce_dict_of_files(
    dict_of_data_files: typing.OrderedDict[str, Sequence[str] | Sequence[DataFile]],
    fn: ReductionFunction,
    column_names: Sequence[str],
    batch_size: int = 1000,
    desc: Optional[str] = None,
    num_proc: int = 1,
    workers_per_file: int = 1,
    fn_kwargs: Optional[Mapping[str, Any]] = None,
    multiprocessing_mode: Literal["process", "threads"] = "process",
    reduction: Literal["sum", "mean", "min", "max"] = "sum",
    progress_bar: bool = True,
) -> OrderedDict[str, int | float]:
    reduction_result = OrderedDict()

    for split, data_files in dict_of_data_files.items():
        result = reduce_files(
            data_files,
            fn,
            column_names,
            batch_size,
            desc,
            num_proc,
            workers_per_file,
            fn_kwargs,
            multiprocessing_mode,
            reduction,
            progress_bar,
        )
        if result is not None:
            reduction_result[split] = result
    return reduction_result


def reduce_files(
    data_files: Sequence[str] | Sequence[DataFile],
    fn: ReductionFunction,
    column_names: Sequence[str],
    batch_size: int = 1000,
    desc: Optional[str] = None,
    num_proc: int = 1,
    workers_per_file: int = 1,
    fn_kwargs: Optional[Mapping[str, Any]] = None,
    multiprocessing_mode: Literal["process", "threads"] = "process",
    reduction: Literal["sum", "mean", "min", "max"] = "sum",
    progress_bar: bool = True,
) -> Union[int, float]:
    _check_faulthandler()

    if fn_kwargs is None:
        fn_kwargs = OrderedDict()

    if not isinstance(data_files[0], DataFile):
        data_files = [DataFile(str(data_file)) for data_file in data_files]

    if desc is None:
        desc = f"reduce({len(data_files)} files, reduction={reduction})"

    if num_proc > 1 and num_proc % workers_per_file != 0:
        raise ValueError(f"{num_proc=} must be divisible by {workers_per_file=}")

    post_reduce_fn = lambda x, num_samples: x  # noqa: E731
    if reduction == "sum":
        reduce_fn = operator.add
    elif reduction == "mean":
        reduce_fn = operator.add
        post_reduce_fn = lambda x, num_samples: x / num_samples  # noqa: E731
    elif reduction == "min":
        reduce_fn = min
    elif reduction == "max":
        reduce_fn = max
    else:
        raise NotImplementedError(reduction)

    total_size = _estimate_size(data_files)

    progress_postfix = OrderedDict(
        completed_files=0,
        waiting_processes=0,
        result_ready=0,
    )

    progress_bar = tqdm(
        desc=desc,
        total=total_size,
        unit=" samples",
        postfix=progress_postfix,
        disable=not progress_bar,
    )

    reduce_result = _reduce_default(reduction)
    if num_proc == 1:
        for data_file in data_files:
            data_file_reduce_result = _reduce_default(reduction)
            for num_processed, data_file_reduce_result in _reduce_data_file(
                fn,
                data_file,
                column_names,
                fn_kwargs,
                batch_size,
                0,
                1,
                reduce_fn,
                reduction,
            ):
                temp_result = post_reduce_fn(
                    reduce_fn(reduce_result, data_file_reduce_result),
                    progress_bar.n + num_processed,
                )
                progress_bar.set_description(desc[:-1] + f", result={temp_result}")
                progress_bar.set_postfix(progress_postfix, refresh=False)
                progress_bar.update(num_processed)
            reduce_result = reduce_fn(reduce_result, data_file_reduce_result)
    else:
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
                reduce_fn,
                reduction,
            )

            async_results = [
                pool.apply_async(_reduce_worker, args) for _ in range(num_proc)
            ]

            check_t = time.time()
            progress = 0
            reduction_results = {
                (data_file, worker_id): _reduce_default(reduction)
                for data_file in data_files
                for worker_id in range(workers_per_file)
            }

            def compute_result():
                _result = _reduce_default(reduction)
                for value in reduction_results.values():
                    _result = reduce_fn(_result, value)
                _result = post_reduce_fn(_result, progress_bar.n)
                return _result

            def update_desc():
                _result = compute_result()

                progress_bar.set_description(desc[:-1] + f", result={_result})")

            try:
                finished_processes = 0
                while True:
                    try:
                        status = out_queue.get(timeout=0.05)
                        if status is None:
                            # thread/process is done and has no more work
                            finished_processes += 1
                            progress_postfix["waiting_processes"] = finished_processes
                            progress_bar.set_postfix(progress_postfix)
                        elif isinstance(status, tuple):
                            # thread/process has completed X amount of samples
                            advanced, temp_reduce_result = status
                            reduction_results[
                                (temp_reduce_result.data_file, temp_reduce_result.rank)
                            ] = temp_reduce_result.value
                            progress += advanced
                            progress_bar.set_postfix(progress_postfix)
                            progress_bar.update(advanced)
                            update_desc()
                        elif isinstance(status, ReductionResult):
                            # thread/process has completed a data_file
                            progress_postfix["completed_files"] += 1
                            progress_bar.set_postfix(progress_postfix)
                            reduction_results[(status.data_file, status.rank)] = (
                                status.value
                            )
                        else:
                            raise NotImplementedError(status)
                        jobs_ready = sum(
                            async_result.ready() for async_result in async_results
                        )
                        progress_postfix["result_ready"] = jobs_ready
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
            reduce_result = compute_result()

    return reduce_result
