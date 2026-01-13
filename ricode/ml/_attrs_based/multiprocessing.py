import multiprocessing.shared_memory
import os
import shutil
from typing import Any, Iterable

from ..distributed import is_rank_zero
from ..training_datasets import SplitInfo

from .dataset import AttrsBasicDataset as BasicDataset


def _rank_distributing_generator(
    rank: int, world_size: int, blocksize: int, iterable: Iterable[Any]
):
    block = 0
    for num, element in enumerate(iterable):
        rank_of_element = (num // blocksize) % world_size
        block_of_element = num // (blocksize * world_size)
        if rank_of_element != rank:
            continue
        if block != block_of_element:
            block = block_of_element
            yield None
        yield element


def _func(self, *args, **kwargs):
    assert False, (self, args, kwargs)


def _patched_load_split_data(
    self: BasicDataset,
    split: str,
    split_info: SplitInfo,
    *,
    rank: int,
    world_size: int,
    blocksize: int,
):
    shm_env_var = os.environ.get("MULTIPROCESSING_SHM_NAME", "shared_memory")
    shm_name = shm_env_var + "_" + split
    if split_info.name is not None:
        shm_name = shm_name + "_" + split_info.name
    else:
        shm_name = shm_name + "_" + split_info.name_or_file

    file_path = self._split_path(split, split_info)

    if not is_rank_zero():
        # to make the whole loading operation more efficient, we load the file on rank zero
        # and share it via shared memory. this should make out the most efficient way
        # to only load a file once and read it into memory, avoiding any unnecessary copies afterward

        # the main questions now are:
        # - how do we make out the naming scheme for the shared memory
        #       -> environment variable + split + split_info.name + split_info.path? would split_info.path vary on different ranks?
        #       -> maybe just env var + split + split_info.name is enough!
        # - how do we determine the size of each shared memory segment
        #       -> use stat to retrieve the size of the file in the filesystem.
        #           file should not be larger (except for compressed files, which we can ignore for the time being)
        # since we are not rank zero, we do not need to calculate the size of the shm and can just attach to it
        shared_mem = multiprocessing.shared_memory.SharedMemory(shm_name, False, 0)
    else:
        # last thing to do is determine the size of the shared memory!
        stat_res = os.stat(file_path)
        shared_mem = multiprocessing.shared_memory.SharedMemory(
            shm_name, True, stat_res.st_size
        )
        with open(self._split_path(split, split_info)) as f:
            shutil.copyfileobj(f, shared_mem.buf)
    raise NotImplementedError

    if file_path.suffix == ".json":
        with (
            open(self._split_path(split, split_info)) as f,
            mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_f,
        ):
            elements: Sequence[Any] = json.load(mmap_f)
            return _rank_distributing_generator(rank, world_size, blocksize, elements)
    elif file_path.suffix == ".jsonl":

        def _iterator():
            with (
                open(file_path) as f,
                mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_f,
            ):
                while line := mmap_f.readline():
                    yield json.loads(line)

        return _rank_distributing_generator(rank, world_size, blocksize, _iterator())
    raise ValueError(
        f"Unknown file extension {file_path.suffix} for {file_path.as_posix()}"
    )


def _patched_setup_dataset(
    self: CoNLL,
):
    in_rank_zero = is_rank_zero()
    if not in_rank_zero:
        distributed_barrier()

    for split in sorted(self.splits.keys()):
        if split not in self.split_names:
            continue

        split_info = self.splits[split]
        if isinstance(split_info, dict):
            self._setup_dict_split(split, split_info)
        else:
            self._setup_split(split, split_info)

    if in_rank_zero:
        distributed_barrier()


def _patched_inner_setup_examples(
    self: CoNLL,
    split: str,
    split_info: SplitInfo,
    elements: Iterable[Any],
    element_fn: SetupElementFunc,
    *,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> SafetensorsDataset:
    output: dict[str, list[Tensor]] = dict()

    def map_without_empty(m: dict):
        return {k: v for k, v in m.items() if len(v) > 0}

    if kwargs:
        element_fn = functools.partial(element_fn, **kwargs)

    infos = OrderedDict()
    iterator = iter(elements)
    iterator_empty = object()

    block = 0
    with tqdm(desc=f"{world_size=}, {rank=}") as tq:
        while True:
            item = next(iterator, iterator_empty)
            if item is None or item is iterator_empty:
                # None comes from the distributing generator
                # and is our signal to save the data we have collected so far

                # shard_number calculation to make the naming deterministic
                shard_number = rank + block * world_size

                # create a shard of the overall dataset
                dataset = SafetensorsDataset(map_without_empty(output), True)
                # derive the part from the path info we have
                split_path = self._split_path(split, split_info, is_final=True)
                split_directory = split_path.with_suffix("")
                split_directory.mkdir(0o700, True, True)
                shard_path = split_directory / f"shard{shard_number}.safetensors"
                dataset.save_to_file(shard_path)
                block += 1
                output = dict()
                if item is iterator_empty:
                    # we found the last item, exit our while True loop
                    break
                continue
            tq.update(1)

            example = element_fn(split, split_info, item)
            if not example or isinstance(example, SkipExample):
                cause = "skipped"
                if isinstance(example, SkipExample):
                    cause = example.cause
                if cause not in infos:
                    infos[cause] = 1
                else:
                    infos[cause] += 1
                tq.set_postfix(infos)
                continue

            if isinstance(example, list) or isinstance(example, tuple):
                for example_elem in example:
                    if isinstance(example_elem, SkipExample):
                        cause = example_elem.cause
                        if cause not in infos:
                            infos[cause] = 1
                        else:
                            infos[cause] += 1
                        tq.set_postfix(infos)
                        continue
                    append_to_map(output, example_elem)
            elif isinstance(example, dict):
                append_to_map(output, example)
            else:
                raise ValueError(type(example))
    return None  # noqa
