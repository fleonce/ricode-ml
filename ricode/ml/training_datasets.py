import functools
import json
import multiprocessing.shared_memory
import os.path
import random
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from random import Random
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
)

import attrs
import torch
from more_itertools import first
from safetensors_dataset import load_safetensors, SafetensorsDataset, SafetensorsDict
from safetensors_dataset.dict_dataset import ShardedSafetensorsDataset
from safetensors_dataset.loading import exists_safetensors
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from ricode.ml.datasets.concatenated import ConcatenatedDataset
from ricode.ml.datasets.index import IndexDataset
from ricode.ml.distributed import distributed_world_size
from ricode.ml.distributed.utils import (
    distributed_barrier,
    distributed_rank,
    is_rank_zero,
    rank_zero_first,
    rank_zero_ordered,
)
from ricode.ml.training_basics import Conf
from ricode.ml.training_utils import cached_property

try:
    import datasets

    DATASETS = True
except ImportError:
    datasets = None
    DATASETS = False
try:
    import posix_ipc

    POSIX_IPC = True
except ImportError:
    posix_ipc = None
    POSIX_IPC = False

Data: TypeAlias = Union[
    SafetensorsDataset, MutableMapping[str, SafetensorsDataset], SafetensorsDict
]
DataDict: TypeAlias = MutableMapping[str, Data]

TAny = TypeVar("TAny")


class SupportsFromPretrained(Protocol):
    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, *inputs, **kwargs):
        pass


@attrs.define()
class SkipExample:
    cause: str

    def __bool__(self):
        return False


@attrs.define(kw_only=True)
class SplitInfo:
    name: str | None = None
    name_or_file: str
    extension: str = ".safetensors"
    types: str | None = None

    @property
    def path(self):
        return Path(self.name_or_file)


ReturnData: TypeAlias = Union[
    Mapping[str, Tensor], list[Mapping[str, Tensor]], Literal[False], SkipExample
]
SetupElementFunc: TypeAlias = Callable[[str, SplitInfo, Any], ReturnData]
SetupSplitFunc: TypeAlias = Callable[
    [str, SplitInfo, Iterable[Any]], SafetensorsDataset | SafetensorsDict
]


def _rank_distributing_generator(
    rank: int,
    world_size: int,
    blocksize: int,
    iterable: Iterable[TAny],
    func: Optional[Callable[[TAny], Any]] = None,
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
        if func:
            yield func(element)
        else:
            yield element


def _readline_memoryview(mv: memoryview, cursor: int) -> tuple[int, memoryview | None]:
    eol = 0x0A
    begin = cursor
    size = len(mv)

    if cursor == size:
        return cursor, None
    while True:
        char = mv[cursor]
        # assert False, (repr(char), eol)
        cursor += 1
        if char == eol or cursor == size:
            return cursor, mv[begin:cursor]


def append_to_map(
    output: MutableMapping[str, list[Tensor]], mapping: Mapping[str, Tensor]
):
    have_grace = not any(len(elem) for elem in output.values())
    for k, v in mapping.items():
        if k not in output:
            if not have_grace:
                raise ValueError(
                    f"Cannot append additional key {k} after keys {output.keys()} were already filled with values"
                )
            output[k] = list()
        output[k].append(v)


def _guess_split_name(value: str):
    if "." in value:
        value = value.split(".")[0]
        if "_" in value:
            return value.split("_")[0]
        return value
    return value


def _resolve_split_info(value: Any):  # value is str or dict[str, str]
    if isinstance(value, str):
        return SplitInfo(name=_guess_split_name(value), name_or_file=value)
    elif not isinstance(value, dict):
        raise ValueError(value)

    try:
        return SplitInfo(**value)
    except TypeError:
        return {key: _resolve_split_info(inner) for key, inner in value.items()}


def _resolve_split_infos(splits: Any):
    return {key: _resolve_split_info(value) for key, value in splits.items()}


def setup_tokenizer(
    tokenizer: Optional[str | PreTrainedTokenizerBase], self: "BasicDataset"
) -> Optional[PreTrainedTokenizerBase]:
    if tokenizer is None:
        return None
    elif isinstance(tokenizer, PreTrainedTokenizerBase):
        return tokenizer
    return self.tokenizer_class.from_pretrained(
        tokenizer,
        model_max_length=self.model_max_length,
        additional_special_tokens=self.additional_special_tokens,
    )


@attrs.define(kw_only=True, repr=False)
class BasicDataset(Conf, Generic[TAny]):
    # a storage prefix used in the filename for the preprocessed data
    storage_prefix: ClassVar[str] = ""
    # the keys of train, test and validation splits
    split_names: ClassVar[set[str]] = {"train", "test", "eval"}

    # when using SafetensorsDataset, should sharding be used?
    use_shards: ClassVar[bool] = True
    # when using SafetensorsDataset, what size should a single stored dataset be allowed to have?
    shard_size: ClassVar[int] = 10000

    # the format of the samples being loaded
    sample_format: ClassVar[str] = "json"
    # whether to always re-create the dataset, disregarding existing on-disk cached files
    force_dataset_setup: ClassVar[bool] = False
    # whether to keep the preprocessed data in memory only or to save it on-disk
    in_memory_only: ClassVar[bool] = False
    # tokenizer class to init the tokenizer with
    tokenizer_class: ClassVar[type[SupportsFromPretrained]] = AutoTokenizer

    if DATASETS:
        streaming_dataset: ClassVar[bool] = False

    # the name of the dataset
    name: str = attrs.field()
    # the directory where the data for this dataset is stored
    data_dir: str = attrs.field()

    # a dictionary of SplitInfo objects regarding where each file for each of the splits is located
    # nested dictionaries imply multiple test or validation splits
    splits: Mapping[str, SplitInfo | dict[str, SplitInfo]] = attrs.field(
        converter=_resolve_split_infos,
    )

    # contains the preprocessed files!
    data: DataDict = attrs.field(factory=OrderedDict)

    # instance field for re-creating the dataset on each setup
    force_setup: bool = attrs.field(
        default=attrs.Factory(lambda self: self.force_dataset_setup, takes_self=True),
    )
    # instance field for keeping the dataset in memory only
    memory_only: bool = attrs.field(
        default=attrs.Factory(
            lambda self: self.in_memory_only,
            takes_self=True,
        )
    )
    # instance field for setting the storage prefix :)
    _storage_prefix: str = attrs.field(
        default=attrs.Factory(
            lambda self: self.storage_prefix,
            takes_self=True,
        )
    )

    model_max_length: Optional[int] = attrs.field(default=None)
    additional_special_tokens: Optional[list[str]] = attrs.field(default=None)
    tokenizer = attrs.field(converter=attrs.Converter(setup_tokenizer, takes_self=True))

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def __getitem__(self, item: str) -> SafetensorsDataset:
        return self.data[item]

    @cached_property
    def tokenizer_config(self) -> PretrainedConfig:
        try:
            return AutoConfig.from_pretrained(self.tokenizer.name_or_path)
        except ValueError as cause:
            warnings.warn(
                f"Caught {cause} when trying to fetch config for {self.tokenizer.name_or_path}"
            )
            raise AttributeError

    def to(self, device: str | int | torch.device):
        def move_to_device(d: DataDict):
            for k, v in d.items():
                if isinstance(v, (SafetensorsDataset, SafetensorsDict)):
                    d[k] = v.to(device)
                else:
                    move_to_device(v)

        move_to_device(self.data)

    def setup_example(self, split: str, split_info: SplitInfo, data: TAny):
        raise NotImplementedError(self.__class__.__name__ + ".setup_example")

    def __repr__(self):
        def format_split(dataset):
            return repr(dataset).replace("\n", "\n  ")

        def format_keys(keys):
            return ", ".join(sorted(map(lambda k: '"' + k + '"', keys)))

        def format_dataset(key, dataset):
            if isinstance(dataset, SafetensorsDataset):
                return (
                    f"  {key}=SafetensorsDataset("
                    f"size={len(dataset)}, "
                    f"keys={format_keys(dataset.keys())})"
                )
            elif isinstance(dataset, ShardedSafetensorsDataset):
                return (
                    f"  {key}=ShardedSafetensorsDataset("
                    f"size={len(dataset)}, "
                    f"num_shards={len(dataset.shards)}, "
                    f"keys={format_keys(dataset.shards[0].keys())}"
                    f")"
                )
            elif isinstance(dataset, Mapping):
                if len(dataset) <= 3 or True:
                    formatted_dataset = [
                        "  " + format_dataset(key, dataset[key])
                        for key in sorted(dataset.keys())
                    ]
                    formatted_dataset = "\n".join(formatted_dataset)
                    return (
                        f"  {key}=SafetensorsDict(" f"splits={{\n{formatted_dataset}}})"
                    )
                else:
                    first_key = first(sorted(dataset.keys()))
                    return (
                        f"  {key}=SafetensorsDict("
                        f"num_splits={len(dataset)}, "
                        f"splits={{{format_dataset(first_key, dataset[first_key])[2:]}}}, ... and {len(dataset) - 1} more)"
                    )
            else:
                return f"  {key}={dataset}"

        info = [f"{self.__class__.__name__}("]
        for split in self.data.keys():
            if split not in self.split_names:
                continue
            info.append(format_dataset(split, self.data[split]))
        return "\n".join(info)

    def inner_setup_examples(
        self,
        split: str,
        split_info: SplitInfo,
        elements: Iterable[Any],
        element_fn: SetupElementFunc,
        **kwargs: Any,
    ) -> SafetensorsDataset:
        output: dict[str, list[Tensor]] = dict()

        if kwargs:
            element_fn = functools.partial(element_fn, **kwargs)

        infos = OrderedDict()
        for pos, item in enumerate(
            tq := tqdm(elements, desc=split_info.name + " " + split)
        ):
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

        def map_without_empty(m: dict):
            return {k: v for k, v in m.items() if len(v) > 0}

        output = map_without_empty(output)
        return SafetensorsDataset(output, preprocess=True)

    @rank_zero_first
    def setup_dataset(self):
        for split in sorted(self.splits.keys()):
            if split not in self.split_names:
                continue

            split_info = self.splits[split]
            if isinstance(split_info, dict):
                self._setup_dict_split(split, split_info)
            else:
                self._setup_split(split, split_info)

    def setup_crossvalidation(self, seed: int):
        flattened_data = OrderedDict(
            {
                (split, name): dataset
                for split, datasets in self.data.items()
                for name, dataset in datasets.items()
            }
        )
        full_dataset = ConcatenatedDataset.from_mapping(flattened_data)
        total_length = len(full_dataset)
        indices = list(range(total_length))
        rand = Random(seed)
        train_indices = set(rand.sample(indices, k=int(total_length * 0.9)))
        test_indices = set(indices) - train_indices

        self.data = {
            "train": {"crossvalidation": IndexDataset(train_indices, full_dataset)},
            "test": {"crossvalidation": IndexDataset(test_indices, full_dataset)},
            "eval": {"crossvalidation": IndexDataset(test_indices, full_dataset)},
        }

    def setup_hpo(self):
        eval_spit = self.data.get("dev", self.data.get("eval", None))
        if eval_spit is None:
            raise ValueError(
                f"Cannot setup hpo, unknown name for eval/dev split: {self.data}"
            )
        self.data["test"] = eval_spit

    def _load_split_data(self, split: str, split_info: SplitInfo):
        if self.sample_format == "huggingface":
            if not DATASETS:
                raise ImportError(
                    'datasets library required, install via "pip install datasets"'
                )
            return datasets.load_dataset(
                self.name,
                split=split_info.name_or_file,
                streaming=self.streaming_dataset,
            )
        elif self.sample_format == "json":
            file_path = self._split_path(split, split_info)
            if not file_path.exists():
                raise FileNotFoundError(file_path)

            if file_path.suffix == ".json":
                with open(file_path) as f:
                    return json.load(f)
            elif file_path.suffix == ".jsonl":

                def _iterator():
                    with open(file_path) as f:
                        while line := f.readline():
                            yield json.loads(line)

                return _iterator()
            else:
                raise ValueError(f"Unknown file suffix {file_path.suffix}")
        else:
            raise ValueError(f"Unknown sample format {self.sample_format!r}")

    def _setup_dict_split(self, split: str, split_infos: dict[str, SplitInfo]):
        datasets = OrderedDict()
        for key in sorted(split_infos.keys()):
            datasets[key] = self._setup_split_base(split, split_infos[key])
        self.data[split] = datasets

    def _setup_split_base(self, split: str, split_info: SplitInfo):

        def exists_dataset(info: SplitInfo):
            if split_info.extension == ".safetensors":
                return exists_safetensors(self._split_path(split, info, is_final=True))
            elif split_info.extension == ".datasets":
                return os.path.exists(self._split_path(split, info, is_final=True))
            else:
                raise ValueError(split_info.extension)

        if self.force_setup or not exists_dataset(split_info):
            # todo: move load split data to this function, setup_examples too! make one larger function
            # so we can do real multiprocessing and not rank zero first, the others follow!
            loaded_data = self._load_split_data(split, split_info)
            dataset = self.inner_setup_examples(
                split, split_info, loaded_data, self.setup_example
            )
            if (
                self.use_shards
                and isinstance(dataset, (SafetensorsDataset, SafetensorsDict))
                and len(dataset) > self.shard_size
            ):
                dataset = dataset.shard(self.shard_size, preprocess_if_unprocessed=True)
            if not self.memory_only:
                self._save_split0(split, split_info, dataset)
                dataset = self._load_split0(split, split_info)
            return dataset
        else:
            return self._load_split0(split, split_info)

    def _setup_split(self, split: str, split_info: SplitInfo):
        self.data[split] = self._setup_split_base(split, split_info)

    def _save_split0(self, split: str, split_info: SplitInfo, dataset: Any):
        split_path = self._split_path(split, split_info, is_final=True)
        if split_info.extension == ".safetensors":
            if not isinstance(
                dataset,
                (SafetensorsDataset, SafetensorsDict, ShardedSafetensorsDataset),
            ):
                raise ValueError
            dataset.save_to_file(split_path)
        elif split_info.extension == ".datasets":
            if DATASETS:
                dataset.save_to_disk(split_path)
            else:
                raise ImportError("datasets is unavailable")

    def _load_split0(self, split: str, split_info: SplitInfo):
        if self.memory_only:
            raise ValueError(f"Cannot load split {split} when {self.memory_only=}")
        if split_info.extension == ".safetensors":
            dataset = load_safetensors(
                self._split_path(split, split_info, is_final=True)
            )
        elif split_info.extension == ".datasets":
            from datasets.load import load_from_disk

            dataset = load_from_disk(self._split_path(split, split_info, is_final=True))
        else:
            raise ValueError(split_info.extension)
        return dataset

    def _split_path(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> Path:
        data_directory = self.data_path
        if not data_directory.exists():
            data_directory.mkdir(parents=True, exist_ok=False)
        return data_directory / self._split_filename(
            split, split_info, is_final=is_final
        )

    def _split_filename_additional_args(
        self, split: str, split_info: SplitInfo
    ) -> list[str]:
        return list()

    def _split_filename(
        self, split: str, split_info: SplitInfo, *, is_final: bool = False
    ) -> str:
        if is_final:
            file_path = split_info.path
            filename = file_path.name[: -len(file_path.suffix)]

            if self.tokenizer is not None:
                tokenizer_name = self.tokenizer_config.model_type
                if tokenizer_name != "t5":
                    filename = filename + "_tk_" + tokenizer_name
            for arg in self._split_filename_additional_args(split, split_info):
                filename = filename + arg
            if self._storage_prefix:
                filename = self._storage_prefix + filename
            if split_info.extension == ".safetensors":
                filename = filename + split_info.extension
            return filename
        return split_info.name_or_file

    def _feature_components(self) -> str:
        features = getattr(self, "features")
        if isinstance(features, int):
            i = 0
            flags = []
            while 2**i <= features:
                if features & (1 << i):
                    flags.append(str(i))
                i += 1
            return ",".join(flags)
        elif isinstance(features, list):
            return ",".join(list(map(str, features)))
        else:
            raise ValueError(features)

    # multiprocessing code!
    def setup_multiprocessing(self, blocksize: int):
        """
        This is an "irreversible" function designed to
        swap out all functions responsible for standard
        preprocessing with their equivalents that enable
        multiprocessing between multiple processes

        Args:
            blocksize (int): The number of examples to process in one BLOCK,
                i.e. the number of contiguous lines in JSONLines format before
                another rank processes the following block. As an example,
                rank zero may process the lines 0-99, rank one 100-199,
                rank zero 200-299 and so on.
        """
        if not POSIX_IPC:
            raise ImportError(
                'posix-ipc is not installed. Install via "pip install posix-ipc"'
            )

        if self.inner_setup_examples is self._multiprocessing_inner_setup_examples:
            # already set up
            return

        self._blocksize = blocksize
        self._world_size = distributed_world_size()
        self._rank = distributed_rank()
        self._load_split_data = self._multiprocessing_load_split_data
        self.inner_setup_examples = self._multiprocessing_inner_setup_examples

    @rank_zero_first
    def _multiprocessing_setup(self):
        flags = 0 if not is_rank_zero() else posix_ipc.O_CREX

        semaphore = posix_ipc.Semaphore("/semaphore", flags, 0o600, 1)
        shared_mem = multiprocessing.shared_memory.SharedMemory(
            "shared_memory", is_rank_zero(), 8
        )
        return semaphore, shared_mem

    def multiprocessing_setup_dataset(self):
        in_rank_zero = is_rank_zero()

        try:
            self.setup_dataset()
        finally:

            def cleanup_shared_memory():
                # post setup, cleanup the shared memory!
                if hasattr(self, "_shared_memory"):
                    for value in getattr(self, "_shared_memory").values():
                        value: multiprocessing.shared_memory.SharedMemory
                        value.close()
                        if in_rank_zero:
                            value.unlink()

            rank_zero_ordered(cleanup_shared_memory, False)

    def _multiprocessing_load_split_data(
        self,
        split: str,
        split_info: SplitInfo,
    ):
        shm_env_var = os.environ.get("MULTIPROCESSING_SHM_NAME", "shared_memory_file")
        shm_name = shm_env_var + "_" + split
        if split_info.name is not None:
            shm_name = shm_name + "_" + split_info.name
        else:
            shm_name = shm_name + "_" + split_info.name_or_file

        file_path = self._split_path(split, split_info)

        shared_mem = None
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
            distributed_barrier()
            shared_mem = multiprocessing.shared_memory.SharedMemory(shm_name, False, 0)
        else:
            # last thing to do is determine the size of the shared memory!
            stat_res = os.stat(file_path)
            shared_mem = multiprocessing.shared_memory.SharedMemory(
                shm_name, True, stat_res.st_size
            )
            with open(self._split_path(split, split_info), "rb") as f:
                fread = f.read
                copied = 0
                size = 64 * 1024
                fwrite = shared_mem.buf
                while True:
                    buf = fread(size)
                    if not buf:
                        break
                    fwrite[copied : copied + len(buf)] = buf
                    copied += len(buf)
            pass
            distributed_barrier()

        if not hasattr(self, "_shared_memory"):
            setattr(self, "_shared_memory", OrderedDict())
        getattr(self, "_shared_memory")[shm_name] = shared_mem

        if file_path.suffix == ".json":
            utf8_content = shared_mem.buf.tobytes().decode("utf-8")
            elements: Sequence[Any] = json.loads(utf8_content)
            return _rank_distributing_generator(
                self._rank, self._world_size, self._blocksize, elements
            )
        elif file_path.suffix == ".jsonl":

            def _parser(memview: memoryview):
                # strip the newline character!
                decoded_str = memview.tobytes().decode("utf-8")[:-1]
                return json.loads(decoded_str)

            def _iterator():
                cursor = 0
                while True:
                    next_cursor, line_memview = _readline_memoryview(
                        shared_mem.buf, cursor
                    )
                    if line_memview is None:
                        break

                    cursor = next_cursor
                    yield line_memview

            return _rank_distributing_generator(
                self._rank, self._world_size, self._blocksize, _iterator(), _parser
            )
        else:
            raise ValueError(
                f"Unknown file extension {file_path.suffix} for {file_path.as_posix()}"
            )

    def _multiprocessing_inner_setup_examples(
        self,
        split: str,
        split_info: SplitInfo,
        elements: Iterable[Any],
        element_fn: SetupElementFunc,
        **kwargs: Any,
    ) -> SafetensorsDataset:
        in_rank_zero = is_rank_zero()

        semaphore, shared_mem = self._multiprocessing_setup()
        shared_buf = shared_mem.buf.cast("L")

        output: dict[str, list[Tensor]] = dict()

        def map_without_empty(m: dict):
            return {k: v for k, v in m.items() if len(v) > 0}

        def _update_progress_bar(bar: tqdm, n: int) -> int:
            semaphore.acquire()
            shared_buf[0] = n = shared_buf[0] + n
            semaphore.release()
            bar.update(n - bar.n)
            return 0

        if kwargs:
            element_fn = functools.partial(element_fn, **kwargs)

        infos = OrderedDict()
        iterator = iter(elements)
        iterator_empty = object()

        distributed_barrier()

        update_diff = 1
        last_update_t = time.time() - random.random()
        items_since_last_update = 0

        block = 0
        with tqdm(desc=f"Rank {self._rank} of {self._world_size}") as tq:
            while True:
                item = next(iterator, iterator_empty)
                if item is None or item is iterator_empty:
                    # None comes from the distributing generator
                    # and is our signal to save the data we have collected so far

                    # shard_number calculation to make the naming deterministic
                    shard_number = self._rank + block * self._world_size

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
                        # before we do that, accumulate the final missing
                        # items from the other processes
                        # todo: maybe this freezes:
                        _update_progress_bar(tq, items_since_last_update)
                        break
                    continue

                items_since_last_update += 1
                if last_update_t + update_diff < time.time():
                    last_update_t = time.time()
                    items_since_last_update = _update_progress_bar(
                        tq, items_since_last_update
                    )

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

        # wait for all procs to complete this split
        distributed_barrier()

        del shared_buf
        if in_rank_zero:
            semaphore.unlink()
            shared_mem.unlink()
        semaphore.close()
        shared_mem.close()
        distributed_barrier()
        return None  # noqa
