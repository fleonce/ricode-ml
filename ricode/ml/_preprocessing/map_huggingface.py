from typing import Any, Literal, Mapping, Optional, Sequence

import datasets

from ricode.ml._preprocessing.data_files import DataFile
from ricode.ml._preprocessing.map_files import map_files


def map_huggingface_dataset(
    dataset: datasets.Dataset,
    fn,
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
    return_mapped: Literal["lazy", "in-memory"] = "in-memory",
):
    return map_files(
        [DataFile(dataset.config_name, "huggingface", dataset)],
        fn,
        column_names,
        mode,
        save_path,
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
