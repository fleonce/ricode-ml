import json
from typing import Sequence

import ijson

from ricode.ml._preprocessing.data_files import DataFile, ViewDataFile
from ricode.utils.imports import is_datasets_available, is_pyarrow_available


def estimate_data_file_size(data_file: DataFile):
    # if we have a view, we assume the indices are a valid descriptor of the size of the underlying file and
    # that they do not over-approximate the size of the file in question
    if (
        data_file.is_view
        and isinstance(data_file, ViewDataFile)
        and data_file.has_indices
    ):
        return len(data_file.indices)

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


def expected_keys_in_data_file(data_file: DataFile) -> Sequence[str]:
    if (
        data_file.is_view
        and isinstance(data_file, ViewDataFile)
        and data_file.has_renamed_fields
    ):
        data_file_without_renames = DataFile(
            data_file.name_or_path,
            data_file.dataset_type,
            data_file.data,
        )
        expected_keys = expected_keys_in_data_file(data_file_without_renames)
        return [data_file.renamed_fields.get(key, key) for key in expected_keys]

    if data_file.dataset_type == "file":
        if data_file.name_or_path.endswith(".parquet"):
            if not is_pyarrow_available():
                raise ValueError("pyarrow is not installed")

            import pyarrow.parquet as pq

            with pq.ParquetFile(data_file.name_or_path) as file:
                return list(file.schema.names)
        elif data_file.name_or_path.endswith(".json"):
            with open(data_file.name_or_path) as f:
                for item in ijson.items(f, "item", buf_size=8 * 1024):
                    return list(item.keys())
        elif data_file.name_or_path.endswith(".jsonl"):
            with open(data_file.name_or_path) as f:
                item = json.loads(f.readline())
                return list(item.keys())
    elif data_file.dataset_type == "flattened":
        raise NotImplementedError("load metadata from safetensors and return keys")
    elif data_file.dataset_type == "safetensors":
        raise NotImplementedError("load metadata from safetensors and return keys")
    elif data_file.dataset_type == "huggingface":
        if not is_datasets_available():
            raise ValueError("datasets is not installed")
        import datasets

        dataset = datasets.load_dataset(data_file.name_or_path, streaming=True)
        return dataset.column_names

    raise NotImplementedError(data_file.dataset_type, data_file.name_or_path)
