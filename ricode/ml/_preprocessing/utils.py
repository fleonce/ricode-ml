import json

from ricode.ml._preprocessing.data_files import DataFile, ViewDataFile
from ricode.utils.imports import is_pyarrow_available


def estimate_data_file_size(data_file: DataFile):
    # if we have a view, we assume the indices are a valid descriptor of the size of the underlying file and
    # that they do not over-approximate the size of the file in question
    if data_file.is_view and isinstance(data_file, ViewDataFile):
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
