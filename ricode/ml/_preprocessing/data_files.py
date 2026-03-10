from typing import Literal, Sequence

import attrs


@attrs.define(hash=True)
class DataFile:
    name_or_path: str
    dataset_type: Literal["huggingface", "flattened", "safetensors", "file"] = "file"
    data: object = attrs.field(default=None, hash=False)

    @property
    def is_view(self):
        return False


@attrs.define(hash=True)
class ViewDataFile(DataFile):
    @staticmethod
    def view_from(data_file: DataFile, indices: Sequence[int]):
        return ViewDataFile(
            data_file.name_or_path, data_file.dataset_type, data_file.data, indices
        )

    indices: Sequence[int] = attrs.field(factory=list)

    @property
    def is_view(self):
        return True
