from typing import Literal, Mapping, Optional, Sequence

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

    indices: Optional[Sequence[int]] = None
    renamed_fields: Optional[Mapping[str, str]] = None

    @property
    def is_view(self):
        return True

    @property
    def has_indices(self):
        return self.indices is not None

    @property
    def has_renamed_fields(self):
        return self.renamed_fields is not None
