import os.path
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar, Generator, Mapping

import attrs
from attrs import validators
from more_itertools import first

from ..ml.training_basics import attrs_conf_to_mapping
from ..utils.decorators import attrs_to_json
from ..utils.json_files import (
    iterate_json_file_type,
    load_json_file_type,
    save_json_file_type,
)

from .utils import (
    calculate_sha1,
    check_integrity,
    DownloadInfo,
    DownloadOption,
    is_package_available,
)


@attrs.define
class SplitMetadata:
    # a mapping from filename to sha1 hash
    data_files: Mapping[str, str] = attrs.field(
        converter=attrs.Converter(lambda inp: OrderedDict(**inp)),
        validator=validators.deep_mapping(
            key_validator=validators.instance_of(str),
            value_validator=validators.instance_of(str),
            mapping_validator=validators.instance_of(OrderedDict),
        ),
    )


def convert_values_if_not_instance_of(type: Any) -> attrs.Converter:
    def _convert(inp):
        inputs = []
        for key, value in inp.items():
            if not isinstance(value, type):
                value = type(**value)
            inputs.append((key, value))
        return OrderedDict(inputs)

    return attrs.Converter(_convert)


@attrs_to_json
@attrs.define
class DatasetMetadata:
    # a mapping from filename to sha1 hash
    data_files: Mapping[str, str] = attrs.field(
        converter=attrs.Converter(lambda inp: OrderedDict(inp)),
        validator=validators.deep_mapping(
            key_validator=validators.instance_of(str),
            value_validator=validators.instance_of(str),
            mapping_validator=validators.instance_of(OrderedDict),
        ),
    )
    splits: Mapping[str, SplitMetadata] = attrs.field(
        converter=convert_values_if_not_instance_of(SplitMetadata)
    )


class TextDataset:
    data_dir: ClassVar[str]
    splits: ClassVar[list[str]]
    required_packages: ClassVar[list[str]] = []
    defer_download: ClassVar[bool] = False

    def __init__(
        self,
        root: str | Path,
        # true = always, None = if missing, false = never
        download: DownloadOption = DownloadOption.IF_MISSING_OR_OUT_OF_DATE,
    ):
        self.base_dir = os.path.join(root, self.data_dir)
        self.download_choice = download
        self._metadata = None

        if self.required_packages:
            unavailable_packages = []
            for package_name in self.required_packages:
                if not is_package_available(package_name):
                    unavailable_packages.append(package_name)

            if len(unavailable_packages) > 0:
                required_packages_str = ", ".join(map(repr, self.required_packages))
                unavailable_packages_str = ", ".join(map(repr, unavailable_packages))
                raise ValueError(
                    f"{self.__class__.__name__} requires the following packages to be installed: "
                    f"{required_packages_str}. The following packages could not be found: {unavailable_packages_str}"
                )

        if not self.defer_download:
            self._init_download()

    def _init_download(self):
        if self.download_choice == DownloadOption.ALWAYS or (
            not self._check_integrity()
            and self.download_choice == DownloadOption.IF_MISSING_OR_OUT_OF_DATE
        ):
            self.download_and_update_metadata()
        elif (
            self.download_choice == DownloadOption.NEVER and not self._check_integrity()
        ):
            raise ValueError(
                f"Missing or outdated files with download option {self.download_choice!r}"
            )

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is not None:
            return self._metadata

        metadata = load_json_file_type(self.metadata_file)
        self._metadata = DatasetMetadata(**metadata)
        return DatasetMetadata(**metadata)

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.base_dir, "metadata.json5")

    def download_and_update_metadata(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

        download_info = self.download()

        data_file_hashes = OrderedDict()
        for data_file in download_info.data_files:
            data_file_hashes[data_file] = calculate_sha1(
                os.path.join(self.base_dir, data_file)
            )

        split_data_files_hashes = OrderedDict()
        for split in sorted(download_info.splits.keys()):
            split_hashes = OrderedDict()
            for data_file in download_info.splits[split]:
                split_hashes[data_file] = calculate_sha1(
                    os.path.join(self.base_dir, data_file)
                )
            split_data_files_hashes[split] = SplitMetadata(split_hashes)

        metadata = self._metadata = DatasetMetadata(
            data_file_hashes, split_data_files_hashes
        )

        save_json_file_type(attrs_conf_to_mapping(metadata), self.metadata_file)

    def download(self) -> DownloadInfo:
        raise NotImplementedError(f"{self.__class__.__name__}.download(self)")

    def _check_integrity(self) -> bool:
        if not os.path.exists(self.metadata_file):
            # metadata is created post download and stores filenames and their corresponding hashes
            return False

        for split_name in self.splits:
            if split_name not in self.metadata.splits:
                # split names are ouf of sync
                return False
            for filename, expected_sha1 in self.metadata.splits[
                split_name
            ].data_files.items():
                file_path = os.path.join(self.base_dir, filename)
                if not check_integrity(file_path, expected_sha1):
                    # file is either missing or hash does not match
                    return False

        for filename, expected_sha1 in self.metadata.data_files.items():
            file_path = os.path.join(self.base_dir, filename)
            if not check_integrity(file_path, expected_sha1):
                # file is either missing or hash does not match
                return False

        return True

    def __repr__(self):
        split_names = ", ".join(list(sorted(self.metadata.splits.keys())))
        return f"{self.__class__.__name__}(splits={split_names})"

    def __getitem__(self, item: str) -> Generator[Any, None, None]:
        metadata = self.metadata
        if item not in metadata.splits:
            raise KeyError(
                f"Unknown split named {item} for splits {list(metadata.splits.keys())}"
            )

        if len(metadata.splits[item].data_files) == 1:
            data_file = first(metadata.splits[item].data_files.keys())
            data_file = os.path.join(self.base_dir, data_file)
            yield from iterate_json_file_type(data_file)
        else:
            data_files = list(sorted(metadata.splits[item].data_files.keys()))
            data_files = [
                os.path.join(self.base_dir, data_file) for data_file in data_files
            ]

            data_generators = map(iterate_json_file_type, data_files)
            for sample in zip(*data_generators):
                yield sample
