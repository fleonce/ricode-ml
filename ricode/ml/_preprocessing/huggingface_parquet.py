from typing import Callable, NoReturn, Sequence, TYPE_CHECKING

from more_itertools.more import first

from ricode.ml._preprocessing.data_files import DataFile
from ricode.ml.datasets.dataset import DataFileDataset, DatasetDict
from ricode.utils.imports import is_datasets_available

if is_datasets_available() or TYPE_CHECKING:
    import datasets
    import datasets.config
    from datasets import DownloadConfig, DownloadManager, load_dataset

    def load_datasets_dataset(
        name_or_path: str,
        revision: str | None = None,
        download_num_proc: int = 1,
        split: str | None = None,
    ):
        dataset = load_dataset(
            name_or_path,
            revision=revision,
            num_proc=download_num_proc,
            split=split,
        )

        if isinstance(dataset, datasets.DatasetDict):
            return DatasetDict(
                {
                    name: DataFileDataset(
                        [DataFile(name_or_path, "huggingface", split)]
                    )
                    for name, split in dataset.items()
                }
            )

        return DataFileDataset([DataFile(name_or_path, "huggingface", dataset)])

    def load_parquet(
        name_or_path: str,
        revision: str | None = None,
        download_num_proc: int = 1,
        split: str | None = None,
        adjust_data_files_fn: (
            Callable[[Sequence[DataFile]], Sequence[DataFile]] | None
        ) = None,
    ):
        """
        Load a dataset of parquet files from the huggingface.co website. Requires internet access.

        This function currently has two modes, loading a DatasetDict and a ParquetDataset standalone
        """

        builder = datasets.load_dataset_builder(
            name_or_path,
            revision=revision,
        )

        if split is not None:
            data_files = list(sorted(first(builder.config.data_files.values())))
            if adjust_data_files_fn:
                data_files = adjust_data_files_fn(data_files)
        else:
            data_files = dict(builder.config.data_files)
            if adjust_data_files_fn:
                data_files = {
                    key: adjust_data_files_fn(value)
                    for key, value in data_files.items()
                }

        cache_downloaded_dir = datasets.config.DOWNLOADED_DATASETS_PATH
        download_manager = DownloadManager(
            name_or_path,
            None,
            DownloadConfig(
                cache_dir=cache_downloaded_dir,
                force_download=False,
                force_extract=False,
                use_etag=False,
                num_proc=download_num_proc,
                token=None,
                storage_options={},
            ),
            base_path=None,
            record_checksums=False,
        )

        def convert(data_file_paths: Sequence[str]):
            return list(map(lambda p: DataFile(p, "file", None), data_file_paths))

        if isinstance(data_files, Sequence):
            data_files = download_manager.download_and_extract(data_files)
            data_files = convert(data_files)
            return DataFileDataset(data_files)
        else:
            data_files = {
                split: download_manager.download_and_extract(split_data_files)
                for split, split_data_files in data_files.items()
            }

            return DatasetDict(
                {
                    split: DataFileDataset(convert(downloaded_split_data_files))
                    for split, downloaded_split_data_files in data_files.items()
                }
            )

else:

    def load_parquet(*args, **kwargs) -> NoReturn:
        raise ImportError("datasets is not installed")

    def load_datasets_dataset(*args, **kwargs) -> NoReturn:
        raise ImportError("datasets is not installed")
