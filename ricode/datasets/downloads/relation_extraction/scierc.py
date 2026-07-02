import os.path
from pathlib import Path
from typing import Literal

from ricode.datasets.text import TextDataset
from ricode.datasets.utils import download_file, DownloadInfo, DownloadOption


class SciERC(TextDataset):
    data_dir = "scierc"
    splits = ["dev", "test", "train"]

    data_files = {
        "train": (
            "scierc_train.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/scierc_train.json",
        ),
        "train+dev": (
            "scierc_train_dev.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/scierc_train_dev.json",
        ),
        "dev": (
            "scierc_dev.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/scierc_dev.json",
        ),
        "test": (
            "scierc_test.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/scierc_test.json",
        ),
        "types": (
            "scierc_types.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/scierc_types.json",
        ),
    }

    def __init__(
        self,
        root: str | Path,
        download: DownloadOption = DownloadOption.IF_MISSING_OR_OUT_OF_DATE,
        train_file: Literal["train", "train+dev"] = "train+dev",
    ):
        self.train_file = train_file

        super().__init__(root, download)

    def download(self) -> DownloadInfo:
        download_info = DownloadInfo()
        for name, (local_file, remote_url) in self.data_files.items():
            file_path = os.path.join(self.base_dir, local_file)

            download_file(file_path, remote_url)
            if name == "types":
                download_info.data_files.append(local_file)
            elif name in {"train", "train+dev"}:
                if name != self.train_file:
                    continue
                else:
                    download_info.splits["train"] = [local_file]
            else:
                download_info.splits[name] = [local_file]

        return download_info
