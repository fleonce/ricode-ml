import os.path
from pathlib import Path
from typing import Literal

from ricode.datasets.text import TextDataset
from ricode.datasets.utils import download_file, DownloadInfo, DownloadOption


class AdverseDrugEvents(TextDataset):
    data_dir = "adverse-drug-events"
    splits = ["dev", "test", "train"]
    defer_download = True

    data_files = {
        "train": (
            "ade_split_{split}_train.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ade_split_{split}_train.json",
        ),
        "test": (
            "ade_split_{split}_test.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ade_split_{split}_test.json",
        ),
        "types": (
            "ade_types.json",
            "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ade_types.json",
        ),
    }

    def __init__(
        self,
        root: str | Path,
        download: DownloadOption = DownloadOption.IF_MISSING_OR_OUT_OF_DATE,
        split: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 0,
        dev_file: Literal["train", "test"] = "train",
    ):
        super().__init__(root, download)

        self.base_dir = os.path.join(self.base_dir, f"split{split}")
        self.dev_file = dev_file
        self.split = split

        self._init_download()

    def download(self) -> DownloadInfo:
        download_info = DownloadInfo()
        for name, (local_file, remote_url) in self.data_files.items():
            local_file = local_file.format(split=self.split)
            file_path = os.path.join(self.base_dir, local_file)

            download_file(file_path, remote_url.format(split=self.split))
            if name == "types":
                download_info.data_files.append(local_file)
            else:
                download_info.splits[name] = [local_file]
                if name == self.dev_file:
                    download_info.splits["dev"] = [local_file]

        return download_info
