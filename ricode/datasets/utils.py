import enum
import hashlib
import importlib
import os.path
from collections import OrderedDict
from pathlib import Path
from typing import MutableMapping, MutableSequence

import attrs
from tqdm import tqdm


class DownloadOption(enum.Enum):
    ALWAYS = 0
    NEVER = 1
    IF_MISSING_OR_OUT_OF_DATE = 2


@attrs.define
class DownloadInfo:
    data_files: MutableSequence[str] = attrs.field(factory=list)
    splits: MutableMapping[str, MutableSequence[str]] = attrs.field(factory=OrderedDict)


# https://github.com/pytorch/vision/blob/61045743aeec0ef4da65d1ac008ac0f2f5e38cb3/torchvision/datasets/utils.py#L35
def calculate_sha1(file: str | Path, chunk_size: int = 1024**2):
    sha1 = hashlib.sha1(usedforsecurity=False)
    with open(file, "rb") as f:
        while chunk := f.read(chunk_size):
            sha1.update(chunk)
    return sha1.hexdigest()


def check_integrity(file: str | Path, expected_sha1: str):
    if not os.path.exists(file):
        return False

    return calculate_sha1(file) == expected_sha1


def download_file(file: str, remote_url: str):
    import urllib.request

    with tqdm(desc=file) as progress_bar:

        def hook(block: int, read: int, total: int):
            if progress_bar.total is None:
                progress_bar.total = total
            done = min((block + 1) * read, total)
            progress_bar.n = done
            progress_bar.refresh()

        urllib.request.urlretrieve(remote_url, file, hook)


def is_package_available(package_name: str):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False
