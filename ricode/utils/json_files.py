import json
import pathlib
import shutil
import tempfile
from typing import Any, Generator, Mapping, TypeVar

import pyjson5
from tqdm import tqdm

from ricode.utils.types import ReturnsGeneratorProtocol


TAny = TypeVar("TAny")


def iterate_json_file_type(
    file_path: str | pathlib.Path,
) -> Generator[Any, None, None]:
    file_name = file_path if isinstance(file_path, str) else file_path.name
    with open(file_path, "r") as f:
        if file_name.endswith(".jsonl"):
            while line := f.readline():
                if line.endswith("\n"):
                    line = line[: -len("\n")]
                yield json.loads(line)
        elif file_name.endswith(".json5"):
            yield from pyjson5.load(f)
        else:
            yield from json.load(f)


def iterate_write_with_tempfile(
    fn: ReturnsGeneratorProtocol[TAny],
    output_path: pathlib.Path,
    fn_kwargs: Mapping[str, Any] | None = None,
    desc: str | None = None,
):
    if fn_kwargs is None:
        fn_kwargs = {}

    with tempfile.NamedTemporaryFile("w", delete=False) as out_f, tqdm(desc=desc) as tq:
        for element in fn(**fn_kwargs):
            out_f.write(json.dumps(element) + "\n")
        out_f.close()
        shutil.copyfile(out_f.name, output_path)
