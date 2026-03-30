import json
import pathlib
from typing import Any, Generator

import pyjson5


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
