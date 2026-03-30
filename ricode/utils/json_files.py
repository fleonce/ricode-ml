import json
import os
from typing import Any, Generator

import pyjson5


def iterate_json_file_type(
    file_path: str | os.PathLike[str],
) -> Generator[Any, None, None]:
    with open(file_path, "r") as f:
        if file_path.endswith(".jsonl"):
            while line := f.readline():
                if line.endswith("\n"):
                    line = line[: -len("\n")]
                yield json.loads(line)
        elif file_path.endswith(".json5"):
            yield from pyjson5.load(f)
        else:
            yield from json.load(f)
