import json
import pathlib
import shutil
import tempfile
from typing import Any, Callable, Generator, Mapping, Sequence, TypeVar

import pyjson5
from tqdm import tqdm


TAny = TypeVar("TAny")


def load_json_file_type(file_path: str | pathlib.Path):
    file_name = file_path if isinstance(file_path, str) else file_path.name
    with open(file_path, "r") as f:
        if file_name.endswith(".jsonl"):
            return list(iterate_json_file_type(file_path))
        elif file_name.endswith(".json5"):
            return pyjson5.load(f)
        else:
            return json.load(f)


def save_json_file_type(
    o: Sequence[Any] | Mapping[str, Any] | bool | int | float | str | None,
    file_path: str | pathlib.Path,
):
    file_name = file_path if isinstance(file_path, str) else file_path.name
    with open(file_path, "w") as f:
        if file_name.endswith(".jsonl"):
            if not isinstance(o, Sequence):
                o = [o]
            for element in o:
                f.write(json.dumps(element) + "\n")
        elif file_name.endswith(".json5"):
            pyjson5.dump(o, f)
        else:
            json.dump(o, f)


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
    fn: Callable[[...], Generator[Any, None, None]],
    output_path: str | pathlib.Path,
    fn_kwargs: Mapping[str, Any] | None = None,
    desc: str | None = None,
    total: int | None = None,
):
    if fn_kwargs is None:
        fn_kwargs = {}

    with (
        tempfile.NamedTemporaryFile("w", delete=False) as out_f,
        tqdm(desc=desc, total=total) as tq,
    ):
        for element in fn(**fn_kwargs):
            out_f.write(json.dumps(element) + "\n")
            tq.update()
        out_f.close()
        shutil.copyfile(out_f.name, output_path)
