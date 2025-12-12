import dataclasses
import pathlib


@dataclasses.dataclass()
class GradientoArgs:
    path: str | pathlib.Path
