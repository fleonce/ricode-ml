import atexit
import os
import shutil
from tempfile import mkdtemp, mkstemp
from typing import Literal

import attrs


@attrs.define
class TemporaryDirectory:
    delete_at: Literal["context-exit", "exit"] = "exit"
    name: str = attrs.field(default=None, init=False)

    def __enter__(self) -> str:
        self.name = mkdtemp()
        if self.delete_at == "exit":
            atexit.register(shutil.rmtree, self.name)
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete_at == "context-exit":
            shutil.rmtree(self.name)


@attrs.define
class TemporaryFile:
    delete_at: Literal["content-exit", "exit"] = "exit"
    suffix: str | None = None
    name: str = attrs.field(default=None, init=False)
    fd: int = attrs.field(default=None, init=False)

    def __enter__(self) -> tuple[int, str]:
        self.fd, self.name = mkstemp(self.suffix)
        if self.delete_at == "exit":
            atexit.register(os.remove, self.name)
        return self.fd, self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete_at == "content-exit":
            os.remove(self.name)
