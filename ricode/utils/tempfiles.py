import atexit
import shutil
from tempfile import mkdtemp
from typing import Literal

import attrs


@attrs.define
class TemporaryDirectory:
    delete_at: Literal["context-exit", "exit"] = "exit"
    name: str = attrs.field(default=None, init=False)

    def __enter__(self):
        self.name = mkdtemp()
        if self.delete_at == "exit":
            atexit.register(shutil.rmtree, self.name)
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.delete_at == "context-exit":
            shutil.rmtree(self.name)
