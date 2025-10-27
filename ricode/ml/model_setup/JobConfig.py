# taking inspiration from torchtitan JobConfig here ...
from typing import Literal

import attrs


@attrs.define
class ACConfig:
    mode: Literal["none", "blockwise", "selective"] = "none"
    allow_multiple_blocks: bool = False


@attrs.define
class CompileConfig:
    enabled: bool = False


@attrs.define
class JobConfig:
    activation_checkpoint: ACConfig = attrs.field(factory=lambda: ACConfig())
    compile: CompileConfig = attrs.field(factory=lambda: CompileConfig())
