# taking inspiration from torchtitan JobConfig here ...
from typing import Literal

import attrs


#    aten.mm,
#    aten.convolution,
#    aten.convolution_backward,
#    aten.bmm,
#    aten.addmm,
#    aten._scaled_dot_product_flash_attention,
#    aten._scaled_dot_product_efficient_attention,
#    aten._flash_attention_forward,
#    aten._efficient_attention_forward,
#    aten.upsample_bilinear2d,
#    aten._scaled_mm
# https://pytorch.org/blog/activation-checkpointing-techniques/
_op_sac_save_list = [
    "torch.ops.aten.mm",
    "torch.ops.aten.bmm",
    "torch.ops.aten.addmm",
    "torch.ops.aten._scaled_mm",
    "torch.ops.aten._flash_attention_forward",
    "torch.ops.aten._efficient_attention_forward",
    "torch.ops.aten._scaled_dot_product_flash_attention",
    "torch.ops.aten._scaled_dot_product_efficient_attention",
]


@attrs.define
class ACConfig:
    mode: Literal["none", "blockwise", "selective"] = "none"
    allow_multiple_blocks: bool = False
    selective_ac_save_list: list[str] = attrs.field(factory=lambda: _op_sac_save_list)


@attrs.define
class CompileConfig:
    enabled: bool = False
    dynamic: bool = False


@attrs.define
class ParallelizeConfig:
    dp_mode: Literal["none", "ddp", "fsdp"] = "none"
    allow_multiple_blocks: bool = False
    param_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    reduce_dtype: Literal["float32", "bfloat16"] = "bfloat16"


@attrs.define
class JobConfig:
    activation_checkpoint: ACConfig = attrs.field(factory=lambda: ACConfig())
    compile: CompileConfig = attrs.field(factory=lambda: CompileConfig())
    parallelize: ParallelizeConfig = attrs.field(factory=lambda: ParallelizeConfig())
