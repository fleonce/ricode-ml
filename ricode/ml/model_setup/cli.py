import dataclasses
from typing import Literal, Optional


@dataclasses.dataclass
class DistributedArguments:
    ac_mode: Literal["none", "blockwise", "selective"] = dataclasses.field(
        default="none",
        metadata={
            "help": "The activation checkpointing (AC) strategy to apply. "
            "Blockwise means applying AC at every transformer block. "
            "Selective applies AC to specific pytorch operations such as a matmul."
        },
    )
    # ac_allow_multiple_blocks: bool = False
    selective_ac_save_list: Optional[list[str]] = None

    compile_enabled: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether to compile each transformer block of a model."},
    )
    compile_dynamic: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "If compile is enabled, use dynamic shapes to compile the transformer block."
        },
    )

    parallelize_mode: Literal["none", "ddp", "fsdp"] = dataclasses.field(
        default="none",
        metadata={
            "help": "The parallelization strategy to apply. DDP replicates the whole model across all available devices. "
            "FSDP shards the model's parameters across multiple devices, effectively reducing the peak memory requirement."
        },
    )
    # parallelize_allow_multiple_blocks: bool = False
    parallelize_param_dtype: Literal["float32", "bfloat16"] = dataclasses.field(
        default="bfloat16",
        metadata={
            "help": "The precision in which to keep the model parameters in during FSDP training."
        },
    )
    parallelize_reduce_dtype: Literal["float32", "bfloat16"] = dataclasses.field(
        default="bfloat16",
        metadata={
            "help": "The precision with which to reduce the parameters during FSDP training."
        },
    )
