import functools
import os
from datetime import timedelta
from functools import partial
from typing import Any, cast, Mapping, TypeVar

import packaging.version
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module
from transformers import PreTrainedModel

from ricode.ml.training_types import ModelInitProtocol

TConfig = TypeVar("TConfig")
TModel = TypeVar("TModel", bound=PreTrainedModel)


def rank_zero_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            res = func(*args, **kwargs)
            distributed_barrier()
        else:
            distributed_barrier()
            res = func(*args, **kwargs)
        return res

    return wrapper


def in_distributed_group() -> bool:
    if dist.is_initialized():
        return True

    if "RANK" in os.environ:
        fsdp_setup()
        return dist.is_initialized()
    return False


def distributed_rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def distributed_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def is_rank_zero() -> bool:
    return not in_distributed_group() or distributed_rank() == 0


def distributed_barrier():
    if dist.is_initialized():
        dist.barrier()


def fsdp_setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if dist.is_initialized():
        return rank, world_size, torch.cuda.current_device()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    local_device = rank
    if (cuda_local_device := fsdp_get_cuda_device()) is not False:
        local_device = cuda_local_device

    torch.cuda.set_device(local_device)

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", local_device),
    )
    return rank, world_size, local_device


def do_fsdp_cleanup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if dist.is_initialized():
                fsdp_cleanup()

    return wrapper


def fsdp_cleanup():
    dist.destroy_process_group()


def fsdp_guess_wrapping_block_from_model_types(model_types: list[type]) -> type:
    model_type_names = list(map(lambda x: x.__name__, model_types))
    candidates = ["DebertaV2Layer", "T5Block", "BertLayer", "ModernBertEncoderLayer"]
    for candidate in candidates:
        if candidate in model_type_names:
            return model_types[model_type_names.index(candidate)]
    raise ValueError(f"Cannot find a candidate for FSDP in {candidates}")


def fsdp_mixed_precision_policy(use_bfloat16: bool):
    is_bfloat16_supported = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if is_bfloat16_supported and use_bfloat16:
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return None


def fsdp_get_cuda_device():
    if "CUDA_LOCAL_DEVICE" in os.environ:
        try:
            return int(os.environ["CUDA_LOCAL_DEVICE"])
        except ValueError:
            return False
    return False


def fsdp_setup_model(
    config: TConfig,
    model_init_fn: ModelInitProtocol[TConfig, TModel],
    model_init_kwargs: Mapping[str, Any],
    use_bfloat16: bool = True,
    use_hsdp: bool = False,
    use_activation_checkpointing: bool = True,
    **kwargs: Any,
) -> tuple[TModel, int, int, str]:
    kwargs = kwargs or dict()

    rank, world_size, device = fsdp_setup()

    # initialize the model on CPU
    model_init_kwargs = dict(model_init_kwargs) or dict()
    if model_init_kwargs:
        if "config" not in model_init_kwargs:
            model_init_kwargs["config"] = config
        model_init_fn = partial(model_init_fn, **model_init_kwargs)
    else:
        model_init_fn = partial(model_init_fn, config=config)
    model = None
    if rank == 0:
        model = model_init_fn()
    else:
        with torch.device("meta"):
            model = model_init_fn()
    if model is None:
        raise ValueError("Cannot initialize FSDP without a model")

    sharding_strategy = kwargs.get("sharding_strategy", ShardingStrategy.FULL_SHARD)
    if use_hsdp:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD

    module_types = list(set(type(module) for module in model.modules()))

    wrapping_block = None
    if "wrapping_block" in kwargs:
        _wrapping_block = kwargs.get("wrapping_block")
        if _wrapping_block is not None and not issubclass(_wrapping_block, Module):
            raise ValueError(
                f"FSDP wrapping_block must be a type[Module], got {type(_wrapping_block)}"
            )
        wrapping_block = _wrapping_block

    wrapping_block = (
        wrapping_block
        if wrapping_block is not None
        else fsdp_guess_wrapping_block_from_model_types(module_types)
    )

    wrapping_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={wrapping_block}
    )

    def param_fn(module: Module):
        module.to_empty(device=torch.device("cuda"), recurse=False)

    # convert model to FSDP model
    model = cast(
        TModel,
        FullyShardedDataParallel(
            model,
            auto_wrap_policy=wrapping_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=fsdp_mixed_precision_policy(use_bfloat16),
            use_orig_params=kwargs.get("use_orig_params", False),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=kwargs.get("limit_all_gathers", True),
            sync_module_states=kwargs.get("sync_module_states", True),
            param_init_fn=param_fn,
        ),
    )

    if use_activation_checkpointing:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )

        def selective_activation_checkpoint(submodule):
            return isinstance(submodule, wrapping_block)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_activation_checkpoint,
        )

    return model, rank, world_size, f"cuda:{device}"
