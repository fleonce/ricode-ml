from functools import partial

import packaging.version
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from more_itertools.more import first
from torch.distributed.fsdp import (
    fully_shard,
    FullyShardedDataParallel,
    MixedPrecision,
    MixedPrecisionPolicy,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from ricode.ml.model_setup.job_config import ParallelizeConfig
from ricode.ml.model_setup.utils import (
    guess_model_block_type,
    guess_model_block_types,
    identity,
)
from ricode.ml.training_types import ModelUpdateProtocol, TModel


def _is_bfloat16_supported():
    is_bfloat16_supported = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
    return is_bfloat16_supported


def _is_fully_sharded_dp_model(model: torch.nn.Module):
    return isinstance(model, FullyShardedDataParallel)


def setup_mixed_precision_policy(
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype | None = None,
    buffer_dtype: torch.dtype | None = None,
) -> MixedPrecision:
    if _is_bfloat16_supported():
        if param_dtype is None:
            param_dtype = torch.bfloat16
        if reduce_dtype is None:
            reduce_dtype = torch.bfloat16
        if buffer_dtype is None:
            buffer_dtype = torch.bfloat16

    if not _is_bfloat16_supported and torch.bfloat16 in {
        param_dtype,
        reduce_dtype,
        buffer_dtype,
    }:
        raise ValueError(
            "torch.bfloat16 is unsupported for mixed precision training on this hardware"
        )

    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )


def setup_fsdp2(
    module: torch.nn.Module,
    config: ParallelizeConfig,
):
    blocks = guess_model_block_types(module)
    if len(blocks) > 1 and not config.allow_multiple_blocks:
        raise ValueError(
            f"Found multiple model block types, unable to guess the right one: {blocks!r}"
        )

    mp_policy = MixedPrecisionPolicy(
        getattr(torch, config.param_dtype),
        getattr(torch, config.reduce_dtype),
        cast_forward_inputs=True,
    )

    kwargs = {"mp_policy": mp_policy, "reshard_after_forward": True}
    for name, submodule in module.named_modules():
        if type(submodule) in blocks:
            fully_shard(submodule, **kwargs)

    for child in module.children():
        fully_shard(child, **kwargs)


def setup_fully_sharded_dp_model(
    wrapping_block: type[torch.nn.Module] | set[type[torch.nn.Module]] | None = None,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    mixed_precision_policy: MixedPrecision | None = None,
    use_orig_params: bool = False,
    limit_all_gathers: bool = True,
    sync_module_states: bool = True,
    allow_multiple_wrapping_blocks: bool = False,
    *,
    disable: bool = False,
) -> ModelUpdateProtocol[TModel]:
    if disable:
        return identity

    if mixed_precision_policy is None:
        mixed_precision_policy = setup_mixed_precision_policy()

    def param_fn(module: torch.nn.Module):
        module.to_empty(device=torch.device("cuda"), recurse=False)

    def _model_init(module: TModel) -> TModel:
        if rank > 0:
            meta_device = torch.device("meta")
            param_device = first(module.parameters()).device
            if param_device != meta_device:
                raise ValueError(
                    f"For FSDP, the model must be initialised on the meta device for all ranks>0, "
                    f"device is {param_device!r}"
                )

        nonlocal wrapping_block
        if wrapping_block is None:
            if not allow_multiple_wrapping_blocks:
                wrapping_block = guess_model_block_type(module)
            else:
                wrapping_block = guess_model_block_types(module)
        else:
            if (
                not allow_multiple_wrapping_blocks
                and isinstance(wrapping_block, set)
                and len(wrapping_block) > 1
            ):
                raise ValueError(
                    f"{allow_multiple_wrapping_blocks=}, but len({wrapping_block!r}) > 1"
                )

        if not isinstance(wrapping_block, set):
            wrapping_block = {wrapping_block}

        wrapping_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls=wrapping_block
        )

        module = FullyShardedDataParallel(
            module,
            auto_wrap_policy=wrapping_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_policy,
            use_orig_params=use_orig_params,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=limit_all_gathers,
            sync_module_states=sync_module_states,
            param_init_fn=param_fn,
        )
        return module  # type: ignore

    return _model_init


def setup_fully_sharded_dp2_model() -> ModelUpdateProtocol[TModel]:
    raise NotImplementedError("todo")
