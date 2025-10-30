from functools import partial
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from ricode.ml.model_setup.job_config import ACConfig
from ricode.ml.model_setup.utils import (
    guess_model_block_type,
    guess_model_block_types,
    identity,
)
from ricode.ml.training_types import ModelUpdateProtocol, TModel


def setup_blockwise_activation_checkpointing(
    wrapping_blocks: list[type[torch.nn.Module]] | None = None,
    *,
    disable: bool = False,
) -> ModelUpdateProtocol[TModel]:
    if disable:
        return identity

    non_reentrant_wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )

    def _model_init(module: TModel) -> TModel:
        nonlocal wrapping_blocks

        if wrapping_blocks is None:
            wrapping_blocks = guess_model_block_types(module)

        def selective_activation_checkpoint(submodule):
            if any(isinstance(submodule, block) for block in wrapping_blocks):
                return True
            return False

        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_activation_checkpoint,
        )

        return module

    return _model_init


def _setup_ac(
    module: torch.nn.Module,
    config: ACConfig,
):
    blocks = guess_model_block_types(module)
    if len(blocks) > 1 and not config.allow_multiple_blocks:
        raise ValueError(
            f"Found multiple model block types, unable to guess the right one: {blocks!r}"
        )

    for name, submodule in module.named_modules():
        if type(submodule) in blocks:
            # we found a module that is a transformer block
            if config.mode == "blockwise":
                checkpointed_module = _setup_blockwise_ac(submodule, config)
            elif config.mode == "selective":
                checkpointed_module = _setup_selective_ac(submodule, config)
            else:
                raise ValueError(config.mode)
            if "." in name:
                submodule_parent, submodule_name = name.rsplit(".", 1)
                module.get_submodule(submodule_parent).register_module(
                    submodule_name, checkpointed_module
                )
            else:
                module.register_module(name, checkpointed_module)
    return None


def _resolve_ops(ops: list[str]):
    def _resolve_op(op: str, parent=None):
        if op.startswith("torch."):
            return _resolve_op(op[len("torch.") :], torch)
        if "." in op:
            base, rest = op.split(".", maxsplit=1)
            return _resolve_op(rest, getattr(parent, base))
        else:
            return getattr(parent, op)

    return [_resolve_op(o) for o in ops]


def _setup_selective_ac(
    module: torch.nn.Module,
    config: ACConfig,
):
    ops = _resolve_ops(config.selective_ac_save_list)

    def policy_fn(ctx, op, *args, **kwargs):
        if op in ops:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    return checkpoint_wrapper(
        module,
        context_fn=partial(create_selective_checkpoint_contexts, policy_fn),
    )


def _setup_blockwise_ac(
    module: torch.nn.Module,
    config: ACConfig,
):
    return checkpoint_wrapper(
        module,
    )


def setup_selective_activation_checkpointing(
    policy_fn: Any,
    *,
    disable: bool = False,
) -> ModelUpdateProtocol[TModel]:
    if disable:
        return identity

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=partial(create_selective_checkpoint_contexts, policy_fn),
    )

    def _model_init(module: TModel) -> TModel:
        wrapping_block = guess_model_block_type(module)

        def selective_activation_checkpoint(submodule):
            return isinstance(submodule, wrapping_block)

        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_activation_checkpoint,
        )

        return module

    return _model_init
