from functools import partial
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from ricode.ml.model_setup.JobConfig import ACConfig
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
    if config.mode == "blockwise":
        _setup_blockwise_ac(module, config)
    elif config.mode == "selective":
        raise NotImplementedError(config)
    else:
        raise ValueError(config.mode)


def _setup_blockwise_ac(
    module: torch.nn.Module,
    config: ACConfig,
    transformer_blocks: list[type[torch.nn.Module]] | None = None,
):
    raise NotImplementedError("todo")


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
