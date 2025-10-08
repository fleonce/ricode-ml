from functools import partial
from typing import Any

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from ricode.ml.optimization.utils import guess_model_block_type
from ricode.ml.training_types import ModelInitProtocol, TConfig, TModel


def setup_activation_checkpointing(
    model_init_fn: ModelInitProtocol[TConfig, TModel],
    policy_fn: Any,
) -> ModelInitProtocol[TConfig, TModel]:

    def _model_init(config: TConfig) -> TModel:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=partial(create_selective_checkpoint_contexts, policy_fn),
        )

        module = model_init_fn(config)
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
