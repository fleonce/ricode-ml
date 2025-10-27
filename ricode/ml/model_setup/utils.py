import contextlib
import pathlib
from typing import Sequence

import more_itertools
import torch.nn

from ricode.ml.distributed.utils import is_distributed, is_rank_zero
from ricode.ml.model_setup.activation_checkpointing import _setup_ac
from ricode.ml.model_setup.JobConfig import JobConfig
from ricode.ml.training_types import (
    ModelInitProtocol,
    ModelUpdateProtocol,
    TConfig,
    TModel,
)


def guess_model_block_type(module: torch.nn.Module) -> type:
    block_types = guess_model_block_types(module)
    if len(block_types) != 1:
        raise ValueError(
            f"Found multiple model block types, unable to guess the right one: {block_types!r}"
        )
    return more_itertools.first(block_types)


def guess_model_block_types(module: torch.nn.Module) -> set[type]:
    module_types = list(set(type(mod) for mod in module.modules()))
    module_type_names = list(map(lambda x: x.__name__, module_types))

    found_types = set()
    for pos, module_name in enumerate(module_type_names):
        if module_name.endswith("Layer") or module_name.endswith("Block"):
            found_types.add(module_types[pos])

    if len(found_types) == 0:
        raise ValueError(f"Cannot find a model block class out of {module_type_names}")
    return found_types


def setup_model(
    model_class: type[TModel],
    *,
    for_fully_sharded_dp: bool = False,
) -> ModelInitProtocol[TConfig, TModel]:
    def _model_init(
        config: TConfig, checkpoint_path: str | pathlib.Path | None = None
    ) -> TModel:
        init_context = contextlib.nullcontext()
        if for_fully_sharded_dp and is_distributed() and not is_rank_zero():
            init_context = torch.device("meta")

        with init_context:
            if checkpoint_path is None:
                return model_class(config)
            else:
                return model_class.from_pretrained(
                    config=config, pretrained_model_name_or_path=checkpoint_path
                )

    return _model_init


def setup_model_complete(
    model_class: type[TModel],
    *,
    job_config: JobConfig,
    for_fully_sharded_dp: bool = False,
) -> ModelInitProtocol[TConfig, TModel]:
    def _model_init(
        config: TConfig, checkpoint_path: str | pathlib.Path | None = None
    ) -> TModel:
        init_context = contextlib.nullcontext()
        if for_fully_sharded_dp and is_distributed() and not is_rank_zero():
            init_context = torch.device("meta")

        with init_context:
            if checkpoint_path is None:
                # initialize the model by a config object
                model = model_class(config)
            else:
                # initialize the model from a checkpoint at `checkpoint_path`
                model = model_class.from_pretrained(
                    config=config, pretrained_model_name_or_path=checkpoint_path
                )

        if job_config.activation_checkpoint.mode != "none":
            _setup_ac(
                model,
                job_config.activation_checkpoint,
            )

        return model

    return _model_init


def identity(model: TModel) -> TModel:
    return model


def chain_model_init(
    model_init: ModelInitProtocol[TConfig, TModel],
    updates: Sequence[ModelUpdateProtocol[TModel]],
) -> ModelInitProtocol[TConfig, TModel]:
    def _model_init(
        config: TConfig, checkpoint_path: str | pathlib.Path | None = None
    ) -> TModel:
        module = model_init(config, checkpoint_path)

        for update in updates:
            module = update(module)
        return module

    return _model_init
