import contextlib
import pathlib
from typing import Sequence

import torch.nn

from ricode.ml.distributed.utils import is_distributed, is_rank_zero
from ricode.ml.training_types import (
    ModelInitProtocol,
    ModelUpdateProtocol,
    TConfig,
    TModel,
)


def guess_model_block_type(module: torch.nn.Module) -> type:
    module_types = list(set(type(mod) for mod in module.modules()))
    module_type_names = list(map(lambda x: x.__name__, module_types))

    for pos, module_name in enumerate(module_type_names):
        if module_name.endswith("Layer") or module_name.endswith("Block"):
            return module_types[pos]

    raise ValueError(f"Cannot find a model block class out of {module_type_names}")


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
