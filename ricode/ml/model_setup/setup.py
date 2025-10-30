import contextlib
import pathlib

import torch
from torch.distributed import init_device_mesh

from ricode.ml.distributed import distributed_world_size, is_distributed, is_rank_zero
from ricode.ml.model_setup.activation_checkpointing import _setup_ac
from ricode.ml.model_setup.compile import setup_compile
from ricode.ml.model_setup.ddp import setup_ddp
from ricode.ml.model_setup.fully_sharded_dp import setup_fsdp2
from ricode.ml.model_setup.job_config import JobConfig
from ricode.ml.training_types import ModelInitProtocol, TConfig, TModel


def setup_model(
    model_class: type[TModel],
    job_config: JobConfig,
) -> ModelInitProtocol[TConfig, TModel]:
    def _model_init(
        config: TConfig,
        checkpoint_path: str | pathlib.Path | None = None,
    ) -> TModel:
        init_context = contextlib.nullcontext()
        if (
            job_config.parallelize.dp_mode == "fsdp"
            and is_distributed()
            and not is_rank_zero()
        ):
            init_context = torch.device("meta")

        with init_context:
            if checkpoint_path is None:
                module = model_class(config)
            else:
                module = model_class.from_pretrained(
                    config=config,
                    pretrained_model_name_or_path=checkpoint_path,
                )

        distributed_model_setup(module, job_config=job_config)
        return module

    return _model_init


def distributed_model_setup(
    module: torch.nn.Module,
    *,
    job_config: JobConfig,
) -> None:
    if job_config.activation_checkpoint.mode != "none":
        _setup_ac(
            module,
            job_config.activation_checkpoint,
        )

    if job_config.compile.enabled:
        setup_compile(module, job_config.compile)

    if job_config.parallelize.dp_mode != "none":
        device_mesh = init_device_mesh("cuda", (distributed_world_size(),))

        if job_config.parallelize.dp_mode == "fsdp":
            setup_fsdp2(module, job_config.parallelize, device_mesh)
        elif job_config.parallelize.dp_mode == "ddp":
            setup_ddp(module, job_config.parallelize)
        else:
            raise ValueError(job_config.parallelize.dp_mode)

    return None
