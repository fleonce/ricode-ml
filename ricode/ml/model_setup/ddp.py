import torch.nn

from torch.distributed._composable import replicate

from ricode.ml.model_setup.job_config import ParallelizeConfig


def setup_ddp(module: torch.nn.Module, config: ParallelizeConfig) -> None:
    replicate(
        module,
        bucket_cap_mb=100,
    )
    pass
