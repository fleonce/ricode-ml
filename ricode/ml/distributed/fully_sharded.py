import torch.nn
from torch.distributed.fsdp import FSDPModule


def _is_fully_sharded_model(module: torch.nn.Module):
    return isinstance(module, FSDPModule)
