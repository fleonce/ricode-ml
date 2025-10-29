from typing import Callable

import torch.nn
from torch.distributed.tensor import DTensor


def num_parameters(module: torch.nn.Module):
    def _reduction(param: torch.nn.Parameter):
        return param.numel()

    return _reduce_parameters(module, _reduction)


def estimated_model_size(module: torch.nn.Module):
    """
    Returns the number of bytes the model occupies in global memory (VRAM)
    """

    def _reduction(param: torch.nn.Parameter):
        return param.numel() * param.dtype.itemsize

    return _reduce_parameters(module, _reduction)


def num_gradient_parameters(module: torch.nn.Module):
    def _reduction(param: torch.nn.Parameter):
        return param.numel() if param.requires_grad else 0

    return _reduce_parameters(module, _reduction)


def num_local_parameters(module: torch.nn.Module):
    def _reduction(param: torch.nn.Parameter):
        if isinstance(param.data, DTensor):
            return param.data.to_local().numel()
        return param.numel()

    return _reduce_parameters(module, _reduction)


def _reduce_parameters(
    module: torch.nn.Module, reduction: Callable[[torch.nn.Parameter], int]
):
    return sum(map(reduction, module.parameters()))
