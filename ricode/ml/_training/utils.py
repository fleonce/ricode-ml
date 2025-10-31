import math
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


def _format_to_memory_units(inp: int):
    return _format_to_powers_of_1000(inp, [None, "KiB", "MiB", "GiB", "TiB"], 1000)


def format_to_energy_usage(inp: float):
    return _format_to_powers_of_1000(inp, ["mWh", "Wh", "kWh", "MWh"])


def _format_to_powers_of_1000(inp: float | int, units=None, base=1000):
    if units is None:
        units = [
            None,
            "K",  # 10**3
            "M",  # 10 ** 6
            "B",  # 10 ** 9
            "T",  # 10 ** 12
            "q",  # 10 ** 15
            "Q",  # 10 ** 18
        ]

    if inp <= base:
        if units[0] is None:
            return str(inp).rjust(7)
        return f"{inp:.0f} {units[0]}".rjust(7)

    scale = math.floor(math.log(inp, base))
    unit = units[scale]
    number = inp / (base**scale)
    return f"{number:.1f} {unit}".rjust(7)


def _format_to_percentage(inp: int):
    return f"{inp}%".rjust(4)
