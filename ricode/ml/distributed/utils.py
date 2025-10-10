import functools
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def rank_zero_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            res = func(*args, **kwargs)
            distributed_barrier()
        else:
            distributed_barrier()
            res = func(*args, **kwargs)
        return res

    return wrapper


def is_distributed() -> bool:
    if dist.is_initialized():
        return True

    if "RANK" in os.environ:
        distributed_setup()
        return dist.is_initialized()
    return False


@functools.lru_cache(1)
def distributed_rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


@functools.lru_cache(1)
def distributed_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


@functools.lru_cache(1)
def is_rank_zero() -> bool:
    return not is_distributed() or distributed_rank() == 0


def distributed_barrier():
    if dist.is_initialized():
        dist.barrier()


def _distributed_get_cuda_device():
    if "CUDA_LOCAL_DEVICE" in os.environ:
        try:
            return int(os.environ["CUDA_LOCAL_DEVICE"])
        except ValueError:
            return False
    return False


def distributed_setup():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if dist.is_initialized():
        return rank, world_size, torch.cuda.current_device()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    local_device = rank
    if (cuda_local_device := _distributed_get_cuda_device()) is not False:
        local_device = cuda_local_device

    torch.cuda.set_device(local_device)

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", local_device),
    )
    return rank, world_size, local_device


def distributed_cleanup():
    dist.destroy_process_group()


def finalise_distributed_environment_after_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if dist.is_initialized():
                distributed_cleanup()

    return wrapper
