import functools


def setup_nvml(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from pynvml import nvmlInit, nvmlShutdown

        try:
            nvmlInit()
            return func(*args, **kwargs)
        finally:
            nvmlShutdown()

    return wrapper
