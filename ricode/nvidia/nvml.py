import functools


def setup_nvml(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from pynvml import nvmlInit, nvmlShutdown

        success = False
        try:
            nvmlInit()
            success = True
            return func(*args, **kwargs)
        finally:
            if success:
                nvmlShutdown()

    return wrapper
