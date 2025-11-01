import functools

from pynvml import NVMLError, nvmlInit, nvmlShutdown


def _nvml_available():
    try:
        nvmlInit()
        nvmlShutdown()
        return True
    except NVMLError:
        return False


def setup_nvml(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        success = False
        try:
            nvmlInit()
            success = True
            return func(*args, **kwargs)
        finally:
            if success:
                nvmlShutdown()

    if not _nvml_available():
        return func

    return wrapper
