import functools
import hashlib
import inspect
import json
import math
from pathlib import Path

_dtype_to_buffer = {
    int: lambda x: x.to_bytes(math.ceil(x.bit_length() / 8)),
    str: lambda s: s.encode("utf-8"),
    float: lambda f: f.hex().encode("ascii"),
}


def func_cache(*, enabled: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            source_code = inspect.getsource(func)
            cache_hash = hashlib.sha256(source_code.encode("utf-8"))
            for arg in args + tuple(kwargs.values()):
                t = type(arg)
                if t not in _dtype_to_buffer:
                    raise NotImplementedError(arg)
                cache_hash.update(_dtype_to_buffer[t](arg))

            cache_entry = cache_hash.hexdigest()
            cache_dirname = cache_entry[:8]
            cache_entry_name = cache_entry[8:]
            cache_dir = Path.home() / ".cache" / "func_cache" / cache_dirname
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

            cache_path = cache_dir / cache_entry_name
            cache_expired = not cache_path.exists()

            if cache_expired:
                result = func(*args, **kwargs)
                with open(cache_path, "w") as write_f:
                    json.dump(result, write_f)
                return result
            else:  # cache not expired
                if not cache_path.exists():
                    raise ValueError(cache_path)
                with open(cache_path, "r") as read_f:
                    return json.load(read_f)

        return _wrapper

    return decorator
