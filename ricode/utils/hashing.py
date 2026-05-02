import hashlib
from typing import Any, Mapping

from ricode.utils.pretty_printing import pretty_print_primitive


def mapping_to_hash(m: Mapping[str, Any], hash_fn=hashlib.sha1) -> str:
    res = hash_fn()
    for key in sorted(m.keys()):
        value = m[key]

        res.update(key.encode())
        res.update(pretty_print_primitive(value).encode())
    return res.hexdigest()
