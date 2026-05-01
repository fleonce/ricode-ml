from typing import Any, Iterable, Mapping, Sequence


_PRIMITIVE_TYPES = (int, float, bool, str)


def _line_spacing(line: str, tabsize: int = 2, first_line: bool = True):
    spacing = " " * tabsize
    if "\n" in line:
        lines = line.split("\n")

        return (spacing * first_line) + ("\n" + spacing).join(lines)
    return (spacing * first_line) + line


def pretty_print_array_like(value: Any) -> str:
    array_shape = value.shape
    if len(array_shape) == 2:
        # 2-dim array
        out = "[\n"
        for inner in value:
            out += _line_spacing(pretty_print_array_like(inner), 2) + "\n"
        out += "]"
        return out
    elif len(array_shape) > 2:
        raise ValueError("Unsupported array dimension: " + str(array_shape))
    else:
        return pretty_print_iterable(value, True)


def pretty_print_primitive(value: Any) -> str:
    if not isinstance(value, _PRIMITIVE_TYPES):
        # value is an array-like wrapper (numpy.ndarray, torch.Tensor, ...)
        if hasattr(value, "__array__"):
            try:
                value = value.item()
            except (RuntimeError, ValueError):
                # array has more than one element!
                return pretty_print_array_like(value)
            return pretty_print_primitive(value)
        if str(type(value)) == "<class 'torch.Tensor'>":
            if value.numel() == 1:
                return pretty_print_primitive(value.item())
        raise NotImplementedError(type(value))
    if isinstance(value, (int, bool)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        return f"{value:g}"


def pretty_print_sequence(it: Iterable[Any]) -> str:
    return pretty_print_iterable(it)


def pretty_print_iterable(it: Iterable[Any], inline: bool = False) -> str:
    """
    [a, b, c]
    """
    out = "["
    for i, value in enumerate(it):
        if isinstance(value, (Sequence, set)):
            value = pretty_print_sequence(value)
        elif isinstance(value, Mapping):
            value = pretty_print_dict(value)
        else:
            # print inline
            value = pretty_print_primitive(value)

        if inline:
            out += value + ", "
        else:
            prefix = f" {i}) "
            out += "\n" + prefix + _line_spacing(value, first_line=False)

    if inline:
        if out.endswith(", "):
            out = out[: -len(", ")]
        out += "]"
    else:
        out += "\n]"
    return out


def pretty_print_dict(
    inp: Mapping[str, Any],
) -> str:
    keys = sorted(inp.keys())

    out = "{"

    for key in keys:
        value = inp[key]

        if isinstance(value, Mapping):
            value = pretty_print_dict(value)
        elif isinstance(value, (Sequence, set)):
            value = pretty_print_sequence(value)
        else:
            value = pretty_print_primitive(value)

        value = key + ": " + value
        out += "\n" + _line_spacing(value) + ","
    if out.endswith(","):
        out = out[: -len(",")]
    out = out + "\n}"
    return out


def to_builtin_type(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, Mapping):
        return {
            to_builtin_type(key): to_builtin_type(value) for key, value in value.items()
        }
    elif isinstance(value, Sequence):
        return [to_builtin_type(inner) for inner in value]
    elif isinstance(value, set):
        return {to_builtin_type(inner) for inner in value}
    elif hasattr(value, "__array__"):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            return value.tolist()
    else:
        raise TypeError(value)
