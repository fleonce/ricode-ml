from typing import Callable


def gt(a: float, b: float, stop_on_optimum: bool = True):
    if a > b:
        if a >= 1.0 and stop_on_optimum:
            raise StopIteration
        return True
    return False


def lt(a: float, b: float, stop_on_optimum: bool = True):
    if a < b:
        if a <= 0 and stop_on_optimum:
            raise StopIteration
        return True
    return False


def safe_score_comparison(operator: Callable[[float, float], bool], a: float, b: float):
    try:
        return operator(a, b)
    except StopIteration:
        # the above methods raise stop iteration if the optimum was reached with 'a', but not with 'b'
        return True
