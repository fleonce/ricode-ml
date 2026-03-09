def is_pyarrow_available():
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def is_datasets_available():
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def is_pandas_available():
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False
