def is_pyarrow_available():
    try:
        import pyarrow
        return True
    except ImportError:
        return False


def is_datasets_available():
    try:
        import datasets
        return True
    except ImportError:
        return False
