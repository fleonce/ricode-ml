from datetime import datetime

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def parse_datetime(inp: str):
    return datetime.strptime(inp, DATETIME_FORMAT)


def format_datetime(inp: datetime):
    return inp.strftime(DATETIME_FORMAT)
