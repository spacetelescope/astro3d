"""Logging convenience functions"""
import logging


# Default logging format
DEFAULT_FORMAT = (
    '%(levelname)s:'
    ' %(filename)s:%(lineno)d (%(funcName)s)'
    ' | %(message)s'
)


def make_logger(name, level=logging.INFO, log_format=None):
    """Configure a basic logger

    Parameters
    ----------
    name: str
        The name of the logger. Usually `__name__` from the caller.

    level: int
        The logging level of the logger

    log_format: str or None
        The logging format
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_format = log_format or DEFAULT_FORMAT
    formatter = logging.Formatter(log_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def make_null_logger(name):
    """Create a null logger for the given name

    Parameters
    ----------
    name: str
        The name of the logger. Usually `__name__` from the caller.

    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger
