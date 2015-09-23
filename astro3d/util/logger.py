import logging


def make_logger(name='__name__', level=logging.WARNING, format=None):
    """Configure a basic logger"""
    if not format:
        format = (
            '%(asctime)s'
            ' | %(levelname)1.1s'
            ' %(filename)s:%(lineno)d (%(funcName)s)'
            ' | %(message)s'
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(format)
    stderrHdlr = logging.StreamHandler()
    stderrHdlr.setFormatter(fmt)
    logger.addHandler(stderrHdlr)

    return logger
