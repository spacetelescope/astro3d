"""Start the UI application
"""
from .qt.pyqt_nonblock import pyqtapplication


def start_ui_app(argv=None):
    """Start the appropriate UI application

    Parameters
    ----------
    argv: str
        The argument string

    Returns
    -------
    UI application
    """

    return pyqtapplication(argv)
