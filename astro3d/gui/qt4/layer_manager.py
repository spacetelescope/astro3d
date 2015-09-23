"""Layer Manager"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger

__all__ = ['LayerManager']


class LayerManager(QtGui.QTreeView):
    """Manager the various layers"""

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Layer Manager')
        self.logger = logger

        super(LayerManager, self).__init__(*args, **kwargs)
