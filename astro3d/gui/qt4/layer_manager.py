"""Layer Manager"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from ..items import LayerItem


__all__ = ['LayerManager']


class LayerManager(QtGui.QTreeView):
    """Manager the various layers"""

    layer_selected = QtCore.pyqtSignal(LayerItem, name='layerSelected')

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Layer Manager')
        self.logger = logger

        super(LayerManager, self).__init__(*args, **kwargs)
        self.setHeaderHidden(True)

    def selectionChanged(self, selected, deselected):
        """QT builtin slot called when a selection is changed"""
        layer = selected.indexes()[0]
        layer = self.model().itemFromIndex(layer)
        self.logger.debug('layer="{}"'.format(layer))
        self.layer_selected.emit(layer)

    def contextMenuEvent(self, event):
        self.logger.debug('event = "{}"'.format(event))

        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            item = self.model().itemFromIndex(index)
            menu = QtGui.QMenu()
            for action_def in item._actions:
                action = menu.addAction(action_def.text)
                action.setData(action_def)

            action = menu.exec_(event.globalPos())
            if action:
                self.logger.debug('chosen action = "{}", data="{}"'.format(action, action.data()))
                action_def = action.data()
                action_def.func(*action_def.args)
