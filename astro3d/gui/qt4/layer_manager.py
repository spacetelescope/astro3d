"""Layer Manager"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from .items import LayerItem, Action
from .. import signaldb

__all__ = ['LayerManager']


class LayerManager(QtGui.QTreeView):
    """Manager the various layers"""

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Layer Manager')
        self.logger = logger

        super(LayerManager, self).__init__(*args, **kwargs)
        self.setHeaderHidden(True)

    def selectionChanged(self, selected, deselected):
        """QT builtin slot called when a selection is changed"""

        def get_selected_item(itemselection):
            try:
                index = itemselection.indexes()[0]
                item = self.model().itemFromIndex(index)
                if not item.is_available:
                    item = None
            except IndexError:
                item = None
            return item

        selected_layer = get_selected_item(selected)
        deselected_layer = get_selected_item(deselected)
        self.logger.debug('selected="{}" deselected="{}"'.format(
            selected_layer,
            deselected_layer
        ))
        signaldb.LayerSelected(
            selected_item=selected_layer,
            deselected_item=deselected_layer,
            source='layermanager')

    def select_from_object(self,
                           selected_item=None,
                           deselected_item=None,
                           source=None):
        # If from the layer manager, there is nothing that need be
        # done.
        if source == 'layermanager':
            return

        try:
            self.setCurrentIndex(selected_item.index())
        except AttributeError:
            """IF cannot select, doesn't matter"""
            pass

    def contextMenuEvent(self, event):
        self.logger.debug('event = "{}"'.format(event))

        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            item = self.model().itemFromIndex(index)
            menu = QtGui.QMenu()
            for action_def in item._actions:
                if isinstance(action_def, Action):
                    action = menu.addAction(action_def.text)
                    action.setData(action_def)
                else:
                    menu.addAction(action_def)

            taken = menu.exec_(event.globalPos())
            if taken:
                self.logger.debug(
                    'taken action = "{}", data="{}"'.format(
                        taken,
                        taken.data()
                    )
                )
                action_def = taken.data()
                action_def.func(*action_def.args)
