"""Preferences"""
from functools import partial

from ...external.qt import (QtGui, QtCore)


class Preferences(QtGui.QMenu):

    def __init__(self, *args, **kwargs):
        super(Preferences, self).__init__(*args, **kwargs)
        self.mutex = QtCore.QMutex()

    def exec_(self, pos, checked):
        """Called by the Prefences QAction to bring up the Preferences panel"""
        if not self.mutex.tryLock():
            return
        try:
            pos = pos()
        except TypeError:
            """pos is already a value, ignore"""
        if pos is None:
            pos = QtGui.QCursor.pos()
        try:
            result = super(Preferences, self).exec_(pos)
        finally:
            self.mutex.unlock()
        return result

    def for_menubar(self, pos=None, parent=None):
        """Create the dummy menu to encorporate into system menubar"""
        act = QtGui.QAction('Preferences', parent)
        act.triggered.connect(partial(self.exec_, pos))
        #act.setMenuRole(QtGui.QAction.PreferencesRole)

        menu = QtGui.QMenu('Preferences')
        menu.addAction(act)
        return menu
