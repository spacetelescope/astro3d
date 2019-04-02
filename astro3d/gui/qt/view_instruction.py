"""View the basic instructions"""
from qtpy import (QtCore, QtWidgets)
from qtpy.QtCore import Signal as pyqtSignal

from ...util.text_catalog import TEXT_CATALOG

__all__ = ['InstructionViewer']


class InstructionViewer(QtWidgets.QTextBrowser):
    """View the basic instructions

    Parameters
    ----------
    qt_args, qt_kwargs: argument expansions
        General Qt arguments
    """

    closed = pyqtSignal(bool)

    def __init__(self, *qt_args, **qt_kwargs):
        super(InstructionViewer, self).__init__(*qt_args, **qt_kwargs)
        self.setHtml(TEXT_CATALOG['instructions_default'])

        self.setOpenExternalLinks(True)

    def toggle_view(self):
        """Toggle visibility"""
        sender = self.sender()
        self.setVisible(sender.isChecked())

    def closeEvent(self, event):
        """Close the window"""
        self.closed.emit(False)
        event.accept()
