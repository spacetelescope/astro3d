"""Main UI Viewer
"""
from ..external.qt import (QtGui, QtCore)
from ..external.qt.QtGui import QMainWindow as GTK_MainWindow
from qt4 import ViewImage

__all__ = ['MainWindow']


class MainWindow(GTK_MainWindow):
    """Main Viewer
    """
    def __init__(self, logger, parent=None):
        super(MainWindow, self).__init__(parent)
        self.logger = logger
        self._build_gui()
        self.show()

    def _build_gui(self):
        """Construct the app's GUI"""

        # Content
        image_viewer = ViewImage(self.logger)
        image_viewer_widget = image_viewer.get_widget()
        image_viewer_widget.resize(512, 512)
        self.image_viewer = image_viewer

        # Window
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        vbox.setSpacing(1)
        vbox.addWidget(image_viewer_widget, stretch=1)

        # Main buttons
        wopen = QtGui.QPushButton("Open File")
        self.wopen = wopen
        wquit = QtGui.QPushButton("Quit")
        self.wquit = wquit

        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(QtCore.QMargins(4, 2, 4, 2))
        hbox.addStretch(1)
        for w in (wopen, wquit):
            hbox.addWidget(w, stretch=0)

        hw = QtGui.QWidget()
        hw.setLayout(hbox)
        vbox.addWidget(hw, stretch=0)

        vw = QtGui.QWidget()
        self.setCentralWidget(vw)
        vw.setLayout(vbox)

    def set_callback(self, name, func):
        """Set callbacks"""

        if name == 'drag-drop':
            self.image_viewer.set_callback('drag-drop', func)
        elif name == 'open-file':
            self.wopen.clicked.connect(func)
        elif name == 'quit':
            self.wquit.clicked.connect(func)
        else:
            self.logger.warn('No such callback: {}'.format(name))

    def get_filename(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                ".", "FITS files (*.fits)")
        if isinstance(res, tuple):
            filename = res[0]
        else:
            filename = str(res)
        return filename
