"""Main UI Viewer
"""

from ..external.qt import (QtGui, QtCore)
from ..external.qt.QtGui import QMainWindow as GTK_MainWindow
from qt4 import (ViewImage, ViewMesh)


__all__ = ['MainWindow']


class MainWindow(GTK_MainWindow):
    """Main Viewer
    """
    def __init__(self, signals, logger, parent=None):
        super(MainWindow, self).__init__(parent)
        self.logger = logger
        self.signals = signals
        self._build_gui()

        # Setup signals
        self.image_viewer.set_callback('drag-drop', self.drop_file)
        self.wopen.clicked.connect(self.open_file)
        self.wquit.clicked.connect(self.signals.quit)
        self.signals.quit.connect(self.quit)
        self.signals.new_image.connect(self.image_update)
        self.signals.update_mesh.connect(self.mesh_viewer.update_mesh)

    def _build_gui(self):
        """Construct the app's GUI"""

        # Content
        image_viewer = ViewImage(self.logger)
        image_viewer_widget = image_viewer.get_widget()
        image_viewer_widget.resize(512, 512)
        self.image_viewer = image_viewer

        mesh_viewer = ViewMesh()
        mesh_viewer.resize(512, 512)
        self.mesh_viewer = mesh_viewer

        # Viewers
        viewer_hbox = QtGui.QHBoxLayout()
        for w in (image_viewer_widget, mesh_viewer):
            viewer_hbox.addWidget(w, stretch=1)

        # Main buttons
        wopen = QtGui.QPushButton("Open File")
        self.wopen = wopen
        wquit = QtGui.QPushButton("Quit")
        self.wquit = wquit

        button_hbox = QtGui.QHBoxLayout()
        button_hbox.setContentsMargins(QtCore.QMargins(4, 2, 4, 2))
        button_hbox.addStretch(1)
        for w in (wopen, wquit):
            button_hbox.addWidget(w, stretch=0)

        # Window
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        vbox.setSpacing(1)
        for w in (viewer_hbox, button_hbox):
            hw = QtGui.QWidget()
            hw.setLayout(w)
            vbox.addWidget(hw)

        vw = QtGui.QWidget()
        self.setCentralWidget(vw)
        vw.setLayout(vbox)

    def open_file(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                ".", "FITS files (*.fits)")
        if isinstance(res, tuple):
            filename = res[0]
        else:
            filename = str(res)
        if len(filename) != 0:
            self.signals.open_file(filename)

    def drop_file(self, viewer, paths):
        filename = paths[0]
        self.signals.open_file(filename)

    def image_update(self, image):
        """Image has updated.

        Parameters
        ----------
        image: `ginga.Astroimage.AstroImage`
            The image.
        """
        self.image_viewer.set_image(image)
        self.setWindowTitle(image.get('name'))

    def quit(self, *args, **kwargs):
        """Shutdown"""
        self.logger.debug('GUI shutting down...')
        self.deleteLater()
