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
        self.signals.Quit.connect(self.quit)
        self.signals.NewImage.connect(self.image_update)
        self.signals.UpdateMesh.connect(self.mesh_viewer.update_mesh)
        self.signals.ProcessStart.connect(self.mesh_viewer.process)

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

        # Menu Bar
        quit_action = QtGui.QAction('&Quit', self)
        quit_action.setStatusTip('Quit application')
        quit_action.triggered.connect(self.signals.Quit)

        open_action = QtGui.QAction('&Open', self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip('Open image')
        open_action.triggered.connect(self.open_file)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(quit_action)

        # Modes
        mode_list = (
            'Textures',
            'Intensity',
            'Spiral Galaxy',
            'Double sided'
        )
        modes = QtGui.QListWidget()
        modes.itemClicked.connect(self.mode_change)
        for mode in mode_list:
            item = QtGui.QListWidgetItem(mode, modes)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)

        # Modifier region
        modifier_hbox = QtGui.QHBoxLayout()
        for w in (modes,):
            modifier_hbox.addWidget(w)

        # Window
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        vbox.setSpacing(1)
        for w in (viewer_hbox, modifier_hbox):
            hw = QtGui.QWidget()
            hw.setLayout(w)
            vbox.addWidget(hw)

        vw = QtGui.QWidget()
        self.setCentralWidget(vw)
        vw.setLayout(vbox)

    def mode_change(self, mode):
        self.logger.debug('mode_change: mode="{}"'.format(mode))
        state = mode.checkState()
        if state == QtCore.Qt.Checked:
            mode.setCheckState(QtCore.Qt.Unchecked)
            state = False
        else:
            mode.setCheckState(QtCore.Qt.Checked)
            state = True
        self.signals.ModeChange(mode.text(), state)

    def open_file(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                ".", "FITS files (*.fits)")
        if isinstance(res, tuple):
            filename = res[0]
        else:
            filename = str(res)
        if len(filename) != 0:
            self.signals.OpenFile(filename)

    def drop_file(self, viewer, paths):
        filename = paths[0]
        self.signals.OpenFile(filename)

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
