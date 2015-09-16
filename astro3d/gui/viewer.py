"""Main UI Viewer
"""

from attrdict import AttrDict

from ..external.qt import (QtGui, QtCore)
from ..external.qt.QtCore import Qt
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

        # Application signals
        self.image_viewer.set_callback('drag-drop', self.drop_file)
        self.signals.Quit.connect(self.quit)
        self.signals.NewImage.connect(self.image_update)
        self.signals.UpdateMesh.connect(self.mesh_viewer.update_mesh)
        self.signals.ProcessStart.connect(self.mesh_viewer.process)

    def _build_gui(self):
        """Construct the app's GUI"""

        ####
        # Setup main content views
        ####

        # Image View
        image_viewer = ViewImage(self.logger)
        self.image_viewer = image_viewer
        image_viewer_widget = image_viewer.get_widget()
        self.setCentralWidget(image_viewer_widget)

        # 3D mesh preview
        self.mesh_viewer = ViewMesh()

        # Setup all the auxiliary gui.
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()

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
            item.setCheckState(QtCore.Qt.Checked)

        # Modifier region
        modifier_hbox = QtGui.QHBoxLayout()
        for w in (modes,):
            modifier_hbox.addWidget(w)

        # Setup main window.

        """
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
        """

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

    def _create_actions(self):
        """Setup the main actions"""
        self.actions = AttrDict()

        quit = QtGui.QAction('&Quit', self)
        quit.setStatusTip('Quit application')
        quit.triggered.connect(self.signals.Quit)
        self.actions.quit = quit

        open = QtGui.QAction('&Open', self)
        open.setShortcut(QtGui.QKeySequence.Open)
        open.setStatusTip('Open image')
        open.triggered.connect(self.open_file)
        self.actions.open = open

        preview_toggle = QtGui.QAction('Mesh View', self)
        preview_toggle.setStatusTip('Open mesh view panel')
        preview_toggle.setCheckable(True)
        preview_toggle.setChecked(False)
        preview_toggle.toggled.connect(self.mesh_viewer.toggle_view)
        self.mesh_viewer.closed.connect(preview_toggle.setChecked)
        self.actions.preview_toggle = preview_toggle

    def _create_menus(self):
        """Setup the main menus"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.actions.open)
        file_menu.addAction(self.actions.quit)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(self.actions.preview_toggle)

    def _create_toolbars(self):
        """Setup the main toolbars"""

    def _create_statusbar(self):
        """Setup the status bar"""
