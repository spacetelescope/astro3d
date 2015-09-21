"""Main UI Viewer
"""

from attrdict import AttrDict

from ginga import AstroImage

from ..util.logger import make_logger
from ..external.qt import (QtGui, QtCore)
from ..external.qt.QtCore import Qt
from ..external.qt.QtGui import QMainWindow as GTK_MainWindow
from qt4 import (ViewImage, ViewMesh)


__all__ = ['MainWindow']


STAGES = {
    'Intensity':     'intensity',
    'Textures':      'textures',
    'Spiral Galaxy': 'spiral_galaxy',
    'Double-sided':  'double_sided'
}


class MainWindow(GTK_MainWindow):
    """Main Viewer"""
    def __init__(self, model, signals, logger=None, parent=None):
        super(MainWindow, self).__init__(parent)
        if logger is None:
            logger = make_logger('astro3d viewer')
        self.logger = logger
        self.model = model
        self.signals = signals
        self._build_gui()
        self._create_signals()

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

        """
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

    def path_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                ".", "FITS files (*.fits)")
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.open_path(pathname)

    def maskpath_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileNames(
            self, "Open Mask files",
            ".", "FITS files (*.fits)"
        )
        self.logger.debug('res="{}"'.format(res))
        if len(res) > 0:
            self.model.read_maskpathlist(res)
            self.actions.textures.setChecked(True)
            self.signals.ModelUpdate()

    def starpath_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open Stellar Catalog",
                                                ".")
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.model.read_star_catalog(pathname)
            self.signals.ModelUpdate()

    def clusterpath_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileName(
            self,
            "Open Star Cluster Catalog",
            "."
        )
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.model.read_cluster_catalog(pathname)
            self.signals.ModelUpdate()

    def path_by_drop(self, viewer, paths):
        pathname = paths[0]
        self.open_path(pathname)

    def open_path(self, pathname):
        """Open the image from pathname"""
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(pathname)
        self.image_update(image)
        self.signals.NewImage(image)

    def image_update(self, image):
        """Image has updated.

        Parameters
        ----------
        image: `ginga.Astroimage.AstroImage`
            The image.
        """
        self.image_viewer.set_image(image)
        self.model.set_image(image.get_data())
        self.setWindowTitle(image.get('name'))
        self.signals.ModelUpdate()

    def stagechange(self, *args, **kwargs):
        """Act on a Stage toggle form the UI"""
        self.logger.debug('args="{}" kwargs="{}"'.format(args, kwargs))

        stage = STAGES[self.sender().text()]
        self.model.stages[stage] = args[0]
        self.signals.ModelUpdate()

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
        open.triggered.connect(self.path_from_dialog)
        self.actions.open = open

        masks = QtGui.QAction('&Masks', self)
        masks.setShortcut('Ctrl+M')
        masks.setStatusTip('Open Masks')
        masks.triggered.connect(self.maskpath_from_dialog)
        self.actions.masks = masks

        stars = QtGui.QAction('&Stars', self)
        stars.setShortcut('Ctrl+S')
        stars.setStatusTip('Open a stellar table')
        stars.triggered.connect(self.starpath_from_dialog)
        self.actions.stars = stars

        clusters = QtGui.QAction('Stellar &Clusters', self)
        clusters.setShortcut('Ctrl+C')
        clusters.setStatusTip('Open a stellar clusters table')
        clusters.triggered.connect(self.clusterpath_from_dialog)
        self.actions.clusters = clusters

        preview_toggle = QtGui.QAction('Mesh View', self)
        preview_toggle.setStatusTip('Open mesh view panel')
        preview_toggle.setCheckable(True)
        preview_toggle.setChecked(False)
        preview_toggle.toggled.connect(self.mesh_viewer.toggle_view)
        self.mesh_viewer.closed.connect(preview_toggle.setChecked)
        self.actions.preview_toggle = preview_toggle

        for name in STAGES:
            action = STAGES[name]
            qaction = QtGui.QAction(name, self)
            qaction.setCheckable(True)
            qaction.setChecked(self.model.stages[action])
            qaction.toggled.connect(self.stagechange)
            self.actions[action] = qaction

    def _create_menus(self):
        """Setup the main menus"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.actions.open)
        file_menu.addAction(self.actions.masks)
        file_menu.addAction(self.actions.clusters)
        file_menu.addAction(self.actions.stars)
        file_menu.addAction(self.actions.quit)

        view_menu = menubar.addMenu('View')
        view_menu.addAction(self.actions.preview_toggle)

        stage_menu = menubar.addMenu('Stages')
        for name in STAGES:
            stage_menu.addAction(self.actions[STAGES[name]])

    def _create_toolbars(self):
        """Setup the main toolbars"""

    def _create_statusbar(self):
        """Setup the status bar"""

    def _create_signals(self):
        """Setup the overall signal structure"""
        self.image_viewer.set_callback('drag-drop', self.path_by_drop)
        self.signals.Quit.connect(self.quit)
        self.signals.NewImage.connect(self.image_update)
        self.signals.UpdateMesh.connect(self.mesh_viewer.update_mesh)
        self.signals.ProcessStart.connect(self.mesh_viewer.process)
        self.signals.StageChange.connect(self.stagechange)
