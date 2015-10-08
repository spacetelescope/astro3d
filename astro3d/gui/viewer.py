"""Main UI Viewer
"""

from attrdict import AttrDict
from ginga.AstroImage import AstroImage
from ginga.LayerImage import LayerImage

from ..util.logger import make_logger
from ..external.qt import (QtGui, QtCore)
from ..external.qt.QtCore import Qt
from ..external.qt.QtGui import QMainWindow as GTK_MainWindow
from qt4 import (LayerManager, ViewImage, ViewMesh, ShapeEditor)


__all__ = ['MainWindow']


STAGES = {
    'Intensity':     'intensity',
    'Textures':      'textures',
    'Spiral Galaxy': 'spiral_galaxy',
    'Double-sided':  'double_sided'
}


class Image(AstroImage, LayerImage):
    """Image container"""


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

    def path_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                ".", "FITS files (*.fits)")
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.open_path(pathname)

    def regionpath_from_dialog(self):
        res = QtGui.QFileDialog.getOpenFileNames(
            self, "Open Region files",
            ".", "FITS files (*.fits)"
        )
        self.logger.debug('res="{}"'.format(res))
        if len(res) > 0:
            self.model.read_regionpathlist(res)
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
        self.image = Image(logger=self.logger)
        self.image.load_file(pathname)
        self.image_update(self.image)

    def image_update(self, image):
        """Image has updated.

        Parameters
        ----------
        image: `ginga.Astroimage.AstroImage`
            The image.
        """
        self.image_viewer.set_image(image)
        self.model.image = image.get_data()
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

    def _build_gui(self):
        """Construct the app's GUI"""

        ####
        # Setup main content views
        ####

        # Image View
        image_viewer = ViewImage(self.logger, model=self.model)
        self.image_viewer = image_viewer
        image_viewer_widget = image_viewer.get_widget()
        self.setCentralWidget(image_viewer_widget)

        # 3D mesh preview
        self.mesh_viewer = ViewMesh()

        # The Layer manager
        self.layer_manager = LayerManager(logger=self.logger)
        self.layer_manager.setModel(self.model)
        layer_dock = QtGui.QDockWidget('Layers', self)
        layer_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        layer_dock.setWidget(self.layer_manager)
        self.addDockWidget(Qt.RightDockWidgetArea, layer_dock)
        self.layer_dock = layer_dock

        # The Shape Editor
        self.shape_editor = ShapeEditor(logger=self.logger)
        shape_editor_dock = QtGui.QDockWidget('Shape Editor', self)
        shape_editor_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        shape_editor_dock.setWidget(self.shape_editor)
        self.addDockWidget(Qt.LeftDockWidgetArea, shape_editor_dock)
        self.shape_editor_dock = shape_editor_dock

        # Setup all the auxiliary gui.
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()

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

        regions = QtGui.QAction('&Regions', self)
        regions.setShortcut('Ctrl+R')
        regions.setStatusTip('Open Regions')
        regions.triggered.connect(self.regionpath_from_dialog)
        self.actions.regions = regions

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
        preview_toggle.setShortcut('Ctrl+V')
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

        reprocess = QtGui.QAction('Reprocess', self)
        reprocess.setShortcut('Shift+Ctrl+R')
        reprocess.setStatusTip('Reprocess the model')
        reprocess.triggered.connect(self.signals.ModelUpdate)
        self.actions.reprocess = reprocess

    def _create_menus(self):
        """Setup the main menus"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.actions.open)
        file_menu.addAction(self.actions.regions)
        file_menu.addAction(self.actions.clusters)
        file_menu.addAction(self.actions.stars)
        file_menu.addAction(self.actions.quit)

        view_menu = menubar.addMenu('View')
        view_menu.addAction(self.actions.preview_toggle)
        view_menu.addAction(self.layer_dock.toggleViewAction())

        stage_menu = menubar.addMenu('Stages')
        for name in STAGES:
            stage_menu.addAction(self.actions[STAGES[name]])
        stage_menu.addSeparator()
        stage_menu.addAction(self.actions.reprocess)

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
        self.signals.ModelUpdate.connect(self.image_viewer.update)
