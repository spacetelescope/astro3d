"""Main UI Viewer
"""
from functools import partial
from os.path import dirname

from attrdict import AttrDict
from ginga.AstroImage import AstroImage


from ..util.logger import make_null_logger
from qtpy import (QtCore, QtGui, QtWidgets)
from . import signaldb
from .qt import (
    ImageView,
    InfoBox,
    InstructionViewer,
    LayerManager,
    OverlayView,
    Parameters,
    ShapeEditor,
    ViewMesh,
)
from .config import config

# Configure logging
logger = make_null_logger(__name__)

# Supported image formats
SUPPORT_IMAGE_FORMATS = (
    'Images ('
    '*.fits'
    ' *.jpg'
    ' *.jpeg*'
    ' *.png'
    ' *.gif'
    ' *.tif*'
    ' *.bmp'
    ');;Uncommon ('
    '*.fpx'
    ' *.pcd'
    ' *.pcx'
    ' *.pixar'
    ' *.ppm'
    ' *.sgi'
    ' *.tga'
    ' *.xbm'
    ' *.xpm'
    ')'
)

# Shortcuts
Qt = QtCore.Qt
GTK_MainWindow = QtWidgets.QMainWindow


__all__ = ['MainWindow']


class Image(AstroImage):
    """Image container"""


class MainWindow(GTK_MainWindow):
    """Main Viewer

    Parameters
    ----------
    model: astro3d.gui.model.Model
        The gui data model

    parent: Qt object
        Parent Qt object
    """
    def __init__(self, model, parent=None):
        super(MainWindow, self).__init__(parent)
        self.model = model

        signaldb.ModelUpdate.set_enabled(False)

        self._build_gui()
        self._create_signals()

    def path_from_dialog(self):
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image file",
            config.get('gui', 'folder_image'),
            SUPPORT_IMAGE_FORMATS
        )
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.open_path(pathname)
            config.set('gui', 'folder_image', dirname(pathname))

    def regionpath_from_dialog(self):
        res = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open Region files",
            config.get('gui', 'folder_regions'),
            "FITS files (*.fits)"
        )
        logger.debug('res="{}"'.format(res))
        if len(res) > 0:
            if isinstance(res, tuple):
                file_list = res[0]
            else:
                file_list = res
            if len(file_list):
                self.model.read_maskpathlist(file_list)
                signaldb.ModelUpdate()
                config.set('gui', 'folder_regions', dirname(file_list[0]))

    def texturepath_from_dialog(self):
        res = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open Texture files",
            config.get('gui', 'folder_textures'),
            "FITS files (*.fits)"
        )
        logger.debug('res="{}"'.format(res))
        if len(res) > 0:
            if isinstance(res, tuple):
                file_list = res[0]
            else:
                file_list = res
            if len(file_list):
                self.model.read_maskpathlist(
                    file_list, container_layer=self.model.textures
                )
                signaldb.ModelUpdate()
                config.set('gui', 'folder_textures', dirname(file_list[0]))

    def catalogpath_from_dialog(self, catalog_item=None):
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Catalog",
            config.get('gui', 'folder_regions')
        )
        if isinstance(res, tuple):
            pathname = res[0]
        else:
            pathname = str(res)
        if len(pathname) != 0:
            self.model.read_stellar_catalog(
                pathname, catalog_item=catalog_item
            )
            config.set('gui', 'folder_regions', dirname(res[0]))
            signaldb.ModelUpdate()

    def starpath_from_dialog(self):
        self.catalogpath_from_dialog(self.model.stars_catalogs)

    def clusterpath_from_dialog(self):
        self.catalogpath_from_dialog(self.model.cluster_catalogs)

    def path_by_drop(self, viewer, paths):
        pathname = paths[0]
        self.open_path(pathname)

    def save_all_from_dialog(self):
        """Specify folder to save all info"""
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Specify prefix to save all as',
            config.get('gui', 'folder_save')
        )
        logger.debug('result="{}"'.format(result))
        if len(result) > 0:
            if isinstance(result, tuple):
                path = result[0]
            else:
                path = result
            signaldb.ProcessFinish.connect(
                partial(self.save, path),
                single_shot=True
            )
            config.set('gui', 'folder_save', dirname(path))
            self.force_update()

    def open_path(self, pathname):
        """Open the image from pathname"""
        self.model.read_image(pathname)
        self.image = Image(logger=logger)
        self.image.set_data(self.model.image)
        self.image_update(self.image)

    def image_update(self, image):
        """Image has updated.

        Parameters
        ----------
        image: `ginga.Astroimage.AstroImage`
            The image.
        """
        self.image_viewer.set_image(image)
        self.setWindowTitle(image.get('name'))
        signaldb.ModelUpdate()

    def save(self, prefix, mesh=None, model3d=None):
        """Save all info to the prefix

        Parameters
        ----------
        prefix: str
            The path prefix to save all the model files to.

        mesh: dict
            Not used, but required due to being called
            by the ProcessFinish signal.

        model3d: Model3D
            The model which created the mesh.
            If None, use the inherent model3d.
        """
        if model3d is None:
            try:
                model3d = self.model.model3d
            except AttributeError:
                return
        model3d.write_all_masks(prefix)
        model3d.write_all_stellar_tables(prefix)
        model3d.write_stl(prefix)

    def force_update(self):
        signaldb.ModelUpdate.set_enabled(True, push=True)
        try:
            signaldb.ModelUpdate()
        except Exception as e:
            logger.warning('Processing error: "{}"'.format(e))
        finally:
            signaldb.ModelUpdate.reset_enabled()

    def quit(self, *args, **kwargs):
        """Shutdown"""
        logger.debug('GUI shutting down...')
        self.model.quit()
        self.mesh_viewer.close()
        self.instruction_viewer.close()
        config.save()
        self.deleteLater()

    def auto_reprocessing_state(self):
        return signaldb.ModelUpdate.enabled

    def toggle_auto_reprocessing(self):
        state = signaldb.ModelUpdate.enabled
        signaldb.ModelUpdate.set_enabled(not state)

    def _build_gui(self):
        """Construct the app's GUI"""

        ####
        # Setup main content views
        ####

        # Image View
        image_viewer = ImageView()
        self.image_viewer = image_viewer
        image_viewer.set_desired_size(512, 512)
        image_viewer_widget = image_viewer.get_widget()
        self.setCentralWidget(image_viewer_widget)

        # Region overlays
        self.overlay = OverlayView(
            parent=image_viewer,
            model=self.model
        )

        # Basic instructions window
        self.instruction_viewer = InstructionViewer()

        # 3D mesh preview
        self.mesh_viewer = ViewMesh()

        # The Layer manager
        self.layer_manager = LayerManager()
        self.layer_manager.setModel(self.model)
        layer_dock = QtWidgets.QDockWidget('Layers', self)
        layer_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        layer_dock.setWidget(self.layer_manager)
        self.addDockWidget(Qt.RightDockWidgetArea, layer_dock)
        self.layer_dock = layer_dock

        # The Shape Editor
        self.shape_editor = ShapeEditor(
            surface=image_viewer,
            canvas=self.overlay.canvas
        )
        shape_editor_dock = QtWidgets.QDockWidget('Shape Editor', self)
        shape_editor_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        shape_editor_dock.setWidget(self.shape_editor)
        self.addDockWidget(Qt.LeftDockWidgetArea, shape_editor_dock)
        self.shape_editor_dock = shape_editor_dock

        # Parameters
        self.parameters = Parameters(
            parent=self,
            model=self.model
        )
        parameters_dock = QtWidgets.QDockWidget('Parameters', self)
        parameters_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        parameters_dock.setWidget(self.parameters)
        self.addDockWidget(Qt.LeftDockWidgetArea, parameters_dock)
        self.parameters_dock = parameters_dock

        # The process busy dialog.
        process_busy = QtWidgets.QProgressDialog(
            'Creating Model',
            'Abort',
            0, 1
        )
        process_busy.reset()
        process_busy.setMinimumDuration(0)
        process_busy.setWindowModality(Qt.ApplicationModal)
        self.process_busy = process_busy

        # General Info Dialog
        self.info_box = InfoBox()

        # Setup all the auxiliary gui.
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()

    def _create_actions(self):
        """Setup the main actions"""
        self.actions = AttrDict()

        # Preferences
        auto_reprocess = QtWidgets.QAction('Auto Reprocess', self)
        auto_reprocess.setStatusTip('Enable/disable automatic 3D processing')
        auto_reprocess.setCheckable(True)
        auto_reprocess.setChecked(self.auto_reprocessing_state())
        auto_reprocess.toggled.connect(self.toggle_auto_reprocessing)
        self.actions.auto_reprocess = auto_reprocess

        quit = QtWidgets.QAction('&Quit', self)
        quit.setStatusTip('Quit application')
        quit.triggered.connect(signaldb.Quit)
        self.actions.quit = quit

        open = QtWidgets.QAction('&Open', self)
        open.setShortcut(QtGui.QKeySequence.Open)
        open.setStatusTip('Open image')
        open.triggered.connect(self.path_from_dialog)
        self.actions.open = open

        regions = QtWidgets.QAction('&Regions', self)
        regions.setShortcut('Ctrl+R')
        regions.setStatusTip('Open Regions')
        regions.triggered.connect(self.regionpath_from_dialog)
        self.actions.regions = regions

        stars = QtWidgets.QAction('Stars', self)
        stars.setShortcut('Shift+Ctrl+S')
        stars.setStatusTip('Open a stellar table')
        stars.triggered.connect(self.starpath_from_dialog)
        self.actions.stars = stars

        clusters = QtWidgets.QAction('Stellar &Clusters', self)
        clusters.setShortcut('Ctrl+C')
        clusters.setStatusTip('Open a stellar clusters table')
        clusters.triggered.connect(self.clusterpath_from_dialog)
        self.actions.clusters = clusters

        textures = QtWidgets.QAction('&Textures', self)
        textures.setShortcut('Ctrl+T')
        textures.setStatusTip('Open Textures')
        textures.triggered.connect(self.texturepath_from_dialog)
        self.actions.textures = textures

        save_all = QtWidgets.QAction('&Save', self)
        save_all.setShortcut(QtGui.QKeySequence.Save)
        save_all.triggered.connect(self.save_all_from_dialog)
        self.actions.save_all = save_all

        preview_toggle = QtWidgets.QAction('Mesh View', self)
        preview_toggle.setShortcut('Ctrl+V')
        preview_toggle.setStatusTip('Open mesh view panel')
        preview_toggle.setCheckable(True)
        preview_toggle.setChecked(False)
        preview_toggle.toggled.connect(self.mesh_viewer.toggle_view)
        self.mesh_viewer.closed.connect(preview_toggle.setChecked)
        self.actions.preview_toggle = preview_toggle

        instruction_toggle = QtWidgets.QAction('&Instructions', self)
        instruction_toggle.setShortcut('Ctrl+I')
        instruction_toggle.setStatusTip('Open Instruction Window')
        instruction_toggle.setCheckable(True)
        instruction_toggle.setChecked(False)
        instruction_toggle.toggled.connect(self.instruction_viewer.toggle_view)
        self.instruction_viewer.closed.connect(instruction_toggle.setChecked)
        self.actions.instruction_toggle = instruction_toggle

        reprocess = QtWidgets.QAction('Reprocess', self)
        reprocess.setShortcut('Shift+Ctrl+R')
        reprocess.setStatusTip('Reprocess the model')
        reprocess.triggered.connect(self.force_update)
        self.actions.reprocess = reprocess

    def _create_menus(self):
        """Setup the main menus"""
        from sys import platform
        if platform == 'darwin':
            menubar = QtWidgets.QMenuBar()
        else:
            menubar = self.menuBar()
        self.menubar = menubar

        # File menu
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.actions.open)
        file_menu.addAction(self.actions.regions)
        file_menu.addAction(self.actions.clusters)
        file_menu.addAction(self.actions.stars)
        file_menu.addAction(self.actions.textures)
        file_menu.addSeparator()
        file_menu.addAction(self.actions.save_all)
        file_menu.addAction(self.actions.quit)

        view_menu = menubar.addMenu('View')
        view_menu.addAction(self.actions.instruction_toggle)
        view_menu.addAction(self.actions.preview_toggle)
        view_menu.addAction(self.layer_dock.toggleViewAction())

    def _create_toolbars(self):
        """Setup the main toolbars"""

    def _create_statusbar(self):
        """Setup the status bar"""

    def _create_signals(self):
        """Setup the overall signal structure"""
        self.image_viewer.set_callback('drag-drop', self.path_by_drop)

        self.process_busy.canceled.connect(signaldb.ProcessForceQuit)

        signaldb.CatalogFromFile.connect(self.catalogpath_from_dialog)

        signaldb.CreateGasSpiralMasks.connect(
            self.parameters.create_gas_spiral_masks
        )

        signaldb.LayerSelected.connect(self.shape_editor.select_layer)
        signaldb.LayerSelected.connect(self.layer_manager.select_from_object)

        signaldb.NewImage.connect(self.image_update)

        signaldb.ProcessFail.connect(
            lambda text, error: self.process_busy.reset()
        )
        signaldb.ProcessFail.connect(
            lambda text, error: signaldb.ProcessFinish.clear(single_shot=True)
        )
        signaldb.ProcessFail.connect(self.info_box.show_error)
        signaldb.ProcessFinish.connect(self.mesh_viewer.update_mesh)
        signaldb.ProcessFinish.connect(
            lambda x, y: self.process_busy.reset()
        )
        signaldb.ProcessForceQuit.connect(self.process_busy.reset)
        signaldb.ProcessStart.connect(self.mesh_viewer.process)
        signaldb.ProcessStart.connect(
            lambda: self.process_busy.setValue(0)
        )

        signaldb.Quit.connect(self.quit)
