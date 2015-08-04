"""Main GUI for Hubble 3D Printing.

It allows the user to upload any FITS or JPEG image file and
subsequently convert the image into an STL file, which can then
be printed with a 3D printer.

These are the available textures for regions:

* ``dots``
* ``dots_small``
* ``lines``
* ``smooth`` - Foreground stars or objects to be smoothed over.

The textures for special spiral galaxy processing are fixed:

* ``spiral`` - Represents spiral arms. Same as ``dots``.
* ``disk`` - Represents galactic disk. Same as ``lines``.
* ``remove_star`` - Represents foreground stars to be smoothed over.
  Same as ``smooth``.

These are pre-defined textures for peaks:

* ``crator1`` (single crator) - Stars or galactic center.
* ``crator3`` (a set of 3 crators) - Star clusters.

.. note::

    On Mac, closing a popup window gives you ``modalSession`` warning, see
    https://groups.google.com/forum/#!topic/eventandtaskmanager/L636sUwZudY

"""
from __future__ import division, print_function

# STDLIB
import os
import platform
import sys
import warnings
from collections import OrderedDict

# Anaconda
import numpy as np
from astropy import log
from astropy.io import ascii
from astropy.utils.exceptions import AstropyUserWarning
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# LOCAL
from .star_scenes import (PreviewScene, StarScene, RegionBrushScene,
                          RegionFileScene, ClusterStarScene)
from .wizard import ThreeDModelWizard
from ..utils import model3d
from ..utils import image_utils
from ..utils.textures import TextureMask


_gui_title = 'Astronomy 3D Model'
__version__ = '0.3.0.dev0'
__vdate__ = '13-Apr-2015'
__author__ = 'STScI'


class AstroGUI(QMainWindow):
    """Main window for image display.

    It can have a menu bar and it displays the image.
    All other codes are initialized from this class.
    Its methods primarily exist to interface between
    different objects, specifically by storing changes
    made by the wizard and the `MainPanel` in ``File``
    and ``TextureMask`` objects. Furthermore, it applies
    changes made from the wizard to the ``Image``, thus
    changing the display.

    Parameters
    ----------
    argv
        Arguments from command line.
        'debug' is used to run the debug script
        (see :meth:`run_auto_login_script`).

    Attributes
    ----------
    model3d : `~astro3d.utils.model3d.Model3D`
        This stores info necessary to make the STL.

    transformation : func
        Provides the transformation (linear, log, or sqrt) applied to the
        image before it is displayed. Applied for visualization and display
        purposes only.

    widget : `MainPanel`
        Initialized in :meth:`createWidgets`. Displays the image.

    preview : `PreviewWindow`
        Displays the preview.

    """
    IMG_TRANSFORMATIONS = OrderedDict(
        [(0, 'Linear'), (1, 'Logarithmic'), (2, 'Sqrt')])
    TRANS_FUNCS = {
        'Linear': image_utils.scale_linear,
        'Logarithmic': image_utils.scale_log,
        'Sqrt': image_utils.scale_sqrt}
    MODEL_TYPES = OrderedDict(
        [(0, 'Flat texture map (one-sided only)'),
         (1, 'Smooth intensity map (one-sided)'),
         (2, 'Smooth intensity map (two-sided)'),
         (3, 'Textured intensity map (one-sided)'),
         (4, 'Textured intensity map (two-sided)')])

    def __init__(self, argv=None):
        super(AstroGUI, self).__init__()
        self.setWindowTitle(_gui_title)
        log.info('Started {0} v{1} ({2}) by {3}'.format(
            _gui_title, __version__, __vdate__, __author__))

        self.model3d = None
        self.transformation = self.TRANS_FUNCS['Linear']
        self._model_type = 0
        self._pixmap = None
        self._enable_photutil = True

        self.createWidgets()
        self.move(0, 0)

        if argv and argv[0] == 'debug':
            debug = True
            log.info('running debug script')
            self.run_auto_login_script()
        else:
            debug = False

        screen = QDesktopWidget().screenGeometry()
        wizard = ThreeDModelWizard(self, debug=debug)
        wizard.move(screen.width() // 4, 0)

        self.show()
        self.statusBar().showMessage('')

        # Un-minimize and bring wizard to foreground
        wizard.setWindowState(
            wizard.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        wizard.activateWindow()

    @property
    def model_type(self):
        """Type of 3D model to make."""
        return self._model_type

    @model_type.setter
    def model_type(self, val):
        """Set model type and associated attributes."""
        if val not in self.MODEL_TYPES:
            raise ValueError('Invalid model type')

        self._model_type = val

        if self.model3d is not None:
            # Single- or double-sided
            if val in (0, 1, 3):
                self.model3d.double_sided = False
            else:
                self.model3d.double_sided = True

            # Add textures?
            if val in (0, 3, 4):
                self.model3d.has_texture = True
            else:
                self.model3d.has_texture = False

            # Add intensity?
            if val > 0:
                self.model3d.has_intensity = True
            else:
                self.model3d.has_intensity = False

    # GUI Menu Bar

    def createWidgets(self):
        """Create menus."""
        self.widget = MainPanel(self)
        self.preview = PreviewWindow(self)
        self.mainLayout = QHBoxLayout()
        self.mainLayout.addWidget(self.widget)
        self.mainLayout.addWidget(self.preview)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        # Special menu for Mac or it will be absorbed into built-in menu
        # system that is not intuitive. It is important to avoid naming
        # the actions "Quit" and "About".
        if platform.system().lower() == 'darwin':
            appMenu = self.menuBar().addMenu('Astro3D')
            fileQuitAction = self.createAction(
                'End Session', QCoreApplication.instance().quit,
                'Quit the session')
            fileMenu = appMenu.addMenu('File')
            self.addActions(fileMenu, (fileQuitAction, ))
            helpAboutAction = self.createAction(
                'Software Info', AboutPopup(self).exec_,
                'About the software')
            helpMenu = appMenu.addMenu('Help')
            self.addActions(helpMenu, (helpAboutAction, ))
        else:
            fileQuitAction = self.createAction(
                '&Quit', QCoreApplication.instance().quit, 'Quit the session')
            fileMenu = self.menuBar().addMenu('&File')
            self.addActions(fileMenu, (fileQuitAction, ))
            helpAboutAction = self.createAction(
                '&About', AboutPopup(self).exec_, 'About the software')
            helpMenu = self.menuBar().addMenu('&Help')
            self.addActions(helpMenu, (helpAboutAction, ))

    def addActions(self, target, actions):
        """Adds an arbitary number of actions to a menu.
        A value of None will create a separator.

        """
        for action in actions:
            if action == None:
                target.addSeparator()
            elif isinstance(action, QMenu):
                target.addMenu(action)
            else:
                target.addAction(action)

    def createAction(self, text, slot=None, tip=None, checkable=False):
        """Creates an action with given text, slot (method), tooltip,
        and checkable.

        """
        action = QAction(text, self)
        if tip != None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot != None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action

    # Basic Actions

    def fileLoad(self):
        """Load image file from FITS, JPEG, or TIFF.

        After file data is converted to Numpy array, it uses
        :func:`~astro3d.utils.image_utils.makeqimage` to create
        a ``QImage``. Then, it creates a ``QPixmap``, which is passed
        to the ``widget``. A `~astro3d.utils.model3d.Model3D`
        object is also created to store information.

        """
        fname = QFileDialog.getOpenFileName(
            self, '3D Model Creator - Load Image', '',
            'Image files (*.fits *.FITS *.jpg *.JPG *.jpeg *JPEG '
            '*.tif *.TIF *.tiff *.TIFF)')
        fnamestr = str(fname)

        if fname.isEmpty():
            return

        elif fnamestr.endswith(('fits', 'FITS')):
            self.model3d = model3d.Model3D.from_fits(fnamestr)

        else:  # color display
            # DISABLED - Used for grayscale display
            #rgb_popup = RGBScalingPopup(self)
            #rgb_popup.exec_()
            #data = image_utils.img2array(
            #    fnamestr, rgb_scaling=rgb_popup.rgb_scaling)[::-1, :]

            self.transformation = None  # Not applicable to RGB layers
            self._enable_photutil = False  # Photometry fails for this format
            self.model3d = model3d.Model3D.from_rgb(fnamestr)

        self._pixmap = self.widget.addImage(QPixmap().fromImage(
            image_utils.makeqimage(self.model3d.input_image,
                                   self.transformation,
                                   self.widget.scene_size)))
        self.preview.scene.set_model(self.model3d)
        self.statusBar().showMessage(os.path.basename(fnamestr))

    def fileSave(self, height=150.0, depth=10, split_halves=True,
                 save_all=True):
        """Save files and make 3D model.

        See :func:`~astro3d.utils.meshes.to_mesh` for more details.

        Parameters
        ----------
        height : float
            Height of the model.

        depth : int
            Depth of back plate.

        split_halves : bool
            Split 3D model into two halves.

        save_all : bool
            Also save regions and star clusters to files.

        Returns
        -------
        msg : str
            Status message.

        """
        self.model3d.height = height
        self.model3d.base_thickness = depth

        path = QFileDialog.getSaveFileName(
            self, '3D Model Creator - Save STL', '')

        if path.isEmpty():
            return 'ERROR - No filename given!'

        fname = str(path)
        prefix = os.path.join(os.path.dirname(fname),
                              os.path.basename(fname).split('.')[0])

        if save_all:
            self.model3d.save_texture_masks(prefix)
            self.model3d.save_peaks(prefix)

        self.model3d.make()
        self.model3d.save_stl(fname, split_halves=split_halves)

        return 'Done!'

    # Region Editing

    def clearRegion(self):
        """Clears a region that is currently being drawn."""
        self.widget.clear_region()

    def renameRegion(self, region):
        """Rename one region."""
        text, ok = QInputDialog.getText(
            self, 'Rename region', 'New region name:')

        if ok:
            region.description = str(text)

    def editRegion(self, key, idx):
        """Edit one existing region.

        Parameters
        ----------
        key : str
            Region key.

        idx : int
            Index of the region in the existing list of regions.

        """
        region = self.model3d.region_masks[key][idx]
        self.widget.region_loader(region, overwrite=(key, idx))

    def deleteRegion(self, key, region):
        """Deletes a previously drawn region, removing it from
        ``regions`` and from the screen.

        Parameters
        ----------
        key : str

        region : `~astro3d.gui.astroObjects.Region` or list

        """
        if isinstance(region, list):
            for reg in region:
                self.deleteRegion(key, reg)
        else:
            self.hideRegion(region)
            self.model3d.region_masks[key].remove(region)

    def handleRegionVisibility(self, region):
        """Auto hide/show a region."""
        if not isinstance(region, Region):
            map(self.handleRegionVisibility, region)
        elif region.visible:  # Hide
            self.hideRegion(region)
        else:  # Show
            self.showRegion(region)

    def showRegion(self, region):
        """Displays the hidden region(s) passed in as the
        region parameter.

        Parameters
        ----------
        #region : `~astro3d.gui.astroObjects.Region` or list
        region : `~astro3d.textures.TextureMask` or list

        """
        if not isinstance(region, TextureMask):
            region = filter(lambda reg: reg.visible == False, region)
            map(self.showRegion, region)
        elif not region.visible:
            self.widget.display_region(region)
            region.visible = True

    def hideRegion(self, region):
        """Hides the displayed region(s) passed in as the
        region parameter.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region` or list

        """
        if not isinstance(region, Region):
            region = filter(lambda reg: reg.visible, region)
            map(self.hideRegion, region)
        elif region.visible:
            self.widget.hide_region(region)
            region.visible = False

    def drawRegion(self, key):
        """Tell the `MainPanel` to switch to the interactive
        region drawing ``QGraphicsScene``.

        Parameters
        ----------
        key : str
            Type of the region to be drawn.

        """
        self.widget.region_drawer(key)

    def highlightRegion(self, region):
        """Highlight selected region(s)."""
        self.widget.highlight_region(region)

    def loadRegion(self):
        """Load region(s) from file."""
        flist = QFileDialog.getOpenFileNames(
            self, '3D Model Creator - Load Regions', '', 'Region files (*.fits)')

        if flist.isEmpty():
            return

        regions = [TextureMask.read(
            str(fname), shape=(self._pixmap.height(),
                               self._pixmap.width())) for fname in flist]

        if len(regions) == 1:
            reg = regions[0]
        else:
            reg = regions
        self.widget.region_loader(reg)

        return ','.join([reg.description for reg in regions])

    def saveRegion(self):
        """Store selected region.

        It obtains the name (string) and region (QPolygonF)
        from the `MainPanel`, and in turn creates a
        `~astro3d.gui.astroObjects.Region` object,
        which is added to ``regions``.

        """
        names, regions, descriptions, overwrite = self.widget.save_region()

        if not isinstance(regions, list):
            names = [names]
            regions = [regions]
            descriptions = [descriptions]
        elif overwrite is not None:
            raise ValueError('Cannot overwrite multiple regions.')

        for key, region, descrip in zip(names, regions, descriptions):
            reg = TextureMask(region, key)
            reg.description = descrip

            key = key.lower()

            if key in self.model3d.allowed_textures():
                if overwrite is not None:
                    okey, oidx = overwrite
                    old_reg = self.model3d.region_masks[okey][oidx]
                    self.hideRegion(old_reg)
                    self.model3d.region_masks[okey][oidx] = reg
                else:
                    self.model3d.region_masks[key].append(reg)
            else:
                warnings.warn('{0} is not a valid region texture'.format(key),
                              AstropyUserWarning)
                continue

            log.info('{0} ({1}) saved'.format(key, descrip))
            self.showRegion(reg)

    def findRegion(self, lo, hi):
        """Automatically find spiral arms and gas.

        Parameters
        ----------
        lo, hi : float
            Percentiles for gas and spiral arms, respectively.

        """
        keys = [self.model3d.dots_key, self.model3d.small_dots_key]

        # Delete any existing spiral arms and gas
        for key in keys:
            reglist = self.model3d.region_masks[key]
            self.deleteRegion(key, reglist)

        # Auto find and show results
        shape = (self._pixmap.height(), self._pixmap.width())
        self.model3d.auto_spiralarms(
            shape=shape, percentile_hi=hi, percentile_lo=lo)
        for key in keys:
            self.showRegion(self.model3d.region_masks[key][0])

    # Image Transformations

    def remake_image(self):
        """Any time a change is made to the image being displayed,
        this can be called to recreate the relevant ``pixmap`` and
        change the image display.

        """
        self._pixmap = QPixmap().fromImage(
            image_utils.makeqimage(self.model3d.input_image,
                                   self.transformation,
                                   self.widget.scene_size))
        self.widget.setImage()

    def setTransformation(self, trans='Linear'):
        """Use methods from ``astropy.visualization`` to scale
        image intensity values, which allows better visualization
        of the images.

        .. note::

            It is important to note that the scaling is for display
            purposes only and has no effect on the engine.

        Parameters
        ----------
        trans : {'Linear', 'Logarithmic', 'Sqrt'}
            Transformation name.

        """
        self.transformation = self.TRANS_FUNCS[trans]
        self.remake_image()

    # OBSOLETE - Kept for debugging.
    #def resizeImage(self, width, height):
    #    """Uses PIL (or Pillow) to resize an array to the
    #    given dimensions.
    #    The width and height are given by the user in the
    #    wizard."""
    #    image = Image.fromarray(self.file.data).resize((width, height))
    #    self.file.data = np.array(image, dtype=np.float64)

    # Clusters editing

    def find_clusters(self, n):
        """Retrieve locations of star clusters using
        :func:`~astro3d.utils.model3d.find_peaks`, then displays
        them on the screen for the user to see using the
        `~astro3d.gui.star_scenes.ClusterStarScene`.
        This action is similar to :func:`matplotlib.pyplot.scatter`.

        Parameters
        ----------
        n : int
           Max number of objects to find.

        """
        if not self._enable_photutil:
            log.info('Photometry is disabled.')
            return

        log.info('Finding star clusters, please be patient...')
        self.model3d.find_peaks(self.model3d.clusters_key, n)
        self.widget.cluster_find()

    def load_clusters(self):
        """Retrieve locations of star clusters from file."""
        fname = QFileDialog.getOpenFileName(
            self, '3D Model Creator - Load Clusters', '',
            'Text files (*.txt *.dat)')

        if fname.isEmpty():
            return

        self.model3d.load_peaks(self.model3d.clusters_key, str(fname))
        self.widget.cluster_find()

    def manual_clusters(self):
        """Select locations from display."""
        self.widget.cluster_find()

    def save_clusters(self):
        """Done selecting star clusters."""
        self.widget.save_clusters()

    # Stars editing

    def find_stars(self, n):
        """Same as :meth:`find_clusters` but objects found are
        registered as individual stars by GUI.

        """
        if not self._enable_photutil:
            log.info('Photometry is disabled.')
            return

        log.info('Finding stars, please be patient...')
        self.model3d.find_peaks(self.model3d.stars_key, n)
        self.widget.star_find()

    def load_stars(self):
        """Retrieve locations of stars from file."""
        fname = QFileDialog.getOpenFileName(
            self, '3D Model Creator - Load Stars', '',
            'Text files (*.txt *.dat)')

        if fname.isEmpty():
            return

        self.model3d.load_peaks(self.model3d.stars_key, str(fname))
        self.widget.star_find()

    def manual_stars(self):
        """Select locations from display."""
        self.widget.star_find()

    def save_stars(self):
        """Done selecting stars."""
        self.widget.save_stars()

    # Preview
    def render_preview(self):
        """Preview model."""
        self.statusBar().showMessage('Rendering preview, please wait...')
        self.repaint()
        self.model3d.make()
        self.preview.draw()
        self.statusBar().showMessage('Preview is on the right')

    # For DEBUG PURPOSES ONLY
    def run_auto_login_script(self):
        """Auto-populate data collected by GUI so it can skip
        to the last page in the wizard.

        .. note::

            This needs updating to work. Currently not used.

        """
        #from astropy.table import Table
        #fname = 'data/ngc3344_uvis_f555w_sci.fits'
        #name = os.path.basename(fname)
        #data = fits.getdata(fname)
        #image = QPixmap().fromImage(
        #    makeqimage(data, self.transformation, self.widget.scene_size))
        #pic = self.widget.addImage(image)
        #self.file = File(data, pic)
        #self.resizeImage(2000, 2000)
        #self.setTransformation('Logarithmic')
        #disk = Region.fromfile('data/Disk.reg')
        #spiral1 = Region.fromfile('data/Spiral1.reg')
        #spiral2 = Region.fromfile('data/Spiral2.reg')
        #spiral3 = Region.fromfile('data/Spiral3.reg')
        #star1 = Region.fromfile('data/Star1.reg')
        #star2 = Region.fromfile('data/Star2.reg')
        #regions = [
        #    reg.scaledRegion(1/float(self.file.scale()))
        #    for reg in [spiral1, spiral2, spiral3, disk, star1, star2]]
        #for reg in regions:
        #    self.regions.append(reg)
        #    if len(self.regions) < 4:
        #        self.file.spiralarms.append(reg)
        #    elif len(self.regions) == 4:
        #        self.file.disk = reg
        #    else:
        #        self.file.stars.append(reg)
        #t = Table.read('data/clusterpath', format='ascii')
        #self.file.peaks['clusters'] = t

        raise NotImplementedError('Script needs updating')


class AboutPopup(QDialog):
    """Popup window to display version info."""
    def __init__(self, parent=None):
        super(AboutPopup, self).__init__(parent)
        self.setWindowTitle('About')

        ver = QLabel('Version {0}\n({1})\n\nBy {2}\n\n\n'.format(
                __version__, __vdate__, __author__))
        ver.setAlignment(Qt.AlignCenter)

        buttonbox = QDialogButtonBox(self)
        buttonbox.setStandardButtons(QDialogButtonBox.Ok)
        self.connect(buttonbox, SIGNAL('accepted()'), self.accept)

        vbox = QVBoxLayout(self)
        vbox.addWidget(ver)
        vbox.addWidget(buttonbox)

        self.setLayout(vbox)


class _RGBScalingPopup(QDialog):
    """Popup window for user to enter RGB scaling factors.

    .. note:: Not used anymore.

    """
    def __init__(self, parent=None):
        super(RGBScalingPopup, self).__init__(parent)
        self.setWindowTitle('RGB Scaling')
        self.rgb_scaling = None
        self._colors = ['red', 'green', 'blue']
        self._texts = {}

        label = QLabel('Enter scaling factors for intensity conversion:')
        label.setWordWrap(True)

        vbox = QVBoxLayout()
        vbox.addStretch()
        vbox.addWidget(label)

        for color in self._colors:
            self._texts[color] = QLineEdit('1')
            self._texts[color].setFixedWidth(80)
            rbox = QHBoxLayout()
            rbox.addWidget(self._texts[color])
            rbox.addWidget(QLabel(color))
            rbox.addStretch()
            vbox.addLayout(rbox)

        buttonbox = QDialogButtonBox(self)
        buttonbox.setStandardButtons(QDialogButtonBox.Ok)
        self.connect(buttonbox, SIGNAL('accepted()'), self.accept)

        vbox.addStretch()
        vbox.addWidget(buttonbox)
        self.setLayout(vbox)

    def accept(self):
        """Get scaling factors when user press OK button."""
        self.rgb_scaling = []

        for color in self._colors:
            try:
                fac = float(self._texts[color].text())
            except ValueError as e:
                warnings.warn('Invalid scaling for {0}: {1}. '
                'Using default'.format(color, e), AstropyUserWarning)
                fac = 1.0

            self.rgb_scaling.append(fac)

        QDialog.accept(self)


class MainPanel(QWidget):
    """The central widget for `AstroGUI`.

    It contains a ``QGraphicsView`` which can show several
    ``QGraphicsScenes``. The primary purpose of this widget
    is to interface between the `AstroGUI` and the ``QGraphicsScenes``
    in order to enable viewing images, drawing regions, and
    finding star clusters. In addition to its ``QGraphicsView``,
    it also contains a non-interactive ``main_scene``, along with
    a pointer to the current scene.

    """
    def __init__(self, parent=None):
        super(MainPanel, self).__init__(parent)

        # Resize based on screen size
        screen = QDesktopWidget().screenGeometry()
        self._GEOM_SZ = int(min(screen.width() * 0.45, screen.height() * 0.9))
        self._SCENE_SZ = int(self._GEOM_SZ * 0.95)

        self.parent = parent
        self.view = QGraphicsView(self)
        self.view.setAlignment(Qt.AlignCenter)

        layout = QGridLayout(self)
        layout.addWidget(self.view, 0, 0)
        self.setLayout(layout)
        self.size = QSize(self._GEOM_SZ, self._GEOM_SZ)
        self.view.setMinimumSize(self.size)
        self.resize(self.size)

        self.scene_size = QSize(self._SCENE_SZ, self._SCENE_SZ)
        self.main_scene = StarScene(self, self._SCENE_SZ, self._SCENE_SZ)
        self.current_scene = self.main_scene
        self.view.setScene(self.current_scene)

        self.show()

    def addImage(self, pixmap):
        """Adds the given ``pixmap`` to the display.

        Parameters
        ----------
        pixmap : QPixmap

        Returns
        -------
        scaledPixmap : QPixmap
            A scaled version for storage in the appropriate
            `~astro3d.gui.astroObjects.File` object.

        """
        return self.main_scene.addImg(pixmap)

    def setImage(self):
        """Sets the image to currently selected image."""
        self.main_scene.addImg(self.parent._pixmap)
        self.update_scene(self.main_scene)

    def update_scene(self, scene):
        """A simple helper method. Sets the current scene to the
        input and changes the view.

        Parameters
        ----------
        scene : QGraphicsScene

        """
        self.current_scene = scene
        self.view.setScene(self.current_scene)

    # Region Editing

    def region_drawer(self, name):
        """Sets the scene to an interactive drawing scene,
        allowing the user to draw regions.

        Parameters
        ----------
        name : str

        """
        draw_scene = RegionBrushScene(self, self.parent._pixmap, name)
        self.update_scene(draw_scene)

    def region_loader(self, reg, overwrite=None):
        """Sets the scene to display region loaded from file.

        Parameters
        ----------
        reg : `~astro3d.gui.astroObjects.Region` or list

        overwrite : tuple or `None`
            ``(key, index)`` to identify am existing region to replace.

        """
        if isinstance(reg, list):
            draw_scene = RegionFileScene(self, self.parent._pixmap, reg)
        else:
            draw_scene = RegionBrushScene.from_region(
                self, self.parent._pixmap, reg)
        draw_scene.overwrite = overwrite
        self.update_scene(draw_scene)

    def save_region(self):
        """Sets the scene to the non-interactive main scene.
        Passes region information to `AstroGUI` to save.

        Returns
        -------
        name : str or list

        region : QPolygonF or list

        description : str or list

        overwrite
            See :meth:`region_loader`.

        """
        name, region, description = self.current_scene.getRegion()
        overwrite = self.current_scene.overwrite
        self.update_scene(self.main_scene)
        self.parent.statusBar().showMessage('')
        return name, region, description, overwrite

    def clear_region(self):
        """Clears the currently displayed region."""
        self.current_scene.clear()
        self.parent.statusBar().showMessage('')

    def display_region(self, region):
        """Shows a region on top of the non-interactive main scene.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region`

        """
        self.main_scene.addReg(region)
        self.update_scene(self.main_scene)

    def hide_region(self, region):
        """Hides a region from the non-interactive main scene.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region`

        """
        self.main_scene.delReg(region)
        self.update_scene(self.main_scene)

    def highlight_region(self, region):
        """Highlight selected region(s) in a different color."""
        self.main_scene.draw(selected=region)
        self.update_scene(self.main_scene)

    # Cluster Editing

    def cluster_find(self):
        """Highlights the locations of clusters to allow
        the user to add new or remove 'invalid' clusters
        (e.g., foreground stars, etc) using the interactive
        `~astro3d.gui.star_scenes.ClusterStarScene`.

        """
        cluster_scene = ClusterStarScene(
            self, self.parent._pixmap, self.parent.model3d,
            self.parent.model3d.clusters_key)
        self.update_scene(cluster_scene)

    def save_clusters(self):
        """Sets the scene to the non-interactive main scene."""
        self.main_scene.set_clusters(
            self.parent.model3d.peaks[self.parent.model3d.clusters_key],
            self.parent.model3d.input_image.shape[0])
        self.update_scene(self.main_scene)

    # Stars Editing

    def star_find(self):
        """Like :meth:`cluster_find` but for stars."""
        markstar_scene = ClusterStarScene(
            self, self.parent._pixmap, self.parent.model3d,
            self.parent.model3d.stars_key)
        self.update_scene(markstar_scene)

    def save_stars(self):
        """Sets the scene to the non-interactive main scene."""
        self.main_scene.set_stars(
            self.parent.model3d.peaks[self.parent.model3d.stars_key],
            self.parent.model3d.input_image.shape[0])
        self.update_scene(self.main_scene)


class PreviewWindow(QWidget):
    """Class to handle preview of the final product.

    Preview window will show intensity as monochrome image with
    texture masks overlay in different colors.

    """
    def __init__(self, parent=None):
        super(PreviewWindow, self).__init__(parent)

        # Resize based on screen size
        screen = QDesktopWidget().screenGeometry()
        self._GEOM_SZ = int(min(screen.width() * 0.45, screen.height() * 0.9))
        self._SCENE_SZ = int(self._GEOM_SZ * 0.95)

        self.parent = parent
        self.view = QGraphicsView(self)
        self.view.setAlignment(Qt.AlignCenter)

        layout = QGridLayout(self)
        layout.addWidget(self.view, 0, 0)
        self.setLayout(layout)
        self.size = QSize(self._GEOM_SZ, self._GEOM_SZ)
        self.view.setMinimumSize(self.size)
        self.resize(self.size)

        self.scene = PreviewScene(self, self._SCENE_SZ, self._SCENE_SZ)
        self.view.setScene(self.scene)

        self.show()

    def draw(self):
        """Render preview."""
        self.scene.draw()


def main(argv):
    """Execute the GUI."""
    app = QApplication(argv)
    window = AstroGUI(argv[1:])
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)
