"""Main GUI for Hubble 3D Printing.

It allows the user to upload any FITS or JPEG image file and
subsequently convert the image into an STL file, which can then
be printed with a 3D printer.

"""
from __future__ import division, print_function

# STDLIB
import os
import sys
import warnings
from collections import defaultdict, OrderedDict

# Anaconda
import numpy as np
from astropy import log
from astropy.io import ascii, fits
from astropy.utils.exceptions import AstropyUserWarning
from PIL import Image
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# THIRD-PARTY
import qimage2ndarray as q2a
import photutils.utils as putils

# LOCAL
from .astroObjects import File, Region
from .star_scenes import (StarScene, RegionStarScene, RegionFileScene,
                          ClusterStarScene)
from .wizard import ThreeDModelWizard
from ..utils import imageprep, imageutils


__version__ = '0.1.0.dev'
__vdate__ = '25-Sep-2014'
__author__ = 'STScI'


class AstroGUI(QMainWindow):
    """Main window for image display.

    It can have a menu bar and it displays the image.
    All other codes are initialized from this class.
    Its methods primarily exist to interface between
    different objects, specifically by storing changes
    made by the wizard and the `MainPanel` in ``File``
    and ``Region`` objects. Furthermore, it applies
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
    file : `~astro3d.gui.astroObjects.File`
        Image file object.

    transformation : func
        Provides the transformation (linear, log, or sqrt) applied to the
        image before it is displayed. Applied for visualization and display
        purposes only.

    regions : dict
        A list of all regions the user has created.

    model_type : int
        Type of 3D model to make.

    widget : `MainPanel`
        Initialized in :meth:`createWidgets`. Displays the image.

    """
    _GEOM_X = 300
    _GEOM_Y = 150
    _GEOM_SZ = 800
    _NEWSZ_W = 840
    _NEWSZ_H = 860
    IMG_TRANSFORMATIONS = OrderedDict(
        [(0, 'Linear'), (1, 'Logarithmic'), (2, 'Sqrt')])
    TRANS_FUNCS = {
        'Linear': lambda img: putils.scale_linear(img, percent=99),
        'Logarithmic': lambda img: putils.scale_log(img, percent=99),
        'Sqrt': lambda img: putils.scale_sqrt(img, percent=99)}
    MODEL_TYPES = OrderedDict(
        [(0, 'Flat texture map (one-sided only)'),
         (1, 'Smooth intensity map (one-sided)'),
         (2, 'Smooth intensity map (two-sided)'),
         (3, 'Textured intensity map (one-sided)'),
         (4, 'Textured intensity map (two-sided)')])
    REGION_TEXT = OrderedDict(
        [('disk', 'Disk'), ('spiral', 'Spiral arm'), ('star', 'Star')])

    def __init__(self, argv=None):
        super(AstroGUI, self).__init__()

        self.file = None
        self.transformation = self.TRANS_FUNCS['Linear']
        self.regions = defaultdict(list)
        self.model_type = 0

        self.setGeometry(
            self._GEOM_X, self._GEOM_Y, self._GEOM_SZ, self._GEOM_SZ)
        self.createWidgets()
        self.resize(self._NEWSZ_W, self._NEWSZ_H)

        self.setWindowTitle('3D Star Field Creator')

        if argv and argv[0] == 'debug':
            debug = True
            print('running debug script')
            self.run_auto_login_script()
        else:
            debug = False

        wizard = ThreeDModelWizard(self, debug=debug)
        self.show()

        self.statusBar().showMessage('')

    # GUI Menu Bar

    def createWidgets(self):
        """Create menus."""
        self.widget = MainPanel(self)
        self.setCentralWidget(self.widget)

        fileQuitAction = self.createAction(
            '&Exit', QCoreApplication.instance().quit, 'Exit the session')
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

    def fileLoad(self, height=150.0):
        """Load image file from FITS or JPEG.

        After file data is converted to Numpy array, it uses
        :func:`makeqimage` to create a ``QImage``. Then, it
        creates a ``QPixmap``, which is passed to the ``widget``.
        A `~astro3d.gui.astroObjects.File` object is also created
        to store information.

        Parameters
        ----------
        height : float
            Height of the model.

        """
        fname = QFileDialog.getOpenFileName(
            self, '3D Model Creator - Load Image', '',
            'Image files (*.fits *.FITS *.jpg *.JPG *.jpeg *JPEG)')
        fnamestr = str(fname)

        if fname.isEmpty():
            return
        elif fnamestr.endswith(('fits', 'FITS')):
            data = fits.getdata(fnamestr)
        elif fnamestr.endswith(('jpg', 'JPG', 'jpeg', 'JPEG')):  # grayscale
            data = imageutils.img2array(fnamestr)[::-1,]
        else:
            QMessageBox.warning(self, 'File Error', 'Unsupported file type')
            return

        if data is None:
            QMessageBox.warning(
                self, 'File Error', 'This file does not contain image data')
            return

        name = os.path.basename(fnamestr)
        image = QPixmap()
        image = image.fromImage(
            makeqimage(data, self.transformation, self.widget.size))
        pic = self.widget.addImage(image)
        self.file = File(data, pic, height=height)

    def fileSave(self, split_halves=True, save_all=True):
        """Get save file location and instructs the
        `~astro3d.gui.astroObjects.File` object to construct
        a 3D model.

        See :func:`~astro3d.utils.meshcreator.to_mesh` for more details.

        Parameters
        ----------
        split_halves : bool
            Split 3D model into two halves.

        save_all : bool
            Also save regions and star clusters to files.

        Returns
        -------
        msg : str
            Status message.

        """
        depth = 1  # Depth of back plate
        _ascii = False  # Binary STL

        # Single- or double-sided
        if self.model_type in (0, 1, 3):
            double = False
        else:
            double = True

        # Add textures?
        if self.model_type in (0, 3, 4):
            has_texture = True
        else:
            has_texture = False

        # Add intensity?
        if self.model_type > 0:
            has_intensity = True
        else:
            has_intensity = False

        path = QFileDialog.getSaveFileName(
            self, '3D Model Creator - Save STL', '')
        if path.isEmpty():
            return 'ERROR - No filename given!'
        else:
            fname = str(path)
            path = os.path.dirname(fname)
            prefix = os.path.basename(fname).split('.')[0]

        if save_all:
            for key, reglist in self.regions.iteritems():
                i = 1
                for reg in reglist:
                    rname = os.path.join(
                        path, '{0}_{1}_{2}.reg'.format(prefix, key, i))
                    i += 1
                    reg.save(rname, _file=self.file)
                    log.info('{0} saved'.format(rname))

            if self.file.clusters is not None:
                cname = os.path.join(path, '{0}_clusters.txt'.format(prefix))
                self.file.clusters.write(cname, format='ascii')
                log.info('{0} saved'.format(cname))

        self.file.make_3d(
            fname, depth=depth, double=double, _ascii=_ascii,
            has_texture=has_texture, has_intensity=has_intensity,
            split_halves=split_halves)

        return 'Done!'

    # Region Editing

    def clearRegion(self):
        """Clears a region that is currently being drawn."""
        self.widget.clear_region()

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
            self.regions[key].remove(region)

    def showRegion(self, region):
        """Displays the hidden region(s) passed in as the
        region parameter.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region` or list

        """
        if not isinstance(region, Region):
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

    def loadRegion(self):
        """Load region(s) from file."""
        flist = QFileDialog.getOpenFileNames(
            self, '3D Model Creator - Load Regions', '', 'Region files (*.reg)')

        if flist.isEmpty():
            return

        regions = [Region.fromfile(str(fname), _file=self.file)
                   for fname in flist]
        self.widget.region_loader(regions)

        return ','.join([reg.name for reg in regions])

    def saveRegion(self):
        """Store selected region.

        It obtains the name (string) and region (QPolygonF)
        from the `MainPanel`, and in turn creates a
        `~astro3d.gui.astroObjects.Region` object,
        which is added to ``regions``.

        """
        names, regions = self.widget.save_region()

        if not isinstance(regions, list):
            names = [names]
            regions = [regions]

        for key, region in zip(names, regions):
            reg = Region(key, region)

            # Polygon must have at least 3 points
            if len(reg.points()) < 3:
                warnings.warn(
                    '{0} not saved - Insufficient points'.format(key),
                    AstropyUserWarning)
                continue

            key = key.lower()

            if 'spiral' in key:
                self.file.spiralarms.append(reg)
                self.regions[key].append(reg)
            elif 'disk' in key:
                self.file.disk = reg
                self.regions[key] = [reg]
            elif 'star' in key:
                self.file.stars.append(reg)
                self.regions[key].append(reg)
            else:
                warnings.warn('{0} not found'.format(key), AstropyUserWarning)
                continue

            log.info('{0} saved'.format(key))
            self.showRegion(reg)

    # Image Transformations

    def remake_image(self):
        """Any time a change is made to the image being displayed,
        this can be called to recreate the relevant ``pixmap`` and
        change the image display.

        """
        pic = QPixmap().fromImage(
            makeqimage(self.file.data, self.transformation, self.widget.size))
        self.file.image = pic
        self.widget.setImage()

    def setTransformation(self, trans='Linear'):
        """Use methods from ``photutils.utils`` to scale
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

    def resizeImage(self, width, height):
        """Uses PIL (or Pillow) to resize an array to the
        given dimensions.

        The width and height are given by the user in the
        wizard.

        """
        image = Image.fromarray(self.file.data).resize((width, height))
        self.file.data = np.array(image, dtype=np.float64)
        self.remake_image()

    def find_clusters(self, n):
        """Retrieve locations of star clusters using
        :func:`~astro3d.utils.imageprep.find_peaks`, then displays
        them on the screen for the user to see using the
        `~astro3d.gui.star_scenes.ClusterStarScene`.
        This action is similar to :func:`matplotlib.pyplot.scatter`.

        Parameters
        ----------
        n : int
           Max number of objects to find.

        """
        log.info('Finding star clusters, please be patient...')
        image = self.file.data
        scale = self.file.scale()
        peaks = imageprep.find_peaks(np.flipud(image))[:n]
        self.file.clusters = peaks
        xcen = peaks['xcen'] * scale
        ycen = peaks['ycen'] * scale
        coords = np.dstack((xcen, ycen)).reshape((-1, 2))
        self.widget.cluster_find(coords)

    def load_clusters(self):
        """Retrieve locations of star clusters from file."""
        fname = QFileDialog.getOpenFileName(
            self, '3D Model Creator - Load Clusters', '',
            'Text files (*.txt *.dat)')

        if fname.isEmpty():
            return

        scale = self.file.scale()
        peaks = ascii.read(
            str(fname), data_start=1,
            names=['id', 'xcen', 'ycen', 'area', 'radius_equiv', 'flux'])
        self.file.clusters = peaks
        xcen = peaks['xcen'] * scale
        ycen = peaks['ycen'] * scale
        coords = np.dstack((xcen, ycen)).reshape((-1, 2))
        self.widget.cluster_find(coords)

    def save_clusters(self):
        """Remove clusters identified by the user as
        other objects (foreground stars, center of galaxy,
        etc.).

        .. note::

            Foreground stars also need to be smoothed out
            from image.

        """
        if self.file.clusters is not None:
            toremove = self.widget.save_clusters()
            if len(toremove) > 0:
                self.file.clusters.remove_rows(toremove)

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
        #    makeqimage(data, self.transformation, self.widget.size))
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
        #self.file.clusters = t

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

        self.parent = parent
        self.view = QGraphicsView(self)
        self.view.setAlignment(Qt.AlignCenter)

        layout = QGridLayout(self)
        layout.addWidget(self.view, 0, 0)
        self.setLayout(layout)
        self.size = QSize(parent.width(), parent.height())
        self.view.setMinimumSize(self.size)
        self.resize(self.size)

        self.main_scene = StarScene(self, self.view.width(), self.view.height())
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
        self.main_scene.addImg(self.parent.file.image)
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

    def region_drawer(self, name):
        """Sets the scene to an interactive drawing scene,
        allowing the user to draw regions.

        Parameters
        ----------
        name : str

        """
        draw_scene = RegionStarScene(self, self.parent.file.image, name)
        self.update_scene(draw_scene)

    def region_loader(self, reg):
        """Sets the scene to display region loaded from file.

        Parameters
        ----------
        reg : `~astro3d.gui.astroObjects.Region` or list

        """
        draw_scene = RegionFileScene(self, self.parent.file.image, reg)
        self.update_scene(draw_scene)

    def save_region(self):
        """Sets the scene to the non-interactive main scene.
        Passes region information to `AstroGUI` to save.

        Returns
        -------
        name : str

        region : QPolygonF or list

        """
        name, region = self.current_scene.getRegion()
        self.update_scene(self.main_scene)
        return name, region

    def clear_region(self):
        """Clears the currently displayed region."""
        self.current_scene.clear()

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

    def cluster_find(self, points):
        """Highlights the locations of clusters given by points
        to allow the user to remove 'invalid' clusters, such as
        foreground stars, etc. using the interactive
        `~astro3d.gui.star_scenes.ClusterStarScene`.

        Parameters
        ----------
        points : ndarray

        """
        cluster_scene = ClusterStarScene(self, self.parent.file.image, points)
        self.update_scene(cluster_scene)

    def save_clusters(self):
        """Sets the scene to the non-interactive main scene.
        Passes a list of indices to remove from the
        `astropy.table.Table` that contains the cluster information.

        Returns
        -------
        toremove : list

        """
        toremove = self.current_scene.toremove
        self.update_scene(self.main_scene)
        return toremove


def makeqimage(nparray, transformation, size):
    """Performs various transformations (linear, log, sqrt, etc.)
    on the image. Clips and scales pixel values between 0 and 255.
    Scales and inverts the image. All transformations are
    non-destructive (performed on a copy of the input array).

    Parameters
    ----------
    nparray : ndarray

    transformation : func

    size : QSize

    Returns
    -------
    qimage : QImage

    """
    npimage = nparray.copy()
    npimage[npimage < 0] = 0
    npimage = q2a._normalize255(transformation(npimage), True)
    qimage = q2a.gray2qimage(npimage, (0, 255))
    qimage = qimage.scaled(size, Qt.KeepAspectRatio)
    qimage = qimage.mirrored(False, True)
    return qimage


def main(argv):
    """Execute the GUI."""
    app = QApplication(argv)
    window = AstroGUI(argv[1:])
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)
