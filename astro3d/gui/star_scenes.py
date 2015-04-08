"""This module contains several subclasses of ``QGraphicsScene``.
These are instantiated by `~astro3d.gui.astroVisual.MainPanel` and
are used to display the image and any regions.
Furthermore, `RegionStarScene` and `ClusterStarScene` allow
user input upon mouse click.

"""
from __future__ import division, print_function

# STDLIB
import warnings
from collections import defaultdict
from copy import deepcopy

# Anaconda
import numpy as np
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# THIRD-PARTY
from qimage2ndarray import array2qimage

# Local
from ..utils import imutils
from ..utils.imageprep import combine_masks
from ..utils.imutils import calc_insertion_pos


def mask2pixmap(mask, alpha, i_layer, size=None):
    """Convert mask to semi-transparent pixmap item for display.

    Parameters
    ----------
    mask : array_like
        Boolean mask to convert.

    alpha : int
        Alpha value for transparency.

    i_layer : {0, 1, 2}
        Index for RGB layer that sets the color of the display.

    Returns
    -------
    item : QGraphicsPixmapItem

    """
    im = np.zeros((mask.shape[0], mask.shape[1], 4))  # RGBA
    im[:, :, i_layer][mask] = 255
    im[:, :, 3][mask] = alpha
    pixmap = QPixmap()
    pixmap = pixmap.fromImage(array2qimage(im))

    if size is not None:
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio)

    return QGraphicsPixmapItem(pixmap)


class PreviewScene(QGraphicsScene):
    """Display preview image.

    Intensity will be monochrome image.
    Each texture is displayed as semi-transparent overlay with
    a pre-defined color.

    Parameters
    ----------
    parent : `~astro3d.gui.astroVisual.PreviewWindow`
        The instantiating class.

    width, height : int
        The size of the `~astro3d.gui.astroVisual.PreviewWindow`,
        which allows `PreviewScene` to scale the image appropriately.

    """
    _MASK_ALPHA = 20
    _ELLIPSE_SZ = 10
    _ELLIPSE_RAD = _ELLIPSE_SZ // 2
    _CLUSTER_COLOR = QColor(200, 50, 50)
    _STAR_COLOR = QColor(50, 50, 200)

    def __init__(self, parent, width, height):
        super(PreviewScene, self).__init__(parent)
        self.width = width
        self.height = height
        self.size = QSize(self.width, self.height)

        # Set later
        self.model3d = None
        self.pixmap = None

    def set_model(self, model3d):
        """Set 3D model object."""
        self.model3d = model3d

    def clear(self):
        """Clear the display."""
        for i in self.items():
            self.removeItem(i)

    def _add_image(self):
        """Scale and display intensity image."""
        self.pixmap = QPixmap().fromImage(imutils.makeqimage(
            self.model3d.preview_intensity, None, self.size))
        self.addItem(QGraphicsPixmapItem(self.pixmap))

    def _add_point(self, xcen, ycen, color):
        """Add a point."""
        x = xcen - self._ELLIPSE_RAD
        y = ycen - self._ELLIPSE_RAD
        item = QGraphicsEllipseItem(x, y, self._ELLIPSE_SZ, self._ELLIPSE_SZ)
        item.setPen(QPen(color))
        self.addItem(item)

    def draw(self):
        """Render the preview."""
        log.info('Generating preview...')
        self.clear()
        self._add_image()

        # Small dots (red)
        sdmask = self.model3d.get_preview_mask(self.model3d.small_dots_key)
        if np.any(sdmask):
            item = mask2pixmap(sdmask, self._MASK_ALPHA, 0, size=self.size)
            self.addItem(item)

        # Dots (green)
        dmask = self.model3d.get_preview_mask(self.model3d.dots_key)
        if np.any(dmask):
            item = mask2pixmap(dmask, self._MASK_ALPHA, 1, size=self.size)
            self.addItem(item)

        # Lines (blue)
        lmask = self.model3d.get_preview_mask(self.model3d.lines_key)
        if np.any(lmask):
            item = mask2pixmap(lmask, self._MASK_ALPHA, 2, size=self.size)
            self.addItem(item)

        scale = self.pixmap.height() / self.model3d.preview_intensity.shape[0]

        # Star clusters
        clusters = self.model3d.get_final_clusters()
        if len(clusters) > 0:
            xcen = clusters['xcen'] * scale
            ycen = clusters['ycen'] * scale
            for x, y in zip(xcen, ycen):
                self._add_point(x, y, self._CLUSTER_COLOR)

        # Stars
        stars = self.model3d.get_final_stars()
        if len(stars) > 0:
            xcen = stars['xcen'] * scale
            ycen = stars['ycen'] * scale
            for x, y in zip(xcen, ycen):
                self._add_point(x, y, self._STAR_COLOR)


class StarScene(QGraphicsScene):
    """This is a non-interactive subclass of ``QGraphicsScene``.
    It will display the image and any regions that have been added.

    Parameters
    ----------
    parent : `~astro3d.gui.astroVisual.MainPanel`
        The instantiating class.

    width, height : int
        The size of the `~astro3d.gui.astroVisual.MainPanel`,
        which allows `StarScene` to scale the image appropriately.

    Attributes
    ----------
    size : QSize
        Size of the `~astro3d.gui.astroVisual.MainPanel`.

    pixmap : QGraphicsPixmapItem
        Image the regions belong to.

    regions : dict
        Contains region type and a list of `~astro3d.gui.astroObjects.Region`, which provides `StarScene` with a pointer to each region, allowing them to be removed if necessary. ``Region.name`` must contain region type.

    clusters, stars : list or Table
        Contains star clusters and stars to display.

    """
    _MASK_ALPHA = 40
    _SELECTED_ALPHA = 60
    _ELLIPSE_SZ = 10
    _ELLIPSE_RAD = _ELLIPSE_SZ // 2
    _CLUSTER_COLOR = QColor(200, 50, 50)
    _STAR_COLOR = QColor(50, 50, 200)

    def __init__(self, parent, width, height):
        super(StarScene, self).__init__(parent)
        self.size = QSize(width, height)
        self.pixmap = None
        self.regions = defaultdict(list)
        self.clusters = []
        self.stars = []

    def addImg(self, pixmap):
        """Scales the input pixmap to appropriate size for the
        `~astro3d.gui.astroVisual.MainPanel`, then adds it to
        the display. Adds all regions on top of image.

        .. note::

            Returns the scaled pixmap to save so that scaling will be
            unnecessary in the future.

        Parameters
        ----------
        pixmap : QPixmap

        Returns
        -------
        scaledPixmap : QPixmap

        """
        scaledPixmap = pixmap.scaled(self.size, Qt.KeepAspectRatio)
        self.pixmap = QGraphicsPixmapItem(scaledPixmap)
        self.draw()
        self._pixmap = scaledPixmap
        return scaledPixmap

    def addReg(self, region):
        """Adds a given region to the display.

        .. note::

            The recursive behavior is for ``MergedRegion``,
            which is not currently supported.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region`

        """
        if isinstance(region, (list, tuple)):
            return map(self.addReg, region)
        else:
            self.regions[region.name].append(region)
            self.draw()

    def delReg(self, region):
        """Remove a given region from the display.

        .. note::

            The recursive behavior is for ``MergedRegion``,
            which is not currently supported.

        Parameters
        ----------
        region : `~astro3d.gui.astroObjects.Region`

        """
        if isinstance(region, (list, tuple)):
            map(self.delReg, region)
        else:
            self.regions[region.name].remove(region)
            self.draw()

    def set_clusters(self, clusters, orig_height):
        """Save selected clusters for display."""
        scale = self._pixmap.height() / orig_height
        self.clusters = deepcopy(clusters)
        self.clusters['xcen'] *= scale
        self.clusters['ycen'] *= scale
        self.draw()

    def set_stars(self, stars, orig_height):
        """Save selected stars for display."""
        scale = self._pixmap.height() / orig_height
        self.stars = deepcopy(stars)
        self.stars['xcen'] *= scale
        self.stars['ycen'] *= scale
        self.draw()

    def clear(self):
        """Removes all items from the display without destroying
        instance variables.

        """
        for i in self.items():
            self.removeItem(i)

    def _add_point(self, xcen, ycen, color):
        """Add a point."""
        x = xcen - self._ELLIPSE_RAD
        y = ycen - self._ELLIPSE_RAD
        item = QGraphicsEllipseItem(x, y, self._ELLIPSE_SZ, self._ELLIPSE_SZ)
        item.setPen(QPen(color))
        self.addItem(item)

    def draw(self, selected=[]):
        """Draw scene. Highlight selected region(s)."""
        if not isinstance(selected, list):
            selected = [selected]

        self.clear()

        if self.pixmap is not None:
            self.addItem(self.pixmap)

        reg_masks = []
        sel_masks = []
        for reglist in self.regions.itervalues():
            for reg in reglist:
                if reg in selected:
                    sel_masks.append(reg.region)
                else:
                    reg_masks.append(reg.region)

        reg_masks = combine_masks(reg_masks)
        if len(reg_masks) > 0:
            item = mask2pixmap(reg_masks, self._MASK_ALPHA, 1)  # Green
            self.addItem(item)

        # Highlight selected regions last
        sel_masks = combine_masks(sel_masks)
        if len(sel_masks) > 0:
            item = mask2pixmap(sel_masks, self._SELECTED_ALPHA, 0)  # Red
            self.addItem(item)

        # Star clusters
        if len(self.clusters) > 0:
            for x, y in zip(self.clusters['xcen'], self.clusters['ycen']):
                self._add_point(x, y, self._CLUSTER_COLOR)

        # Stars
        if len(self.stars) > 0:
            for x, y in zip(self.stars['xcen'], self.stars['ycen']):
                self._add_point(x, y, self._STAR_COLOR)


class _RegionStarScene(QGraphicsScene):
    """This is an interactive subclass of ``QGraphicsScene``.
    Every time the user clicks on the image, it generates a point
    and adds it to a ``QPolygon``, allowing it to display that
    polygon as a region.

    .. note:: Not used.

    Parameters
    ----------
    parent : `~astro3d.gui.astroVisual.MainPanel`
        The instantiating class.

    pixmap : QPixmap
        The scaled QPixmap generated by :meth:`StarScene.addImg`.
        It is added so that `RegionStarScene` can display the image.

    name : str
        Name (type) of the region to be drawn.

    Attributes
    ----------
    name : str
        Same as input.

    description : str
        Description of the region (informational only).

    item : QGraphicsPixmapItem
        Created from ``pixmap``.

    graphicspoints : list of `RegionStarSceneItem`
        Displayed circles representing ``points``.

    overwrite : tuple or `None`
        ``(key, index)`` to identify an existing region in GUI to replace. This is set directly by GUI.

    """
    _REGION_COLOR = QColor(0, 100, 200)
    _MIN_POINTS = 3

    def __init__(self, parent, pixmap, name):
        super(RegionStarScene, self).__init__(parent)
        self.name = name
        self.description = name
        self.item = QGraphicsPixmapItem(pixmap)
        self.graphicspoints = []
        self.overwrite = None
        self._i_to_move = None
        self.draw()

    @classmethod
    def from_region(cls, parent, pixmap, region):
        """Generate scene from existing region."""
        newcls = cls(parent, pixmap, region.name)
        newcls.description = region.description
        newcls.graphicspoints = [
            RegionStarSceneItem(p) for p in region.points()]
        newcls.draw()
        return newcls

    def mousePressEvent(self, event):
        """This method is called whenever the user clicks on
        `RegionStarScene`.

        It adds a new point clicked to ``graphicspoints``.
        If the point already exists, right click will remove it.
        To move an existing point, select it with left click and
        then CNTRL+click the new position.
        If 3 or more points are present, it generates `shape` and
        draws a polygon.

        Parameters
        ----------
        event : QEvent

        """
        p = event.scenePos()
        flag, i = self.has_point(p)

        if flag:
            if event.button() == Qt.RightButton:  # Remove point
                self.graphicspoints.pop(i)
                self.reset_point_selection()
            else:  # Mark point to be moved
                self._i_to_move = i
                self.graphicspoints[self._i_to_move].do_selection()
        else:
            if event.button() == Qt.LeftButton:
                gp = RegionStarSceneItem(p)
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:  # Move point
                    if self._i_to_move is not None:
                        self.graphicspoints[self._i_to_move] = gp
                else:  # Add point
                    self.graphicspoints.append(gp)
            self.reset_point_selection()

        self.draw()

    def reset_point_selection(self):
        """Undo any highlighted point."""
        if self._i_to_move is not None:
            self.graphicspoints[self._i_to_move].undo_selection()
            self._i_to_move = None

    def has_point(self, p):
        """Check if given point already exists.

        Parameters
        ----------
        p : QPointF

        Returns
        -------
        flag : bool
            `True` if point exists, else `False`.

        idx : int or `None`
            Corresponding index in ``self.points`` and
            ``self.graphicspoints``, if exists.

        """
        flag = False
        idx = None

        for i, gp in enumerate(self.graphicspoints):
            if gp.contains(p):
                flag = True
                idx = i
                break

        return flag, idx

    @property
    def shape(self):
        """Polygon shape."""
        if len(self.graphicspoints) < self._MIN_POINTS:
            s = None
        else:
            s = QPolygonF([gp.pos for gp in self.graphicspoints])
        return s

    def draw(self):
        """Draw points and polygon."""
        for i in self.items():
            self.removeItem(i)

        self.addItem(self.item)

        for gp in self.graphicspoints:
            self.addItem(gp)

        shape = self.shape
        if shape is not None:
            display_shape = QGraphicsPolygonItem(shape)
            display_shape.setPen(QPen(self._REGION_COLOR))
            self.addItem(display_shape)

    def getRegion(self):
        """Return information of region to save.

        Returns
        -------
        name : str
            Name (key) of the region.

        shape : QPolygonF
            Shape of the region.

        description : str
            Description of the region.

        """
        return self.name, self.shape, self.description

    def clear(self):
        """Removes all items from the display except the image.
        Resets all instance variables except for ``item``.

        """
        for i in self.items():
            self.removeItem(i)
        self.addItem(self.item)
        self.graphicspoints = []


class _RegionStarSceneItem(QGraphicsEllipseItem):
    """Class to handle data points in `RegionStarScene`.

    .. note:: Not used.

    Parameters
    ----------
    pos : QPointF
        Initial position of the data point.

    """
    _ELLIPSE_SZ = 10
    _ELLIPSE_RAD = _ELLIPSE_SZ // 2
    _ELLIPSE_COLOR = QColor(0, 255, 0)
    _SELECTED_COLOR = QColor(255, 0, 0)

    def __init__(self, pos):
        super(RegionStarSceneItem, self).__init__(
            pos.x() - self._ELLIPSE_RAD, pos.y() - self._ELLIPSE_RAD,
            self._ELLIPSE_SZ, self._ELLIPSE_SZ)
        self.pos = pos
        self.setPen(QPen(self._ELLIPSE_COLOR))
        self.setAcceptHoverEvents(True)
        self._default_brush = self.brush()
        self._highlight_brush = QBrush(self._ELLIPSE_COLOR)

    def do_selection(self):
        """Mark as selected."""
        self.setPen(QPen(self._SELECTED_COLOR))

    def undo_selection(self):
        """Undo :meth:`do_selection`."""
        self.setPen(QPen(self._ELLIPSE_COLOR))

    def hoverEnterEvent(self, event):
        """Highlight point on mouse over."""
        self.setBrush(self._highlight_brush)

    def hoverLeaveEvent(self, event):
        """Un-highlight point when mouse leaves."""
        self.setBrush(self._default_brush)


class RegionFileScene(QGraphicsScene):
    """Like `RegionBrushScene` but multiple regions are
    pre-loaded from files.

    """
    _MASK_ALPHA = 50

    def __init__(self, parent, pixmap, regions):
        super(RegionFileScene, self).__init__(parent)

        if not isinstance(regions, list):
            raise ValueError('Only support multiple regions')

        self.item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.item)
        self.name = []
        self._mask = []
        self.description = []
        self.overwrite = None  # Not allowed here
        masklist = []

        for reg in regions:
            self.name.append(reg.name)
            self._mask.append(reg.region)
            self.description.append(reg.description)

        self.show_mask(self._mask)

    def show_mask(self, masklist):
        """Semi-transparent masks for display."""
        mask = combine_masks(masklist)
        item = mask2pixmap(mask, self._MASK_ALPHA, 1)  # Green
        self.addItem(item)

    def getRegion(self):
        """Return information of region(s) to save.

        Returns
        -------
        name :list of str
            Name (key) of the region.

        mask : list of array_like
            Boolean mask of the region.

        description : list of str
            Description of the region.

        """
        return self.name, self._mask, self.description

    def clear(self):
        """Removes all items from the display except the image."""
        for i in self.items():
            self.removeItem(i)
        self.addItem(self.item)


# This replaces RegionStarScene
class RegionBrushScene(QGraphicsScene):
    """An interactive  subclass of ``QGraphicsScene``.
    Instead of defining each data point to build a polygon,
    user uses brush strokes to draw the region.

    """
    _REGION_COLOR = QColor(0, 255, 0)
    _BRUSH_COLOR = QColor(255, 0, 0)
    _BRUSH_BUFFPIX = 2  # 2 pix buffer on each end for gradient calculations
    _BRUSH_SIZE_MIN = 5
    _BRUSH_SIZE_MAX = 100
    _BRUSH_SIZE_STEP = 5
    _MASK_ALPHA = 50

    def __init__(self, parent, pixmap, name, radius=15):
        super(RegionBrushScene, self).__init__(parent)
        self.parent = parent
        self.name = name
        self.description = name
        self.item = QGraphicsPixmapItem(pixmap)
        self.overwrite = None

        self.radius = radius
        self._height = pixmap.height()
        self._width = pixmap.width()
        self._mask = None
        self._mode = None
        self._oldx = -1
        self._oldy = -1
        self._brushgraphics = None

        self.parent.parent.statusBar().showMessage(
            'Brush radius is {0} pixels'.format(self.radius))
        self.draw()

    @classmethod
    def from_region(cls, parent, pixmap, region):
        """Generate scene from existing region."""
        newcls = cls(parent, pixmap, region.name)
        newcls.description = region.description
        newcls._mask = region.region
        newcls.draw()
        return newcls

    def mousePressEvent(self, event):
        """First click sets up initial brush.
        Subsequent clicks edit the region.

        """
        pos = event.scenePos()
        self._oldx = int(pos.x())
        self._oldy = int(pos.y())

        if self._mask is None:
            self._mask = np.zeros((self._height, self._width), dtype=np.bool)
            ix1, ix2, iy1, iy2, mx1, mx2, my1, my2 = calc_insertion_pos(
                self._mask, self.brush, self._oldx - self.radius,
                self._oldy - self.radius)
            self._mask[iy1:iy2, ix1:ix2] = self.brush[my1:my2, mx1:mx2]

        elif self._mask[self._oldy, self._oldx]:
            self._mode = 'inside'

        else:
            self._mode = 'outside'

    def mouseMoveEvent(self, event):
        """Edit the region with brush.

        Construct a mask indicating coverage of a moving circle
        with center starting at old and finishing at new positions.
        The size of the mask is buffered by 2 extra pixels on all sides
        to facilitate the edge highlighting step.

        """
        if self._mode is None:
            return

        pos = event.scenePos()
        x = int(pos.x())
        y = int(pos.y())

        # Make grid arrays
        diam = 2 * self.radius + 1
        xsize = 2 * self._BRUSH_BUFFPIX + diam + np.abs(x - self._oldx)
        ysize = 2 * self._BRUSH_BUFFPIX + diam + np.abs(y - self._oldy)
        beg = -self.radius - self._BRUSH_BUFFPIX
        ygrid, xgrid = np.mgrid[beg:beg+ysize, beg:beg+xsize]
        p = np.zeros((ysize, xsize, 2), dtype=np.float32)
        p[:, :, 0] = ygrid
        p[:, :, 1] = xgrid

        # Ensure move positions are always positive
        xmin = min(x, self._oldx)
        ymin = min(y, self._oldy)

        movemask = imutils.in_rectangle(
            p, (y - ymin, x - xmin), (self._oldy - ymin, self._oldx - xmin),
            self.radius)

        for xx, yy in [(self._oldx, self._oldy), (x, y)]:
            yy1 = yy + self._BRUSH_BUFFPIX - ymin
            yy2 = yy1 + diam
            xx1 = xx + self._BRUSH_BUFFPIX - xmin
            xx2 = xx1 + diam
            movemask[yy1:yy2, xx1:xx2] = movemask[yy1:yy2, xx1:xx2] | self.brush

        # Update the image mask (inplace) by using the supplied position of the
        # drawing mask to either extend it (mode="inside"), or erode it
        # (mode="outside").

        ix1, ix2, iy1, iy2, mx1, mx2, my1, my2 = calc_insertion_pos(
            self._mask, movemask, x + self._BRUSH_BUFFPIX - self.radius,
            y + self._BRUSH_BUFFPIX - self.radius)
        icommon = self._mask[iy1:iy2, ix1:ix2]
        dmask = movemask[my1:my2, mx1:mx2]

        # Check for overlap; if not, nothing will change
        if np.any(icommon & dmask):
            if self._mode == 'inside':
                np.logical_or(icommon, dmask, out=icommon)
            elif self._mode == 'outside':
                icommon[np.logical_and(icommon, dmask)] = False
            else:
                warnings.warn(
                    'Invalid brush mode={0}; must be "inside" or '
                    '"outside"'.format(self._mode), AstropyUserWarning)

        self.set_brush(pos=(y, x))  # Display brush
        self.draw()

        self._oldx = x
        self._oldy = y

    def mouseReleaseEvent(self, event):
        """Done editing."""
        self._mode = None
        self.set_brush()  # Hide brush
        self.draw()

    def keyPressEvent(self, event):
        """Change brush size with Alt+Plus or Alt+Minus."""
        size_change_mode = False
        if (event.modifiers() & Qt.AltModifier):
            # Increase brush size
            if event.key() == Qt.Key_Plus:
                size_change_mode = True
                r = self.radius + self._BRUSH_SIZE_STEP
                if r <= self._BRUSH_SIZE_MAX:
                    self.radius = r
            # Decrease brush size
            elif event.key() == Qt.Key_Minus:
                size_change_mode = True
                r = self.radius - self._BRUSH_SIZE_STEP
                if r >= self._BRUSH_SIZE_MIN:
                    self.radius = r
        if size_change_mode:
            msg = 'Brush radius is {0} pixels'.format(self.radius)
            log.info(msg)
            self.parent.parent.statusBar().showMessage(msg)

    @property
    def graphicspoints(self):
        """Data points of the edges generated from mask."""
        gp = []
        if self._mask is not None:
            # Compute differentials to detect mask edges
            idx = np.where(self._mask[1:, 1:] ^ self._mask[:-1, :-1])
            for x, y in zip(idx[1], idx[0]):
                p = QGraphicsRectItem(x, y, 1, 1)
                p.setPen(self._REGION_COLOR)
                gp.append(p)
        return gp

    @property
    def maskgraphics(self):
        """Semi-transparent mask for display."""
        if self._mask is None:
            item = None
        else:
            item = mask2pixmap(self._mask, self._MASK_ALPHA, 1)  # Green
        return item

    @property
    def brush(self):
        """Circular mask that defines the brush."""
        diam = 2 * np.ceil(self.radius) + 1
        return imutils.circular_mask(
            (diam, diam), self.radius, self.radius, self.radius)

    def set_brush(self, pos=None):
        """Handle brush graphics.

        Parameters
        ----------
        pos : tuple or `None`
            If ``(y,x)`` is given, re-draw brush at that position.
            Else, just remove the existing brush graphics.

        """
        if pos is None:
            self._brushgraphics = None
        else:
            y, x = pos
            diam = 2 * self.radius + self._BRUSH_BUFFPIX
            self._brushgraphics = QGraphicsEllipseItem(
                x - self.radius + 2 * self._BRUSH_BUFFPIX,
                y - self.radius + 2 * self._BRUSH_BUFFPIX, diam, diam)
            self._brushgraphics.setPen(self._BRUSH_COLOR)

    def draw(self):
        """Draw polygon."""
        for i in self.items():
            self.removeItem(i)

        self.addItem(self.item)

        # DISABLED - Only draw mask edges
        #for gp in self.graphicspoints:
        #    self.addItem(gp)

        # Draw semi-transparent mask
        if self.maskgraphics is not None:
            self.addItem(self.maskgraphics)

        if self._brushgraphics is not None:
            self.addItem(self._brushgraphics)

    def getRegion(self):
        """Return information of region to save.

        Returns
        -------
        name : str
            Name (key) of the region.

        mask : array_like
            Boolean mask of the region.

        description : str
            Description of the region.

        """
        return self.name, self._mask, self.description

    def clear(self):
        """Removes all items from the display except the image.
        Resets all instance variables except for ``item``.

        """
        for i in self.items():
            self.removeItem(i)
        self.addItem(self.item)
        self._mask = None
        self._mode = None
        self._oldx = -1
        self._oldy = -1
        self._brushgraphics = None


class ClusterStarScene(QGraphicsScene):
    """An interactive subclass of ``QGraphicsScene``.
    Displays the given stars or star clusters, using circles
    to highlight their locations.

    The user may click on existing point to remove it from
    consideration, or on unmarked area to add it.

    Parameters
    ----------
    parent : `~astro3d.gui.astroVisual.MainPanel`
        The instantiating class.

    pixmap : QPixmap
        Background image pixmap.

    model3d : `~astro3d.utils.imageprep.ModelFor3D`
        Handles 3D model generation. Can have existing point sources already.

    key : {'clusters', 'stars'}
        Type of ``File.peaks`` table to process.

    Attributes
    ----------
    model3d, key
        Same as inputs.

    scale : float
        Scaling factor to convert between display and native coordinates.

    graphicspoints : QGraphicsEllipseItems
        Points displayed on screen.

    """
    _ELLIPSE_SZ = 10
    _ELLIPSE_RAD = _ELLIPSE_SZ // 2
    _ELLIPSE_COLOR = QColor(200, 50, 50)
    _FLUX_RAD = 5

    def __init__(self, parent, pixmap, model3d, key):
        super(ClusterStarScene, self).__init__(parent)
        self.model3d = model3d
        self.key = key
        self.scale = pixmap.height() / model3d.orig_img.shape[0]
        self.graphicspoints = []

        # Display static background image
        self.addItem(QGraphicsPixmapItem(pixmap))

        # Display existing point sources, if any
        data = model3d.peaks[key]
        if len(data) > 0:
            xcen = data['xcen'] * self.scale
            ycen = data['ycen'] * self.scale
            for x, y in zip(xcen, ycen):
                self.add_point(x, y)

    def add_point(self, xcen, ycen):
        """Add a point to graphics scene.

        Parameters
        ----------
        xcen, ycen : float
            Coordinate of the point in display scale.

        """
        x = xcen - self._ELLIPSE_RAD
        y = ycen - self._ELLIPSE_RAD
        item = QGraphicsEllipseItem(x, y, self._ELLIPSE_SZ, self._ELLIPSE_SZ)
        item.setPen(QPen(self._ELLIPSE_COLOR))
        self.graphicspoints.append(item)
        self.addItem(item)

    def mousePressEvent(self, event):
        """This method is called whenever the user clicks
        on the screen.

        If the click is on one of the displayed
        points, that point is removed from the screen
        and table. Otherwise, it is added.

        Parameters
        ----------
        event : QEvent

        """
        p = event.scenePos()
        is_removed = False

        for i, gp in enumerate(self.graphicspoints):
            if gp.contains(p):
                is_removed = True
                self.removeItem(gp)
                self.graphicspoints.remove(gp)
                self.model3d.peaks[self.key].remove_row(i)
                break

        if not is_removed:
            xdisp = p.x()
            ydisp = p.y()
            xcen = xdisp / self.scale
            ycen = ydisp / self.scale

            # Estimate flux at selected point
            ix1 = max(int(xcen - self._FLUX_RAD), 0)
            ix2 = min(int(xcen + self._FLUX_RAD), self.model3d.orig_img.shape[1])
            iy1 = max(int(ycen - self._FLUX_RAD), 0)
            iy2 = min(int(ycen + self._FLUX_RAD), self.model3d.orig_img.shape[0])
            flux = np.flipud(self.model3d.orig_img)[iy1:iy2, ix1:ix2].sum()

            self.add_point(xdisp, ydisp)
            self.model3d.peaks[self.key].add_row([xcen, ycen, flux])
            log.info('Added X={0:.1f} Y={1:.2f} FLUX={2:.3f}'.format(
                xcen, ycen, flux))
