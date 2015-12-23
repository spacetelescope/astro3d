"""Region overlay handling"""
from __future__ import absolute_import, print_function

from collections import defaultdict

import numpy as np

from ginga import colors
from ginga.AstroImage import AstroImage
from ginga.RGBImage import RGBImage
from ginga.canvas.CanvasObject import get_canvas_types

from ...external.qt import (QtGui, QtCore)
from ...core.region_mask import RegionMask
from ...util.logger import make_logger
from .items import *

from .util import EventDeferred


__all__ = [
    'OverlayView'
]

COLORS = defaultdict(
    lambda: 'red',
    {
        'bulge': 'blue',
        'gas': 'green',
        'remove_star': 'red',
        'spiral': 'orange',
    }
)


class BaseOverlay(object):
    """Base class for Overlays

    Parameters
    ----------
    parent: `Overlay`
    """
    def __init__(self, parent=None):
        self._dc = get_canvas_types()
        self.canvas = self._dc.Canvas()
        if parent is not None:
            self.parent = parent

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        parent.canvas.add(self.canvas)
        self._parent = parent

    def delete_all_objects(self):
        """Remove the immediate children"""
        self.canvas.delete_all_objects()
        self.children = []


class Overlay(BaseOverlay):
    """Overlays

    Overlays on which regions are shown.

    Parameters
    ----------
    parent: `Overlay`
        The parent overlay.

    color: str
        The color that shapes have for this overlay.

    Attributes
    ----------
    parent
        The parent overlay

    color
        The default color of shapes on this overlay.

    view
        The shape id on the ginga canvas.
    """

    def __init__(self, parent=None, color='red'):
        super(Overlay, self).__init__()
        self.canvas = self._dc.DrawingCanvas()
        self.parent = parent
        self.color = color
        self.children = []

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if parent is not None:
            self.view = parent.canvas.add(self.canvas)
        self._parent = parent

    def add_tree(self, layer):
        """Add layer's children to overlay"""
        for child in layer:
            view = self.add(child)
            try:
                view.add_tree(child)
            except AttributeError:
                """Leaf node or not available. Stop recursion"""
                pass

    def add(self, layer):
        """Add a layer

        Parameters
        ----------
        layer: LayerItem
            The layer to add.

        Returns
        -------
        None if the layer cannot be added. Usually due to non-availability.
        Otherwise, will be one of:
            Overlay: For non-leaf layers
            ginga shape: For leaf layers.
        """
        print('Overlay.add: layer="{}"'.format(layer))

        if not layer.is_available:
            return None
        if isinstance(layer, (RegionItem,)):
            view = self.add_region(layer)
        elif isinstance(layer, (Regions, Textures, Clusters, Stars, TypeItem)):
            view = self.add_overlay(layer)
        return view

    def add_child(self, overlay):
        """Add a child overlay"""
        self.children.append(overlay)

    def add_region(self, region_item):
        """Add a region to an overlay

        Parameters
        ----------
        region_item: LayerItem
            A region.

        Returns
        -------
        The ginga object identifier, or None if the item
        is not available.
        """
        if not region_item.is_available:
            return None
        if region_item.view is None:
            region = region_item.value
            if isinstance(region, RegionMask):
                mask = AstroImage(data_np=region.mask)
                maskrgb = masktorgb(mask, color=self.color, opacity=0.3)
                maskrgb_obj = self._dc.Image(0, 0, maskrgb)
                region_item.view = maskrgb_obj
            else:
                raise NotImplementedError(
                    'Cannot create view of region "{}"'.format(region_item)
                )
        self.canvas.add(region_item.view)
        return region_item.view

    def add_overlay(self, layer_item):
        """Add another overlay

        Parameters
        ----------
        layer_item: LayerItem
            A higher level LayerItem which has children.

        Returns
        -------
        The new overlay.
        """
        if not layer_item.is_available:
            return None
        if layer_item.view is not None:
            overlay = layer_item.view
            overlay.delete_all_objects()
            overlay.parent = self
        else:
            overlay = Overlay(parent=self)
            layer_item.view = overlay
            overlay.color = COLORS[layer_item.text()]
        self.add_child(overlay)
        return overlay


class RegionsOverlay(BaseOverlay):
    """Top level Regions overlay

    Individual region types are sub-overlays
    to this overlay.
    """

    def __init__(self, parent=None):
        super(RegionsOverlay, self).__init__(parent=parent)


class OverlayView(QtCore.QObject):
    """Present an overlay view to a QStandardItemModel

    Parameters
    ----------
    parent: `ginga.CanvasObject`
        The ginga canvas on which the view will render

    model: `astro3d.gui.Model`
        The Model which will be viewed.
    """

    def __init__(self, parent=None, model=None, logger=None):
        super(OverlayView, self).__init__()
        if logger is None:
            logger = make_logger('OverlayView')
        self.logger = logger

        self._defer_paint = QtCore.QTimer()
        self._defer_paint.setSingleShot(True)
        self._defer_paint.timeout.connect(self._paint)

        self.model = model
        self.parent = parent

    @property
    def parent(self):
        try:
            return self._root.canvas
        except AttributeError:
            return None

    @parent.setter
    def parent(self, parent):
        self._root = Overlay(parent=parent)
        self.paint()

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            return None

    @model.setter
    def model(self, model):
        try:
            logger = model.logger
        except AttributeError:
            """Model has no logger, ignore"""
            pass
        else:
            if logger is not None:
                self.logger = logger

        self._disconnect()
        self._model = model
        self._connect()
        self.paint()

    @EventDeferred
    def paint(self, *args, **kwargs):
        self.logger.debug(
            'Called: args="{}" kwargs="{}".'.format(args, kwargs)
        )
        self._paint(*args, **kwargs)

    def _paint(self, *args, **kwargs):
        """Show all overlays"""
        self.logger.debug(
            'Called: args="{}" kwargs="{}".'.format(args, kwargs)
        )
        try:
            self.logger.debug('sender="{}"'.format(self.sender()))
        except AttributeError:
            """No sender, ignore"""
            pass

        if self.model is None or self.parent is None:
            return
        root = self._root
        root.delete_all_objects()
        root.add_tree(self.model)
        root.canvas.redraw(whence=2)

    def _connect(self):
        """Connect model signals"""
        try:
            self.model.itemChanged.connect(self.paint)
            self.model.columnsInserted.connect(self.paint)
            self.model.columnsMoved.connect(self.paint)
            self.model.columnsRemoved.connect(self.paint)
            self.model.rowsInserted.connect(self.paint)
            self.model.rowsMoved.connect(self.paint)
            self.model.rowsRemoved.connect(self.paint)
        except AttributeError:
            """Model is probably not defined. Ignore"""
            pass

    def _disconnect(self):
        """Disconnect signals"""
        try:
            self.model.itemChanged.disconnect(self.paint)
            self.model.columnsInserted.disconnect(self.paint)
            self.model.columnsMoved.disconnect(self.paint)
            self.model.columnsRemoved.disconnect(self.paint)
            self.model.rowsInserted.disconnect(self.paint)
            self.model.rowsMoved.disconnect(self.paint)
            self.model.rowsRemoved.disconnect(self.paint)
        except AttributeError:
            """Model is probably not defined. Ignore"""
            pass


# Utilities
def masktorgb(mask, color='blue', opacity=0.3):
    wd, ht = mask.get_size()
    r, g, b = colors.lookup_color(color)
    rgbarr = np.zeros((ht, wd, 4), dtype=np.uint8)
    rgbobj = RGBImage(data_np=rgbarr)

    rc = rgbobj.get_slice('R')
    gc = rgbobj.get_slice('G')
    bc = rgbobj.get_slice('B')
    ac = rgbobj.get_slice('A')

    data = mask.get_data()
    ac[:] = 0
    idx = data > 0
    rc[idx] = int(r * 255)
    gc[idx] = int(g * 255)
    bc[idx] = int(b * 255)
    ac[idx] = int(opacity * 255)

    return rgbobj
