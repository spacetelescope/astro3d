"""Region overlay handling"""
from functools import partial

import numpy as np

from ginga import colors
from ginga.RGBImage import RGBImage
from ginga.canvas.CanvasObject import get_canvas_types
from qtpy import QtCore

from ...core.region_mask import RegionMask
from ...util.logger import make_null_logger
from .shape_editor import image_shape_to_regionmask
from .items import *

from .util import EventDeferred

# Configure logging
logger = make_null_logger(__name__)

__all__ = ['OverlayView']


class BaseOverlay():
    """Base class for Overlays

    Parameters
    ----------
    parent: `Overlay`
    """

    def __init__(self, parent=None):
        self.canvas = None
        if parent is not None:
            self.parent = parent

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        self.canvas = parent.canvas
        self._dc = parent._dc

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
        self.canvas = None
        self.parent = parent
        self.color = color
        self.children = []

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        if parent is not None:
            self.canvas = parent.canvas
            self._dc = parent._dc

    def add_tree(self, layer):
        """Add layer's children to overlay"""
        for child in layer.children():
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
        logger.debug('Called: layer="{}"'.format(layer))

        view = None
        if layer.is_available:
            if isinstance(layer, (RegionItem,)):
                view = self.add_region(layer)
            elif isinstance(
                    layer,
                    (
                        Catalogs,
                        CatalogTypeItem,
                        Clusters,
                        Regions,
                        Stars,
                        Textures,
                        TypeItem
                    )
            ):
                view = self.add_overlay(layer)
            elif isinstance(layer, (
                    CatalogItem,
                    ClusterItem,
                    StarsItem
            )):
                view = self.add_table(layer)

        logger.debug('Returned view="{}"'.format(view))
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
                mask = RGBImage(data_np=region.mask)
                maskrgb = masktorgb(
                    mask,
                    color=self.draw_params['color'],
                    opacity=self.draw_params['fillalpha'])
                maskrgb_obj = self._dc.Image(0, 0, maskrgb)
                maskrgb_obj.item = region_item
                region_item.view = maskrgb_obj
                region_item.view.type_draw_params = self.draw_params

                # Redefine the region value so that
                # it will dynamically update during
                # editing.
                region_item.value = partial(
                    image_shape_to_regionmask,
                    shape=maskrgb_obj,
                    mask_type=region.mask_type
                )
            else:
                raise NotImplementedError(
                    'Cannot create view of region "{}"'.format(region_item)
                )
        self.canvas.add(region_item.view, tag=region_item.text())
        return region_item.view

    def add_table(self, layer):
        logger.debug('Called.')
        if not layer.is_available:
            return None
        table = layer.value
        logger.debug('len(table)="{}"'.format(len(table)))
        container = self._dc.CompoundObject()
        container.initialize(None, self.canvas.viewer, logger)
        for row in table:
            point = self._dc.Point(
                x=row['xcentroid'],
                y=row['ycentroid'],
                **layer.draw_params
            )
            point.pickable = True
            point.editable = False
            point.idx = row.index
            point.item = layer
            point.add_callback('pick-key', layer.key_callback)
            for cb_name in ['pick-down',
                            'pick-up',
                            'pick-move',
                            'pick-hover',
            ]:
                point.add_callback(cb_name, cb_debug)
            container.add_object(point)
        layer.view = container
        self.canvas.add(layer.view, tag=layer.text())
        return layer.view

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
            overlay.parent = self
        else:
            overlay = Overlay(parent=self)
            layer_item.view = overlay
            try:
                overlay.draw_params = layer_item.draw_params
            except AttributeError:
                """Layer does not have any, ignore."""
                pass
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
    parent: `ginga.ImageViewCanvas`
        The ginga canvas on which the view will render

    model: `astro3d.gui.Model`
        The Model which will be viewed.
    """

    def __init__(self, parent=None, model=None):
        super(OverlayView, self).__init__()

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
        self._dc = get_canvas_types()
        canvas = self._dc.DrawingCanvas()
        self.canvas = canvas
        p_canvas = parent.get_canvas()
        p_canvas.add(self.canvas)
        self._root = Overlay(self)
        self.paint()

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            return None

    @model.setter
    def model(self, model):
        self._disconnect()
        self._model = model
        self._connect()
        self.paint()

    @EventDeferred
    def paint(self, *args, **kwargs):
        logger.debug(
            'Called: args="{}" kwargs="{}".'.format(args, kwargs)
        )
        self._paint(*args, **kwargs)

    def _paint(self, *args, **kwargs):
        """Show all overlays"""
        logger.debug(
            'Called: args="{}" kwargs="{}".'.format(args, kwargs)
        )
        try:
            logger.debug('sender="{}"'.format(self.sender()))
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
def masktorgb(mask, color='red', opacity=0.3):
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
    rc[:] = int(r * 255)
    gc[:] = int(g * 255)
    bc[:] = int(b * 255)
    ac[idx] = int(opacity * 255)

    return rgbobj


def cb_debug(*args, **kwargs):
    print('overlay.cb_debug: args="{}" kwargs="{}"'.format(args, kwargs))
