"""Region overlay handling"""
from __future__ import absolute_import, print_function

from collections import defaultdict
import numpy as np

from ginga import colors
from ginga.AstroImage import AstroImage
from ginga.RGBImage import RGBImage
from ginga.canvas.CanvasObject import get_canvas_types

from ...core.region_mask import RegionMask

COLORS = defaultdict(
    lambda: 'blue',
    {
        'bulge': 'blue',
        'gas': 'green',
        'remove_star': 'red',
        'spiral': 'orange',
    }
)


class Overlay(object):
    """Overlays

    Parameters
    ----------
    parent: ginga Canvas
        The canvas we'll work on.
    """

    def __init__(self, parent=None, color='red'):
        self.dc = get_canvas_types()
        self.canvas = self.dc.DrawingCanvas()
        if parent is not None:
            self.parent = parent
        self.color = color

        self._known_shapes = {}

    @property
    def parent(self):
        return self.canvas.get_surface()

    @parent.setter
    def parent(self, parent):
        self.canvas.set_surface(parent)
        parent.add(self.canvas)

    def add(self, shape):
        """Add a ginga shape to the canvas"""
        self.canvas.add(shape)

    def add_region(self, region):
        print('Overlay.add_region: region="{}"'.format(region))
        if isinstance(region, RegionMask):
            try:
                maskrgb_obj = self._known_shapes[region]
            except KeyError:
                mask = AstroImage(data_np=region.mask)
                maskrgb = masktorgb(mask, color=self.color, opacity=0.3)
                maskrgb_obj = self.dc.Image(0, 0, maskrgb)
                self._known_shapes[region] = maskrgb_obj
            self.canvas.add(maskrgb_obj)


class RegionsOverlay(Overlay):
    """Top level Regions overlay

    Individual region types are sub-overlays
    to this overlay.
    """

    def __init__(self, parent=None):
        super(RegionsOverlay, self).__init__(parent=parent)
        self.type_overlays = {}

    def delete_all_objects(self):
        self.canvas.delete_all_objects()
        self.type_overlays = {}

    def add_region(self, region):
        try:
            overlay = self.type_overlays[region.mask_type]
        except KeyError:
            overlay = Overlay(
                parent=self.canvas,
                color=COLORS[region.mask_type]
            )
            self.type_overlays[region.mask_type] = overlay
        overlay.add_region(region)


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
