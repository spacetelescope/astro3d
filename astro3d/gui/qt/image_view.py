from ginga.gw.Viewers import CanvasView
from ginga.canvas.CanvasObject import get_canvas_types

from ...util.logger import make_null_logger
from ...core.region_mask import RegionMask

# Configure logging
logger = make_null_logger(__name__)

__all__ = ['ImageView']


class ImageView(CanvasView):
    """The image view"""

    def __init__(self):
        super(ImageView, self).__init__(logger)

        # Enable the image viewing functions.
        self.enable_autocuts('on')
        self.set_autocut_params('zscale')
        self.enable_autozoom('on')
        self.set_zoom_algorithm('rate')
        self.set_zoomrate(1.4)
        self.set_bg(0.2, 0.2, 0.2)
        self.ui_set_active(True)

        bd = self.get_bindings()
        bd.enable_all(True)

        # Show the mode.
        dc = get_canvas_types()
        self.private_canvas.add(dc.ModeIndicator(corner='ur', fontsize=14))
        bm = self.get_bindmap()
        bm.add_callback('mode-set', lambda *args: self.redraw(whence=3))

    def get_shape_mask(self, mask_type, shape):
        """Return the RegionMask representing the shape"""
        data = self.get_image()
        shape_mask = data.get_shape_mask(shape)
        region_mask = RegionMask(shape_mask, mask_type)
        return region_mask
