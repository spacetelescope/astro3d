from ginga.gw.Viewers import CanvasView
from ginga.canvas.CanvasObject import get_canvas_types

from ...util.logger import make_logger
from ...core.model3d import Model3D
from ...core.region_mask import RegionMask

__all__ = ['ImageView']


class ImageView(CanvasView):

    def __init__(self, logger=None, model=None):
        if logger is None:
            logger = make_logger('astro3d ImageView')
        self.logger = logger

        super(ImageView, self).__init__(self.logger)

        # Enable the image viewing functions.
        self.enable_autocuts('on')
        self.set_autocut_params('zscale')
        self.enable_autozoom('on')
        self.set_zoom_algorithm('rate')
        self.set_zoomrate(1.4)
        self.set_bg(0.2, 0.2, 0.2)
        self.ui_setActive(True)

        bd = self.get_bindings()
        bd.enable_all(True)

        # Show the mode.
        dc = get_canvas_types()
        self.private_canvas.add(dc.ModeIndicator(corner='ur', fontsize=14))
        bm = self.get_bindmap()
        bm.add_callback('mode-set', lambda *args: self.redraw(whence=3))

    def get_shape_mask(self, mask_type, shape, model):
        """Return the RegionMask representing the shape"""
        self.logger.debug('Called.')
        self.logger.debug('mask_type="{}"'.format(mask_type))
        self.logger.debug('shape="{}"'.format(shape))
        self.logger.debug('model="{}"'.format(model))
        data = self.get_image()
        shape_mask = data.get_shape_mask(shape)
        self.logger.debug('shape_mask.shape="{}"'.format(shape_mask.shape))

        # If a model was given, create one and get the size.
        model_shape = None
        if model is not None:
            self.logger.debug('Creating model3D')
            model3d = Model3D(model.image, **model.params.model)
            model3d._prepare_data()
            model_shape = model3d.data_original_resized.shape
            self.logger.debug('model_shape="{}"'.format(model_shape))

        self.logger.debug('again mask_type="{}"'.format(mask_type))
        model_shape = None
        region_mask = RegionMask(shape_mask, mask_type, shape=model_shape)
        self.logger.debug('Done.')
        return region_mask
