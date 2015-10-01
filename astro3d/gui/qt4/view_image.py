from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas

from ...util.logger import make_logger
from .overlay import RegionsOverlay

__all__ = ['ViewImage']


class ViewImage(ImageViewCanvas):

    def __init__(self, logger=None, model=None):
        if logger is None:
            logger = make_logger('astro3d ViewImage')

        self.logger = logger

        super(ViewImage, self).__init__(self.logger, render='widget')

        # Enable the image viewing functions.
        self.enable_autocuts('on')
        self.set_autocut_params('zscale')
        self.enable_autozoom('on')
        self.set_bg(0.2, 0.2, 0.2)
        self.ui_setActive(True)
        self.enable_draw(False)

        bd = self.get_bindings()
        bd.enable_pan(True)
        bd.enable_zoom(True)
        bd.enable_cuts(True)
        bd.enable_flip(True)

        self.overlay = RegionsOverlay(parent=self)
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def update(self):
        """Update the image display"""
        self.logger.debug('Updating...')

        self._update_overlays()

    def _update_overlays(self):
        """Update the overlays from the model"""
        self.logger.debug('updating overlays')

        overlay = self.overlay
        overlay.delete_all_objects()
        for region in self.model.regions:
            self.logger.debug('overlaying region "{}"'.format(region))
            overlay.add_region(region)
