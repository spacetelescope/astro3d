from ginga.gw.Viewers import ImageViewCanvas

from ...util.logger import make_logger
from .overlay import OverlayView

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

        self.logger.debug('Creating overlay.')
        self.overlay = OverlayView(parent=self, model=model)
        self.logger.debug('Overlay created.')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.overlay.model = model

    def update(self, *args, **kwargs):
        """Update the image display"""
        self.logger.debug(
            'Updating args="{}" kwargs="{}"'.format(args, kwargs)
        )
