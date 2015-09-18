from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas

from ...util.logger import make_logger


__all__ = ['ViewImage']


class ViewImage(ImageViewCanvas):

    def __init__(self, logger=None):
        if logger is None:
            logger = make_logger('astro3d ViewImage')

        self.logger = logger

        super(ViewImage, self).__init__(self.logger, render='widget')

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
