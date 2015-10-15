"""Shape Editor"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger

__all__ = ['ShapeEditor']


class ShapeEditor(QtGui.QWidget):
    """Shape Editor

    Paremeters
    ----------
    color: str
        Color to draw in.

    canvas: `ginga.canvas`
        The canvas to draw on.
    """

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Shape Editor')
        self.logger = logger
        self.canvas = kwargs.pop('canvas', None)
        self.color = kwargs.pop('color', 'red')

        super(ShapeEditor, self).__init__(*args, **kwargs)

        self._build_gui()

    @property
    def canvas(self):
        return self._canvas

    @canvas.setter
    def canvas(self, canvas):
        if canvas is None or \
           self._canvas == canvas:
            return
        self.enabled = False
        self._canvas = canvas
        self.enabled = True
        self.drawtypes = self.canvas.get_drawtypes()
        self.drawtypes.sort()
        self._build_gui()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, state):
        self._enabled = state
        self._canvas.enable_draw(state)

    def set_drawparams(self, kind):
        index = self.wdrawtype.currentIndex()
        kind = self.drawtypes[index]

        params = {'color': self.color,
                  'alpha': 0.5,
                  'fill': True,
                  'fillalpha': 0.5
        }

        self.canvas.set_drawtype(kind, **params)

    def _build_gui(self):

        if getattr(self, 'wdrawtype', None) is not None:
            self.wdrawtype.hide()
        wdrawtype = QtGui.QComboBox(self)
        try:
            for name in self.drawtypes:
                wdrawtype.addItem(name)
            index = self.drawtypes.index('rectangle')
            wdrawtype.setCurrentIndex(index)
            wdrawtype.activated.connect(self.set_drawparams)
        except AttributeError:
            pass
        self.wdrawtype = wdrawtype
