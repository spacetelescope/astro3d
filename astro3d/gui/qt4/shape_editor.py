"""Shape Editor"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from .. import signaldb


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
        self.surface = kwargs.pop('surface', None)

        super(ShapeEditor, self).__init__(*args, **kwargs)
        self._canvas = None
        self.drawtypes = []
        self.enabled = False
        self._build_gui()

        signaldb.NewRegion.connect(self.new_region)

    @property
    def canvas(self):
        return self._canvas

    @canvas.setter
    def canvas(self, canvas):
        if canvas is None or \
           self._canvas == canvas:
            return
        self._canvas = canvas
        self.drawtypes = self.canvas.get_drawtypes()
        self.drawtypes.sort()
        self._build_gui()

        # Setup for actual drawing
        canvas.enable_draw(True)
        canvas.enable_edit(True)
        canvas.set_drawtype('point', color='cyan')
        canvas.set_callback('draw-event', self.draw_cb)
        canvas.set_callback('edit-event', self.edit_cb)
        canvas.set_callback('edit-select', self.edit_select_cb)
        canvas.set_surface(self.surface)
        canvas.register_for_cursor_drawing(self.surface)

        # Let the user at it
        self.enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, state):
        self._enabled = state
        try:
            self._canvas.enable_draw(state)
        except AttributeError:
            pass

    def new_region(self, region_item):
        self.logger.debug('Called with region_item="{}"'.format(region_item))

    def set_drawparams(self):
        self.logger.debug('Called.')

    def _build_gui(self):
        """Build out the GUI"""

        # Select drawing types
        drawtype_widget = QtGui.QComboBox()
        for name in self.drawtypes:
            drawtype_widget.addItem(name)
        try:
            index = self.drawtypes.index('circle')
        except ValueError:
            pass
        else:
            drawtype_widget.setCurrentIndex(index)
        drawtype_widget.activated.connect(self.set_drawparams)
        self.drawtype_widget = drawtype_widget

        # Put it together
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        layout.setSpacing(1)
        layout.addWidget(drawtype_widget)
        self.setLayout(layout)
