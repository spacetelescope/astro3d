"""Shape Editor"""

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from .. import signaldb


__all__ = ['ShapeEditor']


class ShapeEditor(QtGui.QWidget):
    """Shape Editor

    Paremeters
    ----------
    surface: `ginga.Canvas`
        The canvas to interact on.

    logger: logging.Logger
        The common logger.
    """
    def __init__(self, *args, **kwargs):
        self.logger = kwargs.pop(
            'logger',
            make_logger('astro3d Shape Editor')
        )
        self.surface = kwargs.pop('surface', None)

        super(ShapeEditor, self).__init__(*args, **kwargs)
        self._canvas = None
        self.drawtypes = []
        self.enabled = False
        self.canvas = self.surface.canvas
        self._build_gui()

        signaldb.NewRegion.connect(self.new_region)

    @property
    def canvas(self):
        """The canvas the draw object will appear"""
        return self._canvas

    @canvas.setter
    def canvas(self, canvas):
        if canvas is None or \
           self._canvas == canvas:
            return
        try:
            self._canvas.ui_setActive(False)
        except AttributeError:
            pass

        self._canvas = canvas
        self.drawtypes = self.canvas.get_drawtypes()
        self.drawtypes.sort()
        self._build_gui()

        # Setup for actual drawing
        canvas.set_drawtype('point', color='cyan')
        canvas.set_callback('draw-event', self.draw_cb)
        canvas.set_callback('edit-event', self.edit_cb)
        canvas.set_callback('edit-select', self.edit_select_cb)
        canvas.register_for_cursor_drawing(self.surface)
        canvas.set_draw_mode('draw')

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, state):
        self._enabled = state
        try:
            self.canvas.enable_draw(state)
            self.canvas.enable_edit(state)
            self.surface.ui_setActive(not state)
            self.canvas.ui_setActive(state)
        except AttributeError:
            pass

    def new_region(self, region_type):
        self.logger.debug('Called with region_type="{}"'.format(region_type))
        self.region_type = region_type
        self._build_gui()
        self.enabled = True

    def set_drawparams(self):
        self.logger.debug('Called.')
        kind = self.drawtypes[self.drawtype_widget.currentIndex()]
        params = {
            'color': 'red',
            'alpha': 0.0,
            'fill': True
        }
        self.canvas.set_drawtype(kind, **params)

    def draw_cb(self, *args, **kwargs):
        """Draw callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def edit_cb(self, *args, **kwargs):
        """Edit callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def edit_select_cb(self, *args, **kwargs):
        """Edit selected object callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def _build_gui(self):
        """Build out the GUI"""
        # Remove old layout
        if self.layout() is not None:
            QtGui.QWidget().setLayout(self.layout())

        # Select drawing types
        drawtype_widget = QtGui.QComboBox()
        self.drawtype_widget = drawtype_widget
        for name in self.drawtypes:
            drawtype_widget.addItem(name)
        try:
            index = self.drawtypes.index('circle')
        except ValueError:
            pass
        else:
            drawtype_widget.setCurrentIndex(index)
        drawtype_widget.activated.connect(self.set_drawparams)

        # Put it together
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        layout.setSpacing(1)
        layout.addWidget(drawtype_widget)
        self.setLayout(layout)
