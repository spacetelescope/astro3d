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
        self._build_gui()

        signaldb.NewRegion.connect(self.new_region)

    @property
    def canvas(self):
        """The canvas the draw object will appear"""
        return self._canvas

    @canvas.setter
    def canvas(self, canvas):
        if canvas is None or \
           self._canvas is canvas:
            return
        try:
            self.enabled = False
        except AttributeError:
            pass

        self._canvas = canvas
        self.drawtypes = self.canvas.get_drawtypes()
        self.drawtypes.sort()
        self._build_gui()

        # Setup for actual drawing
        canvas.enable_draw(True)
        canvas.enable_edit(True)
        canvas.set_callback('draw-event', self.draw_cb)
        canvas.set_callback('edit-event', self.edit_cb)
        canvas.set_callback('edit-select', self.edit_select_cb)
        canvas.setSurface(self.surface)
        canvas.register_for_cursor_drawing(self.surface)
        canvas.set_draw_mode('draw')

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, state):
        self.logger.debug('Called: state="{}"'.format(state))
        self._enabled = state
        try:
            self._canvas.ui_setActive(state)
        except AttributeError:
            pass

    def new_region(self, type_item):
        self.logger.debug('Called with type_item="{}"'.format(type_item))
        self.type_item = type_item
        self.canvas = type_item.view.canvas
        self.enabled = True

    def set_drawparams(self):
        kind = self.drawtypes[self.drawtype_widget.currentIndex()]
        params = {
            'color': 'red',
            'alpha': 0.0,
            'fill': True
        }
        self.canvas.set_drawtype(kind, **params)

    def draw_cb(self, canvas, tag):
        """Draw callback"""
        self.logger.debug('Called: canvas="{}" shape_id="{}"'.format(canvas, tag))
        shape = canvas.get_object_by_tag(tag)
        self.type_item.add_shape(shape, tag)
        self.enabled = False

    def edit_cb(self, *args, **kwargs):
        """Edit callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def edit_select_cb(self, *args, **kwargs):
        """Edit selected object callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def _build_gui(self):
        """Build out the GUI"""
        # Remove old layout
        self.logger.debug('Called.')
        if self.layout() is not None:
            QtGui.QWidget().setLayout(self.layout())

        # Select drawing types
        drawtype_widget = QtGui.QComboBox()
        self.drawtype_widget = drawtype_widget
        for name in self.drawtypes:
            drawtype_widget.addItem(name)
        drawtype_widget.currentIndexChanged.connect(self.set_drawparams)
        try:
            index = self.drawtypes.index('circle')
        except ValueError:
            pass
        else:
            drawtype_widget.setCurrentIndex(index)

        # Put it together
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(2, 2, 2, 2))
        layout.setSpacing(1)
        layout.addWidget(drawtype_widget)
        self.setLayout(layout)
