"""Shape Editor"""

from functools import partial

from ginga.misc.Bunch import Bunch
from ginga.gw import Widgets

from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from .. import signaldb


__all__ = ['ShapeEditor']


VALID_KINDS = set((
    'freepolygon', 'freepath',
    'circle', 'rectangle',
    'triangle', 'righttriangle',
    'square', 'ellipse', 'box'
))

INSTRUCTIONS = (
    'Draw a region with the cursor. '
    'For polygons/paths press \'v\' to create a vertex, '
    '\'z\' to remove last vertex.'
)


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
        canvas = kwargs.pop('canvas', None)

        super(ShapeEditor, self).__init__(*args, **kwargs)

        self._canvas = None
        self.drawkinds = []
        self.enabled = False
        self.canvas = canvas

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

        self._canvas = canvas

        # Setup parameters
        self.drawkinds = list(
            VALID_KINDS.intersection(self.canvas.get_drawtypes())
        )
        self.drawkinds.sort()

        # Setup for actual drawing
        canvas.enable_draw(True)
        canvas.enable_edit(True)
        canvas.set_callback('draw-event', self.draw_cb)
        canvas.set_callback('edit-event', self.edit_cb)
        canvas.set_callback('edit-select', self.edit_select_cb)
        canvas.setSurface(self.surface)
        canvas.register_for_cursor_drawing(self.surface)
        canvas.set_draw_mode('edit')
        self._build_gui()
        self.enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, state):
        self._enabled = state
        try:
            self._canvas.ui_setActive(state)
        except AttributeError:
            pass

    def new_region(self, type_item):
        self.logger.debug('Called type_item="{}"'.format(type_item))
        self.type_item = type_item
        if self.canvas is None:
            raise RuntimeError('Internal error: no canvas to draw on.')
        self.set_drawparams_cb()
        self.canvas.set_draw_mode('draw')

    def set_drawparams_cb(self):
        self.logger.debug('children="{}"'.format(self.children))
        kind = self.drawkinds[self.children.draw_type.get_index()]
        try:
            params = self.type_item.draw_params
        except AttributeError:
            params = {}
        self.canvas.set_drawtype(kind, **params)

    def draw_cb(self, canvas, tag):
        """Draw callback"""
        self.canvas.set_draw_mode('edit')
        shape = canvas.get_object_by_tag(tag)
        region_mask = partial(
            self.surface.get_shape_mask,
            self.type_item.text(),
            shape
        )
        self.type_item.add_shape(shape=shape, mask=region_mask, id=tag)

    def edit_cb(self, *args, **kwargs):
        """Edit callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))
        signaldb.ModelUpdate()

    def edit_select_cb(self, *args, **kwargs):
        """Edit selected object callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))

    def edit_deselect_cb(self, *args, **kwargs):
        """Deselect"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))
        self.canvas.clear_selected()

    def rotate_object(self, w):
        delta = float(w.get_text())
        self.canvas.edit_rotate(delta, self.surface)
        signaldb.ModelUpdate()

    def scale_object(self, w):
        delta = float(w.get_text())
        self.canvas.edit_scale(delta, delta, self.surface)
        signaldb.ModelUpdate()

    def _build_gui(self):
        """Build out the GUI"""
        # Remove old layout
        self.logger.debug('Called.')
        if self.layout() is not None:
            QtGui.QWidget().setLayout(self.layout())
        self.children = Bunch()
        spacer = Widgets.Label('')

        # Instructions
        tw = Widgets.TextArea(wrap=True, editable=False)
        font = QtGui.QFont('sans serif', 12)
        tw.set_font(font)
        tw.set_text(INSTRUCTIONS)
        tw_frame = Widgets.Expander("Instructions")
        tw_frame.set_widget(tw)
        self.children['tw'] = tw
        self.children['tw_frame'] = tw_frame

        # Setup for the drawing types
        captions = (
            ("Draw type:", 'label', "Draw type", 'combobox'),
        )
        dtypes_widget, dtypes_bunch = Widgets.build_info(captions)
        self.children.update(dtypes_bunch)
        self.logger.debug('children="{}"'.format(self.children))

        combobox = dtypes_bunch.draw_type
        for name in self.drawkinds:
            combobox.append_text(name)
        index = self.drawkinds.index('freepolygon')
        combobox.add_callback(
            'activated',
            lambda w, idx: self.set_drawparams_cb()
        )
        combobox.set_index(index)

        dtypes_frame = Widgets.Frame("Drawing")
        dtypes_frame.set_widget(dtypes_widget)

        # Setup for editing
        captions = (("Rotate By:", 'label', 'Rotate By', 'entry'),
                    ("Scale By:", 'label', 'Scale By', 'entry'),
        )
        edit_widget, edit_bunch = Widgets.build_info(captions)
        self.children.update(edit_bunch)
        edit_bunch.scale_by.add_callback('activated', self.scale_object)
        edit_bunch.scale_by.set_text('0.9')
        edit_bunch.scale_by.set_tooltip("Scale selected object in edit mode")
        edit_bunch.rotate_by.add_callback('activated', self.rotate_object)
        edit_bunch.rotate_by.set_text('90.0')
        edit_bunch.rotate_by.set_tooltip("Rotate selected object in edit mode")

        edit_frame = Widgets.Frame('Editing')
        edit_frame.set_widget(edit_widget)

        # Put it together
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(20, 20, 20, 20))
        layout.setSpacing(1)
        layout.addWidget(tw_frame.get_widget(), stretch=0)
        layout.addWidget(dtypes_frame.get_widget(), stretch=0)
        layout.addWidget(edit_frame.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=1)
        self.setLayout(layout)
