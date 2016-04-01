"""Shape Editor"""

from functools import partial
from numpy import (zeros, uint8)

from ginga import colors
from ginga.misc.Bunch import Bunch
from ginga.RGBImage import RGBImage
from ginga.gw import Widgets

from ...core.region_mask import RegionMask
from ...external.qt import (QtGui, QtCore)
from ...util.logger import make_logger
from .. import signaldb
from ..helps import instructions

__all__ = ['ShapeEditor']


VALID_KINDS = [
    'freepolygon', 'paint',
    'circle', 'rectangle',
    'triangle', 'righttriangle',
    'square', 'ellipse', 'box'
]


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
        self._mode = None
        self.drawkinds = []
        self.enabled = False
        self.canvas = canvas
        self.mask = None
        self.type_item = None
        self.draw_params = None

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
        self.drawkinds = VALID_KINDS

        # Create paint mode
        canvas.add_draw_mode(
            'paint',
            down=self.paint_start,
            move=self.paint_stroke,
            up=self.paint_stroke_end
        )

        # Setup common events.
        canvas.set_callback('draw-event', self.draw_cb)
        canvas.set_callback('edit-event', self.edit_cb)
        canvas.set_callback('edit-select', self.edit_select_cb)

        # Initial canvas state
        canvas.enable_edit(True)
        canvas.enable_draw(True)
        canvas.setSurface(self.surface)
        canvas.register_for_cursor_drawing(self.surface)
        self._build_gui()
        self.mode = None
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

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):

        # Close off the current state.
        self.logger.debug('Called new_mode="{}"'.format(new_mode))
        for mode in self.mode_frames:
            for frame in self.mode_frames[mode]:
                frame.hide()
        try:
            for frame in self.mode_frames[new_mode]:
                frame.show()
        except KeyError:
            """Doesn't matter if there is nothing to show."""
            pass

        # Setup the new mode.
        canvas_mode = new_mode
        if new_mode is None:
            canvas_mode = 'edit'
        elif new_mode == 'edit_select':
            canvas_mode = 'edit'
        elif new_mode == 'paint':
            self.new_mask()
        elif new_mode == 'paint_edit':
            canvas_mode = 'paint'
        self.canvas.set_draw_mode(canvas_mode)

        # Success. Remember the mode
        self._mode = new_mode
        self.children.tw.set_text(instructions[new_mode])

    def new_region(self, type_item):
        self.logger.debug('Called type_item="{}"'.format(type_item))
        if self.canvas is None:
            raise RuntimeError('Internal error: no canvas to draw on.')

        self.type_item = type_item
        self.draw_params = type_item.draw_params
        self.set_drawparams_cb()

    def set_drawparams_cb(self, kind=None):
        params = {}
        if kind is None:
            kind = self.get_selected_kind()
        if kind == 'paint':
            self.mode = 'paint'
        else:
            self.mode = 'draw'
            try:
                params.update(self.draw_params)
            except AttributeError:
                pass
            self.canvas.set_drawtype(kind, **params)

    def draw_cb(self, canvas, tag):
        """Shape draw completion"""
        shape = canvas.get_object_by_tag(tag)
        shape.type_draw_params = self.draw_params
        region_mask = partial(
            self.surface.get_shape_mask,
            self.type_item.text(),
            shape
        )
        shape.item = self.type_item.add_shape(
            shape=shape,
            mask=region_mask,
            id='{}{}'.format(shape.kind, tag)
        )
        self.mode = None
        self.type_item = None
        self.draw_params = None

    def edit_cb(self, *args, **kwargs):
        """Edit callback"""
        self.logger.debug('Called with args="{}" kwargs="{}".'.format(args, kwargs))
        signaldb.ModelUpdate()

    def edit_select_cb(self, canvas, obj):
        """Edit selected object callback"""
        if self.canvas.num_selected() > 0:
            self.draw_params = obj.type_draw_params
            self.mode = 'edit_select'
            signaldb.LayerSelected(
                selected_item=obj.item,
                source='edit_select_cb'
            )
        else:
            self.mode = None

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

    def new_brush(self, copy_from=None):
        """Create a new brush shape"""
        brush_size = self.children.brush_size.get_value()
        brush = self.canvas.get_draw_class('squarebox')(
            x=0., y=0., radius=max(brush_size / 2, 0.5),
            **self.draw_params
        )
        if copy_from is not None:
            brush.x = copy_from.x
            brush.y = copy_from.y
            brush.radius = copy_from.radius
        self.canvas.add(brush)
        return brush

    def paint_start(self, canvas, event, data_x, data_y, surface):
        """Start a paint stroke"""
        self.brush = self.new_brush()
        self.brush_move(data_x, data_y)

    def paint_stroke(self, canvas, event, data_x, data_y, surface):
        """Perform a paint stroke"""
        previous = self.brush
        self.brush = self.new_brush(previous)
        self.brush_move(data_x, data_y)
        self.stroke(previous, self.brush)
        self.canvas.delete_object(previous)
        self.canvas.redraw(whence=0)

    def paint_stroke_end(self, canvas, event, data_x, data_y, surface):
        self.paint_stroke(canvas, event, data_x, data_y, surface)
        self.canvas.delete_object(self.brush)

        # If starting to paint, go into edit mode.
        self.finalize_paint()

    def paint_stop(self):
        self.finalize_paint()
        self.mode = None

    def finalize_paint(self):
        """Finalize the paint mask"""
        try:
            self.canvas.delete_object(self.brush)
        except AttributeError:
            """If no brush, we were not painting"""
            return

        # If mode is paint_edit, there is no
        # reason to create the item.
        if self.mode == 'paint':
            if self.mask.any():
                shape = self.mask_image
                self.canvas.delete_object(shape)
                shape.type_draw_params = self.draw_params
                region_mask = partial(
                    image_shape_to_regionmask,
                    shape=shape,
                    mask_type=self.type_item.text()
                )
                shape.item = self.type_item.add_shape(
                    shape=shape,
                    mask=region_mask,
                    id='mask{}'.format(self.mask_id)
                )
                signaldb.LayerSelected(selected_item=shape.item, source='finalize_paint')

    def stroke(self, previous, current):
        """Stroke to current brush position"""
        # Due to possible object deletion from an update
        # to treeview, ensure that the image mask is still
        # on the canvas.
        if self.mask_id not in self.canvas.tags:
            self.mask_id = self.canvas.add(self.mask_image)

        # Create a polygon between brush positions.
        poly_points = get_bpoly(previous, current)
        polygon = self.canvas.get_draw_class('polygon')(
            poly_points,
            **self.draw_params
        )
        self.canvas.add(polygon)
        view, contains = self.surface.get_image().get_shape_view(polygon)
        if self.painting:
            self.mask[view][contains] = self.draw_params['fillalpha'] * 255
        else:
            self.mask[view][contains] = 0
        self.canvas.delete_object(polygon)

    def new_mask(self):
        self.logger.debug('Called.')
        self.draw_params = self.type_item.draw_params
        color = self.draw_params['color']
        r, g, b = colors.lookup_color(color)
        height, width = self.surface.get_image().shape
        rgbarray = zeros((height, width, 4), dtype=uint8)
        mask_rgb = RGBImage(data_np=rgbarray)
        mask_image = self.canvas.get_draw_class('image')(0, 0, mask_rgb)
        rc = mask_rgb.get_slice('R')
        gc = mask_rgb.get_slice('G')
        bc = mask_rgb.get_slice('B')
        rc[:] = int(r * 255)
        gc[:] = int(g * 255)
        bc[:] = int(b * 255)
        alpha = mask_rgb.get_slice('A')
        alpha[:] = 0
        self.mask = alpha
        self.mask_image = mask_image
        self.mask_id = self.canvas.add(mask_image)

    def get_selected_kind(self):
        kind = self.drawkinds[self.children.draw_type.get_index()]
        return kind

    def brush_move(self, x, y):
        self.brush.move_to(x, y)
        self.canvas.update_canvas(whence=3)

    def set_painting(self, state):
        """Set painting mode

        Parameters
        ----------
        state: bool
            True for painting, False for erasing
        """
        self.logger.debug('Called state="{}"'.format(state))
        self.painting = state

    def select_layer(self,
                     selected_item=None,
                     deselected_item=None,
                     source=None):
        """Change layer selection"""
        self.logger.debug(
            'selected="{}" deselected="{}" source="{}"'.format(
                selected_item, deselected_item, source
            )
        )

        # If the selection was initiated by
        # selecting the object directly, there is
        # no reason to handle here.
        if source == 'edit_select_cb':
            return

        self.mode = None
        try:
            self.canvas.select_remove(deselected_item.view)
        except AttributeError:
            """We tried. No matter"""
            pass

        try:
            shape = selected_item.view
            self.draw_params = shape.type_draw_params
        except AttributeError:
            """We tried. No matter"""
            pass
        else:
            if shape.kind == 'image':
                self.mask_image = shape.get_image()
                self.mask = self.mask_image.get_slice('A')
                self.mask_id = selected_item.text()
                self.draw_params = shape.type_draw_params
                self.mode = 'paint_edit'
            else:
                x, y = selected_item.view.get_center_pt()
                self.canvas._prepare_to_move(selected_item.view, x, y)
                self.mode = 'edit_select'

        self.canvas.process_drawing()

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

        draw_frame = Widgets.Frame("Drawing")
        draw_frame.set_widget(dtypes_widget)

        # Setup for painting
        captions = (
            ('Brush size:', 'label', 'Brush size', 'spinbutton'),
            ('Paint mode: ', 'label',
             'Paint', 'radiobutton',
             'Erase', 'radiobutton'),
        )
        paint_widget, paint_bunch = Widgets.build_info(captions)
        self.children.update(paint_bunch)
        brush_size = paint_bunch.brush_size
        brush_size.set_limits(1, 100)
        brush_size.set_value(10)

        painting = paint_bunch.paint
        painting.add_callback(
            'activated',
            lambda widget, value: self.set_painting(value)
        )
        painting.set_state(True)
        self.set_painting(True)

        paint_frame = Widgets.Frame('Painting')
        paint_frame.set_widget(paint_widget)

        # Setup for editing
        captions = (
            ("Rotate By:", 'label', 'Rotate By', 'entry'),
            ("Scale By:", 'label', 'Scale By', 'entry')
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
        layout.addWidget(draw_frame.get_widget(), stretch=0)
        layout.addWidget(paint_frame.get_widget(), stretch=0)
        layout.addWidget(edit_frame.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=1)
        self.setLayout(layout)

        # Setup mode frames
        self.mode_frames = {
            'draw': [draw_frame],
            'edit_select': [edit_frame],
            'paint': [
                draw_frame,
                paint_frame
            ],
            'paint_edit': [paint_frame]
        }


def get_bpoly(box1, box2):
    """Get the bounding polygon of two boxes"""
    left = box1
    right = box2
    if box2.x < box1.x:
        left = box2
        right = box1
    left_llx, left_lly, left_urx, left_ury = left.get_llur()
    right_llx, right_lly, right_urx, right_ury = right.get_llur()

    b = [(left_llx, left_lly)]
    if left.y <= right.y:
        b.extend([
            (left_urx, left_lly),
            (right_urx, right_lly),
            (right_urx, right_ury),
            (right_llx, right_ury),
            (left_llx, left_ury)
        ])
    else:
        b.extend([
            (right_llx, right_lly),
            (right_urx, right_lly),
            (right_urx, right_ury),
            (left_urx, left_ury),
            (left_llx, left_ury)
        ])
    return b


def corners(box):
    """Get the corners of a box

    Returns
    -------
    List of the corners starting at the
    lower left, going counter-clockwise
    """
    xll, yll, xur, yur = box.get_llur()
    corners = [
        (xll, yll),
        (xll, yur),
        (xur, yur),
        (xur, yll)
    ]
    return corners


def image_shape_to_regionmask(shape, mask_type):
    """Convert and Image shape to regionmask"""
    return RegionMask(
        mask=shape.get_image().get_slice('A') > 0,
        mask_type=mask_type
    )
