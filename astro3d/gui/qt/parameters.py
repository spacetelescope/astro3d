"""Parameter Editor"""

from ginga.misc.Bunch import Bunch
from ginga.gw import Widgets
from qtpy import (QtCore, QtWidgets)
from qtpy.QtCore import Qt

from ...util.logger import make_null_logger
from ..store_widgets import StoreWidgets

# Configure logging
logger = make_null_logger(__name__)

__all__ = ['Parameters']


class Parameters(QtWidgets.QScrollArea):
    """Parameters Editor
    """

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.parent = kwargs.pop('parent')
        super(Parameters, self).__init__(*args, **kwargs)

        self._build_gui()

    def preview_model(self):
        self.parent.mesh_viewer.setVisible(True)
        self.parent.force_update()

    def create_gas_spiral_masks(self, *args, **kwargs):
        """Set Gas/Spiral arm esitmator parameters"""
        self.model.create_gas_spiral_masks(
            smooth_size=self.children.smooth_size.get_value(),
            gas_percentile=self.children.gas_percentile.get_value(),
            spiral_percentile=self.children.spiral_percentile.get_value(),
            model_params=self.model.params.model
        )

    def create_gas_dust_masks(self, *args, **kwargs):
        """Set Gas/Dust esitmator parameters"""
        params = self.children['gasdust']
        self.model.create_gas_dust_masks(
            smooth_size=params.smooth_size.get_value(),
            gas_percentile=params.gas_percentile.get_value(),
            dust_percentile=params.dust_percentile.get_value(),
            model_params=self.model.params.model
        )

    def _build_gui(self):
        """Build out the GUI"""
        logger.debug('Called.')
        self.children = Bunch()
        spacer = Widgets.Label('')

        # Processing parameters
        captions = [('Save Model', 'button'),
                    ('Preview Model', 'button')]
        params_store = StoreWidgets(
            self.model.params.stages,
            extra=captions
        )
        self.model.params_widget_store.stages = params_store
        params_widget = params_store.container
        params_bunch = params_store.widgets
        self.children.update(params_bunch)

        params_bunch.save_model.add_callback(
            'activated',
            lambda w: self.parent.save_all_from_dialog()
        )

        params_bunch.preview_model.add_callback(
            'activated',
            lambda w: self.preview_model()

        )

        params_frame = Widgets.Frame('Processing')
        params_frame.set_widget(params_widget)

        # Model parameters
        model_store = StoreWidgets(self.model.params.model)
        self.model.params_widget_store.model = model_store
        model_widget = model_store.container
        model_bunch = model_store.widgets
        self.children.update(model_bunch)

        model_frame = Widgets.Frame()
        model_frame.set_widget(model_widget)
        model_expander = Widgets.Expander('Model Params')
        model_expander.set_widget(model_frame)
        self.children['model_expander'] = model_expander

        # Model Making parameters
        model_make_store = StoreWidgets(self.model.params.model_make)
        self.model.params_widget_store.model_make = model_make_store
        model_make_widget = model_make_store.container
        model_make_bunch = model_make_store.widgets
        self.children.update(model_make_bunch)

        model_make_frame = Widgets.Frame()
        model_make_frame.set_widget(model_make_widget)
        model_make_expander = Widgets.Expander('Model Making Params')
        model_make_expander.set_widget(model_make_frame)
        self.children['model_make_expander'] = model_make_expander

        # Gas/Spiral parameters
        captions = (
            ('Gas Percentile:', 'label', 'Gas Percentile', 'spinbutton'),
            ('Spiral Percentile:', 'label', 'Spiral Percentile', 'spinbutton'),
            ('Smooth Size:', 'label', 'Smooth Size', 'spinbutton'),
            ('Create masks', 'button'),
        )
        gasspiral_widget, gasspiral_bunch = Widgets.build_info(captions)
        self.children.update(gasspiral_bunch)

        gasspiral_bunch.gas_percentile.set_limits(0., 100.)
        gasspiral_bunch.gas_percentile.set_value(55)
        gasspiral_bunch.gas_percentile.set_tooltip(
            'The percentile of values above which'
            ' are assigned to the Gas mask'
        )

        gasspiral_bunch.spiral_percentile.set_limits(0., 100.)
        gasspiral_bunch.spiral_percentile.set_value(75)
        gasspiral_bunch.spiral_percentile.set_tooltip(
            'The percential of values above which are'
            ' assigned to the Spiral Arm mask'
        )

        gasspiral_bunch.smooth_size.set_limits(3, 100)
        gasspiral_bunch.smooth_size.set_value(11)
        gasspiral_bunch.smooth_size.set_tooltip(
            'Size of the smoothing window'
        )

        gasspiral_bunch.create_masks.add_callback(
            'activated',
            self.create_gas_spiral_masks
        )

        gasspiral_frame = Widgets.Frame('Gas/Spiral Arm parameters')
        gasspiral_frame.set_widget(gasspiral_widget)

        # Gas/Dust parameters
        captions = (
            ('Gas Percentile:', 'label', 'Gas Percentile', 'spinbutton'),
            ('Dust Percentile:', 'label', 'Dust Percentile', 'spinbutton'),
            ('Smooth Size:', 'label', 'Smooth Size', 'spinbutton'),
            ('Create masks', 'button'),
        )
        gasdust_widget, gasdust_bunch = Widgets.build_info(captions)
        self.children['gasdust'] = gasdust_bunch

        gasdust_bunch.gas_percentile.set_limits(0., 100.)
        gasdust_bunch.gas_percentile.set_value(75.)
        gasdust_bunch.gas_percentile.set_tooltip(
            'The percentile of values above which'
            ' are assigned to the Gas mask'
        )

        gasdust_bunch.dust_percentile.set_limits(0., 100.)
        gasdust_bunch.dust_percentile.set_value(55.)
        gasdust_bunch.dust_percentile.set_tooltip(
            'The percentile of pixel values in the weighted data above'
            'which (and below gas_percentile) to assign to the "dust"'
            'mask.  dust_percentile must be lower than'
            'gas_percentile.'
        )

        gasdust_bunch.smooth_size.set_limits(3, 100)
        gasdust_bunch.smooth_size.set_value(11)
        gasdust_bunch.smooth_size.set_tooltip(
            'Size of the smoothing window'
        )

        gasdust_bunch.create_masks.add_callback(
            'activated',
            self.create_gas_dust_masks
        )

        gasdust_frame = Widgets.Frame('Gas/Dust parameters')
        gasdust_frame.set_widget(gasdust_widget)

        # Put it together
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(20, 20, 20, 20))
        layout.setSpacing(1)
        layout.addWidget(params_frame.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=1)
        layout.addWidget(model_expander.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=1)
        layout.addWidget(model_make_expander.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=1)
        layout.addWidget(gasspiral_frame.get_widget(), stretch=0)
        layout.addWidget(gasdust_frame.get_widget(), stretch=0)
        layout.addWidget(spacer.get_widget(), stretch=2)
        content = QtWidgets.QWidget()
        content.setLayout(layout)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(content)
