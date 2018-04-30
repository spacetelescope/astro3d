"""Model Items"""
from __future__ import absolute_import, print_function

from collections import (defaultdict, namedtuple)
import copy
from itertools import count

from astropy.table import Table
from ginga.canvas.types.image import Image
from qtpy import (QtCore, QtGui, QtWidgets)

from ...util.logger import make_logger
from ...core.image_utils import combine_region_masks
from ...core.region_mask import RegionMask

from .. import signaldb

# Shortcuts
QAction = QtWidgets.QAction
QStandardItem = QtGui.QStandardItem
Qt = QtCore.Qt
QObject = QtCore.QObject


__all__ = [
    'ClusterItem',
    'Clusters',
    'LayerItem',
    'RegionItem',
    'Regions',
    'Stars',
    'StarsItem',
    'Textures',
    'TypeItem',
    'Action',
    'ActionSeparator',
]


def _merge_dicts(*dictionaries):
    result = {}
    for dictionary in dictionaries:
        result.update(dictionary)
    return result

DRAW_PARAMS_DEFAULT = {
    'color': 'red',
    'alpha': 0.3,
    'fill': True,
    'fillalpha': 0.3,
    'linewidth': 0.,
}

DRAW_PARAMS = defaultdict(
    lambda: DRAW_PARAMS_DEFAULT,
    {
        'bulge': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'blue'}
        ),
        'cluster': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'darkgoldenrod',
             'linewidth': 10,
             'fill': False,
             'radius': 4.0}
        ),
        'disk': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'cornflowerblue'}
        ),
        'dust': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'debianred'}
        ),
        'filament': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'aquamarine'}
        ),
        'gas': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'green'}
        ),
        'remove_star': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'red'}
        ),
        'spiral': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'orange'}
        ),
        'stars': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'purple',
             'linewidth': 10,
             'fill': False,
             'radius': 4.0}
        )
    }
)

Action = namedtuple('Action', ('text', 'func', 'args'))


class ActionSeparator(QAction):
    """Indicate a separator"""
    def __init__(self):
        self.obj = QObject()
        super(ActionSeparator, self).__init__(self.obj)
        self.setSeparator(True)


class InstanceDefaultDict(defaultdict):
    """A default dict with class instantion using the key as argument"""
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            self[key] = self.default_factory(key)
            return self[key]


class LayerItem(QStandardItem):
    """Layers

    Parameters
    ----------
    All `QStandardItem` parameters plus:

    value: type
        Specific value this item is associated with

    view: `ginga shape`
        How this item is viewed.
    """

    logger = None

    # Use sequence to create unique identifiers
    _sequence = count(1)

    def __init__(self, *args, **kwargs):
        self.value = kwargs.pop('value', None)
        self.view = kwargs.pop('view', None)
        logger = kwargs.pop('logger', make_logger('LayerItem'))
        super(LayerItem, self).__init__(*args, **kwargs)
        if LayerItem.logger is None:
            LayerItem.logger = logger

    @property
    def value(self):
        """Value of the item"""
        try:
            value = self._value()
        except TypeError:
            value = self._value

        return value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def is_available(self):
        return self.isEnabled() and self.checkState()

    @property
    def _actions(self):
        actions = []
        return actions

    def children(self):
        """Iterator returning all children"""
        for row in range(self.rowCount()):
            yield self.child(row)

    def available(self):
        """Iterator returning all available children"""
        for child in self.children():
            if child.is_available:
                yield child

    def fix_family(self):
        """Change ancestor/children states based on self state"""
        if self.checkState() == Qt.Unchecked:
            signaldb.LayerSelected(deselected_item=self, source='fix_family')

        fix_children_availabilty(self)
        if self.isEnabled():
            fix_tristate(self.parent())

    def clone(self):
        """Clone this item

        Returns
        -------
        The clone.
        """
        new = self.__class__(logger=self.logger)
        if isinstance(self.view, Image):
            new.value = self.value
            new.view = None
        else:
            new.view = copy.copy(self.view)
            new.view.item = new
            new.value = None

        return new

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        return self.clone()


class FixedMixin(object):
    """Item cannot be edited"""
    def __init__(self, *args, **kwargs):
        super(FixedMixin, self).__init__(*args, **kwargs)
        self.setEditable(False)


class CheckableItem(LayerItem):
    """Items that are checkable"""
    def __init__(self, *args, **kwargs):
        super(CheckableItem, self).__init__(*args, **kwargs)
        self.setCheckable(True)
        self.setCheckState(Qt.Checked)

    @property
    def _actions(self):
        actions = super(CheckableItem, self)._actions
        actions.extend([
            ActionSeparator(),
            Action(
                text='Hide' if self.checkState() else 'Show',
                func=self.toggle_available,
                args=()
            )
        ])
        return actions

    def toggle_available(self):
        self.setCheckState(Qt.Unchecked if self.checkState() else Qt.Checked)


class RegionItem(CheckableItem):
    """The regions"""
    @property
    def _actions(self):
        base_actions = super(RegionItem, self)._actions
        actions = [
            Action(
                text='Duplicate',
                func=self.duplicate,
                args=()
            ),
            Action(
                text='Remove',
                func=self.remove,
                args=()
            ),
        ] + base_actions
        return actions

    def remove(self):
        self.parent().removeRow(self.row())

    def duplicate(self):
        """Duplicate this item and put into model"""
        new = self.clone()
        new.setText(self.text() + 'copy@' + str(next(self._sequence)))
        self.parent().appendRow(new)
        new.fix_family()


class CatalogItem(CheckableItem):
    """Items that are catalogs"""
    @property
    def _actions(self):
        base_actions = super(ClusterItem, self)._actions
        actions = [
            Action(
                text='Remove',
                func=self.remove,
                args=()
            )
        ] + base_actions
        return actions

    def remove(self):
        self.parent().removeRow(self.row())

    def add_entry(self, x, y, flux=1.0):
        """Add an entry"""
        table = self.value
        table.add_row([x, y, flux])
        self.emitDataChanged()

    def remove_entry(self, idx):
        """Remove entry from the catalog"""
        table = self.value
        try:
            table.remove_row(idx)
        except Exception as e:
            self.logger.debug(e)
        else:
            self.logger.debug('removal successful! Signalling...')
            self.emitDataChanged()

    def key_callback(self, draw_obj, canvas_obj, event, coords):
        """Handle key-press callback"""
        self.logger.debug(
            'obj.idx="{}" event="{}" key="{}"'.format(
                draw_obj.idx, event, event.key
            )
        )
        if event.key == 'd':
            self.remove_entry(draw_obj.idx)
        elif event.key == 's':
            self.add_entry(*coords)


class ClusterItem(CatalogItem):
    """A cluster"""
    def __init__(self, *args, **kwargs):
        super(ClusterItem, self).__init__(*args, **kwargs)
        self.draw_params = DRAW_PARAMS['cluster']


class StarsItem(CatalogItem):
    """A star list"""
    def __init__(self, *args, **kwargs):
        super(StarsItem, self).__init__(*args, **kwargs)
        self.draw_params = DRAW_PARAMS['stars']


class TypeItem(FixedMixin, CheckableItem):
    """Types of regions"""
    def __init__(self, *args, **kwargs):
        super(TypeItem, self).__init__(*args, **kwargs)

        self.draw_params = DRAW_PARAMS[self.text()]

    @property
    def _actions(self):
        base_actions = super(TypeItem, self)._actions
        actions = [
            Action(
                text='Add Region',
                func=self.add_region_interactive,
                args=()
            ),
            Action(
                text="Merge regions",
                func=self.merge_masks,
                args=()
            ),
        ] + base_actions
        return actions

    def add_region_interactive(self):
        """Add a new region."""
        signaldb.NewRegion(self)

    def add_shape(self, shape, mask, id):
        region_item = RegionItem(id, value=mask, view=shape)
        region_item.setCheckState(Qt.Checked)
        self.appendRow(region_item)
        region_item.fix_family()
        return region_item

    def add_mask(self, mask, id):
        region_item = RegionItem(id, value=mask)
        region_item.setCheckState(Qt.Checked)
        self.appendRow(region_item)
        region_item.fix_family()
        return region_item

    def merge_masks(self):
        """Merge all masks"""
        regionmasks = []
        for region in self.available():
            regionmasks.append(region.value)
            region.toggle_available()

        if len(regionmasks) == 0:
            return
        mergedmask = combine_region_masks(regionmasks)
        merged = RegionMask(mergedmask, self.text())
        id = 'merged@' + str(next(self._sequence))
        self.add_mask(merged, id)


class Regions(FixedMixin, CheckableItem):
    """Regions container"""

    def __init__(self, *args, **kwargs):
        super(Regions, self).__init__(*args, **kwargs)
        self.setText('Regions')

        self.types = InstanceDefaultDict(TypeItem)

    @property
    def regions(self):
        """Iterate over all the region masks"""
        regions = (
            self.child(type_id).child(region_id).value
            for type_id in range(self.rowCount())
            if self.child(type_id).is_available
            for region_id in range(self.child(type_id).rowCount())
            if self.child(type_id).child(region_id).is_available
        )
        return regions

    @property
    def _actions(self):
        base_actions = super(Regions, self)._actions
        actions = [
            Action(
                text='Add Bulge',
                func=self.add_region_interactive,
                args=('bulge',)
            ),
            Action(
                text='Add Disk',
                func=self.add_region_interactive,
                args=('disk',)
            ),
            Action(
                text='Add Gas',
                func=self.add_region_interactive,
                args=('gas',)
            ),
            Action(
                text='Add Spiral',
                func=self.add_region_interactive,
                args=('spiral',)
            ),
            Action(
                text='Add Dust',
                func=self.add_region_interactive,
                args=('dust',)
            ),
            Action(
                text='Add Filament',
                func=self.add_region_interactive,
                args=('filament',)
            ),
            Action(
                text='Add Remove Star',
                func=self.add_region_interactive,
                args=('remove_star',)
            ),
            ActionSeparator(),
            Action(
                text='Autocreate Gas/Spiral masks',
                func=signaldb.CreateGasSpiralMasks,
                args=()
            ),
            Action(
                text='Merge all regions',
                func=self.merge_masks,
                args=()
            ),
        ] + base_actions
        return actions

    def add_mask(self, mask, id):
        """Add a new region from a RegionMask"""
        type_item = self.types[mask.mask_type]
        region_item = RegionItem(id, value=mask)
        region_item.setCheckState(Qt.Checked)
        type_item.appendRow(region_item)
        if not type_item.index().isValid():
            self.appendRow(type_item)
        region_item.fix_family()

    def add_region_interactive(self, mask_type):
        """Add a type"""
        self.logger.debug('Called mask_type="{}"'.format(mask_type))
        type_item = self.types[mask_type]
        if not type_item.index().isValid():
            self.appendRow(type_item)
        signaldb.NewRegion(type_item)

    def merge_masks(self):
        """Merge masks for all types"""
        for type_id in range(self.rowCount()):
            if self.child(type_id).is_available:
                self.child(type_id).merge_masks()


class Textures(FixedMixin, CheckableItem):
    """Textures container"""
    def __init__(self, *args, **kwargs):
        super(Textures, self).__init__(*args, **kwargs)
        self.setText('Textures')


class Clusters(FixedMixin, CheckableItem):
    """Cluster container"""
    def __init__(self, *args, **kwargs):
        super(Clusters, self).__init__(*args, **kwargs)
        self.setText('Clusters')

    def add(self, catalog, id):
        item = ClusterItem(id, value=catalog)
        item.setCheckState(Qt.Checked)
        self.appendRow(item)
        item.fix_family()

    def new_catalog(self):
        """Create a new catalog"""
        catalog = Table(names=['xcentroid', 'ycentroid', 'flux'])
        self.add(catalog, 'cluster@' + str(next(self._sequence)))

    @property
    def _actions(self):
        base_actions = super(Clusters, self)._actions
        actions = [
            Action(
                text='Add Catalog',
                func=self.new_catalog,
                args=()
            ),
        ] + base_actions
        return actions


class Stars(FixedMixin, CheckableItem):
    """Stars container"""

    def __init__(self, *args, **kwargs):
        super(Stars, self).__init__(*args, **kwargs)
        self.setText('Stars')

    def add(self, catalog, id):
        item = StarsItem(id, value=catalog)
        item.setCheckState(Qt.Checked)
        self.appendRow(item)
        item.fix_family()

    def new_catalog(self):
        """Create a new catalog"""
        catalog = Table(names=['xcentroid', 'ycentroid', 'flux'])
        self.add(catalog, 'stars@' + str(next(self._sequence)))

    @property
    def _actions(self):
        base_actions = super(Stars, self)._actions
        actions = [
            Action(
                text='Add Catalog',
                func=self.new_catalog,
                args=()
            ),
        ] + base_actions
        return actions


# Utilities
def fix_tristate(item):
    """Set tristate based on siblings"""
    if item is not None and item.hasChildren():
        current = item.rowCount() - 1
        state = item.child(current).checkState()
        current -= 1
        while current >= 0:
            if state != item.child(current).checkState():
                state = Qt.PartiallyChecked
                break
            current -= 1
        if state == Qt.Unchecked:
            state = Qt.PartiallyChecked
        item.setCheckState(state)
        fix_tristate(item.parent())


def fix_children_availabilty(item):
    """Enable/disable children based on current state"""
    if item is not None and item.hasChildren():
        enable = item.isEnabled() and \
                 item.checkState() in (Qt.PartiallyChecked, Qt.Checked)
        for idx in range(item.rowCount()):
            child = item.child(idx)
            child.setEnabled(enable)
            if not enable:
                signaldb.LayerSelected(
                    deselected_item=child,
                    source='fix_children_availabilty'
                )
            fix_children_availabilty(child)
