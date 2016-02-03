"""Model Items"""
from __future__ import absolute_import, print_function

from collections import (defaultdict, namedtuple)

from ...util.logger import make_logger
from ...external.qt.QtGui import (QAction, QStandardItem)
from ...external.qt.QtCore import (QObject, Qt)

from .. import signaldb

__all__ = [
    'ClusterItem',
    'Clusters',
    'LayerItem',
    'RegionItem',
    'Regions',
    'Stars',
    'Textures',
    'TypeItem',
    'Action',
    'ActionSeparator'
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
    'fillalpha': 0.3
}

DRAW_PARAMS = defaultdict(
    lambda: DRAW_PARAMS_DEFAULT,
    {
        'bulge': _merge_dicts(
            DRAW_PARAMS_DEFAULT,
            {'color': 'blue'}
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

    def __init__(self, *args, **kwargs):
        self.value = kwargs.pop('value', None)
        self.view = kwargs.pop('view', None)
        logger = kwargs.pop('logger', make_logger('LayerItem'))
        super(LayerItem, self).__init__(*args, **kwargs)
        if self.__class__.logger is None:
            self.__class__.logger = logger
        self._currentrow = None

    def __iter__(self):
        self._currentrow = None
        return self

    def next(self):
        self._currentrow = self._currentrow + 1 \
                           if self._currentrow is not None \
                           else 0
        child = self.child(self._currentrow)
        if child is None:
            raise StopIteration
        return child

    @property
    def value(self):
        """Value of the item"""
        return self._value

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

    def fix_family(self):
        """Change ancestor/children states based on self state"""
        fix_children_availabilty(self)
        if self.isEnabled():
            fix_tristate(self.parent())


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
                text='Remove',
                func=self.remove,
                args=()
            )
        ] + base_actions
        return actions

    def remove(self):
        self.parent().removeRow(self.row())


class ClusterItem(CheckableItem):
    """A cluster"""


class TypeItem(CheckableItem):
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


class Regions(CheckableItem):
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
                text='Add Remove Star',
                func=self.add_region_interactive,
                args=('remove_star',)
            )
        ] + base_actions
        return actions


class Textures(CheckableItem):
    """Textures container"""
    def __init__(self, *args, **kwargs):
        super(Textures, self).__init__(*args, **kwargs)
        self.setText('Textures')


class Clusters(CheckableItem):
    """Cluster container"""
    def __init__(self, *args, **kwargs):
        super(Clusters, self).__init__(*args, **kwargs)
        self.setText('Clusters')

    def add(self, cluster, id):
        item = ClusterItem(id, value=cluster)
        item.setCheckState(Qt.Checked)
        self.appendRow(item)
        item.fix_family()


class Stars(CheckableItem):
    """Stars container"""

    def __init__(self, *args, **kwargs):
        super(Stars, self).__init__(*args, **kwargs)
        self.setText('Stars')

    def add(self, stars, id):
        item = ClusterItem(id, value=stars)
        item.setCheckState(Qt.Checked)
        self.appendRow(item)
        item.fix_family()


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
            fix_children_availabilty(child)