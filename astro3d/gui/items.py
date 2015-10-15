"""Model Items"""
from __future__ import absolute_import, print_function

from collections import (defaultdict, namedtuple)

from ..external.qt.QtGui import QStandardItem
from ..external.qt.QtCore import Qt


__all__ = [
    'ClusterItem',
    'Clusters',
    'LayerItem',
    'RegionItem',
    'Regions',
    'Stars',
    'Textures',
    'TypeItem',
]

Action = namedtuple('Action', ('text', 'func', 'args'))


class InstanceDefaultDict(defaultdict):
    """A default dict with class instantion using the key as argument"""
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            self[key] = self.default_factory(key)
            return self[key]


class LayerItem(QStandardItem):
    """Layers"""
    def __init__(self, *args, **kwargs):
        value = kwargs.pop('value', None)
        super(LayerItem, self).__init__(*args, **kwargs)
        self._currentrow = None
        self.value = value
        self.view = None

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
        actions = ()
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
        self.setCheckState(Qt.Unchecked)


class RegionItem(CheckableItem):
    """The regions"""
    @property
    def _actions(self):
        actions = (
            Action(text='Remove',
                   func=self.remove,
                   args=()
            ),
        )
        return actions

    def remove(self):
        self.parent().removeRow(self.row())


class ClusterItem(CheckableItem):
    """A cluster"""


class TypeItem(CheckableItem):
    """Types of regions"""
    @property
    def _actions(self):
        actions = (
            Action(text='Add Region',
                   func=self.add_type,
                   args=()
            ),
        )
        return actions

    def add_type(self):
        """Add a new region."""


class Regions(CheckableItem):
    """Regions container"""

    def __init__(self, *args, **kwargs):
        super(Regions, self).__init__(*args, **kwargs)
        self.setText('Regions')

        self.types = InstanceDefaultDict(TypeItem)

    # Regions iterate over all the leaf nodes.
    # These would be the masks themselves.
    @property
    def regions(self):
        regions = (
            self.child(type_id).child(region_id).value
            for type_id in range(self.rowCount())
            if self.child(type_id).is_available
            for region_id in range(self.child(type_id).rowCount())
            if self.child(type_id).child(region_id).is_available
        )
        return regions

    def add(self, region, id):
        """Add a new region"""
        type_item = self.types[region.mask_type]
        region_item = RegionItem(id, value=region)
        region_item.setCheckState(Qt.Checked)
        type_item.appendRow(region_item)
        if not type_item.index().isValid():
            self.appendRow(type_item)
        region_item.fix_family()

    def add_type(self):
        """Add a type"""

    @property
    def _actions(self):
        actions = (
            Action(text='Add Bulge',
                   func=self.add_type,
                   args=('bulge')
            ),
            Action(text='Add Gas',
                   func=self.add_type,
                   args=('gas')
            ),
            Action(text='Add Spiral',
                   func=self.add_type,
                   args=('spiral')
            ),
            Action(text='Add Remove Star',
                   func=self.add_type,
                   args=('remove_star')
            )
        )
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
