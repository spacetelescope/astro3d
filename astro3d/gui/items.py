"""Model Items"""
from __future__ import absolute_import, print_function

from ..external.qt.QtGui import QStandardItem


__all__ = ['LayerItem']


class LayerItem(QStandardItem):
    """Layers"""

    def __init__(self, *args, **kwargs):
        super(LayerItem, self).__init__(*args, **kwargs)
        self._value = None
        self.setCheckable(True)
        self.setCheckState(True)
        self._currentrow = None

    def __iter__(self):
        self._currentrow = None
        return self

    def next(self):
        self._currentrow = self._currentrow + 1 \
                           if self._currentrow is not None \
                           else 0
        item = self.child(self._currentrow, 2)
        if item is None:
            raise StopIteration
        else:
            return item

    @property
    def value(self):
        """Value of the item"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @classmethod
    def empty(cls):
        result = cls('')
        result.setCheckable(False)
        result.setEnabled(False)
        return result


class Regions(LayerItem):
    """Regions container"""

    def __init__(self, *args, **kwargs):
        super(Regions, self).__init__(*args, **kwargs)
        self.setText('Regions')

    def next(self):
        while True:
            item = super(Regions, self).next()
            if item.checkState():
                return item.value

    def add(self, region, id):
        """Add a new region"""
        type_item = LayerItem(region.mask_type)
        region_item = LayerItem(id)
        region_item.value = region
        self.appendRow([
            LayerItem.empty(),
            type_item,
            region_item
        ])


class Textures(LayerItem):
    """Textures container"""
    def __init__(self, *args, **kwargs):
        super(Textures, self).__init__(*args, **kwargs)
        self.setText('Textures')


class Clusters(LayerItem):
    """Cluster container"""
    def __init__(self, *args, **kwargs):
        super(Clusters, self).__init__(*args, **kwargs)
        self.setText('Clusters')


class Stars(LayerItem):
    """Stars container"""

    def __init__(self, *args, **kwargs):
        super(Stars, self).__init__(*args, **kwargs)
        self.setText('Stars')
