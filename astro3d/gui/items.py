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
