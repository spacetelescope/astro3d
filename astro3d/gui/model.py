"""Data Model"""

from __future__ import absolute_import, print_function

from os.path import basename

from attrdict import AttrDict

from numpy import concatenate

from ..external.qt.QtGui import QStandardItemModel
from ..external.qt.QtCore import Qt
from ..core.model3d import Model3D
from ..core.region_mask import RegionMask
from ..core.meshes import (make_triangles, reflect_triangles)
from ..util.logger import make_logger
from . import signaldb
from .qt4.items import (Regions, Textures, Clusters, Stars)


__all__ = ['Model']


class Model(QStandardItemModel):
    """Data model"""

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Layer Manager')
        self.logger = logger

        super(Model, self).__init__(*args, **kwargs)

        # Setup the basic structure
        self.image = None
        self.regions = Regions(logger=self.logger)
        self.textures = Textures()
        self.cluster_catalogs = Clusters()
        self.stars_catalogs = Stars()

        root = self.invisibleRootItem()
        root.appendRow(self.regions)
        root.appendRow(self.cluster_catalogs)
        root.appendRow(self.stars_catalogs)
        root.appendRow(self.textures)
        self._root = root

        self.stages = AttrDict({
            'intensity': True,
            'textures': False,
            'spiral_galaxy': False,
            'double_sided': False
        })

        # Signals
        self.itemChanged.connect(self._update)

        self.columnsInserted.connect(signaldb.ModelUpdate)
        self.columnsMoved.connect(signaldb.ModelUpdate)
        self.columnsRemoved.connect(signaldb.ModelUpdate)
        self.rowsInserted.connect(signaldb.ModelUpdate)
        self.rowsMoved.connect(signaldb.ModelUpdate)
        self.rowsRemoved.connect(signaldb.ModelUpdate)

    def __iter__(self):
        self._currentrow = None
        return self

    def next(self):
        self._currentrow = self._currentrow + 1 \
                           if self._currentrow is not None \
                           else 0
        child = self._root.child(self._currentrow)
        if child is None:
            raise StopIteration
        return child

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        """Set the image

        Parameters
        ----------
        image: 2D numpy array
               The image data.
        """
        self._image = image

    def read_maskpathlist(self, pathlist):
        """Read a list of mask files"""
        signaldb.ModelUpdate.set_enabled(False, push=True)
        try:
            for path in pathlist:
                mask = RegionMask.from_fits(path)
                id = basename(path)
                self.regions.add_mask(mask=mask, id=id)
        finally:
            signaldb.ModelUpdate.reset_enabled()

    def read_star_catalog(self, pathname):
        """Read in a star catalog"""
        id = basename(pathname)
        self.stars_catalogs.add(pathname, id)

    def read_cluster_catalog(self, pathname):
        """Read in a star cluster catalog"""
        id = basename(pathname)
        self.cluster_catalogs.add(pathname, id)

    def process(self):
        """Create the 3D model."""
        self.logger.debug('Starting processing...')

        # Setup steps in the thread. Between each step,
        # check to see if stopped.
        if self.image is None:
            return
        m = Model3D(self.image)

        for region in self.regions.regions:
            try:
                m.add_mask(region)
            except AttributeError:
                """Not a RegionMask, ignore"""
                pass

        for (catalog, id) in self.cluster_catalogs:
            m.read_star_clusters(catalog)

        for (catalog, id) in self.stars_catalogs:
            m.read_stars(catalog)

        m.has_textures = self.stages.textures
        m.has_intensity = self.stages.intensity
        m.spiral_galaxy = self.stages.spiral_galaxy
        m.double_sided = self.stages.double_sided

        m.make()

        triset = make_triangles(m.data)
        if m.double_sided:
            triset = concatenate((triset, reflect_triangles(triset)))
        self.triset = triset
        return (triset, m)

    def save_all(self, prefix):
        """Save all info to the prefix"""
        try:
            model3d = self.model3d
        except AttributeError:
            return
        model3d.write_all_masks(prefix)
        model3d.write_all_stellar_tables(prefix)
        model3d.write_stl(prefix)

    def _update_from_index(self, index_ul, index_br):
        """Update model due to an item change

        Slot for the dataChanged signal

        Parameters
        ----------
        index_ul, index_br: Qt::QModelIndex
            The Upper-Left (ul) and Bottom-Right (br) indexes
            of the model's table.
        """
        if not index_ul.isValid():
            return
        item = self.itemFromIndex(index_ul)
        self.logger.debug('item="{}"'.format(item.text()))
        item.fix_family()

    def _update(self, item):
        """Update model due to an item change

        Slot for the dataChanged signal

        Parameters
        ----------
        item: Qt::QStandardItem
            The item that has changed.
        """
        self.logger.debug('item="{}"'.format(item.text()))
        self.itemChanged.disconnect(self._update)
        item.fix_family()
        self.itemChanged.connect(self._update)
        signaldb.ModelUpdate()
