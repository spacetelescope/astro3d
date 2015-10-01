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
from .items import (Regions, Textures, Clusters, Stars)


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
        self.regions = Regions()
        self.textures = Textures()
        self.cluster_catalogs = Clusters()
        self.stars_catalogs = Stars()

        root = self.invisibleRootItem()
        root.appendRow(self.regions)
        root.appendRow(self.cluster_catalogs)
        root.appendRow(self.stars_catalogs)
        root.appendRow(self.textures)

        self.stages = AttrDict({
            'intensity': True,
            'textures': False,
            'spiral_galaxy': False,
            'double_sided': False
        })

        # Signals
        self.dataChanged.connect(self._update)


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

    def read_regionpathlist(self, pathlist):
        """Read a list of mask files"""
        for path in pathlist:
            region = RegionMask.from_fits(path)
            id = basename(path)
            self.regions.add(region=region, id=id)

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

        for region in self.regions:
            m.add_mask(region)

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
        return triset

    def _update(self, index_ul, index_br):
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
