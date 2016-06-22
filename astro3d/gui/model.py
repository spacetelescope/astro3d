"""Data Model"""

from __future__ import absolute_import, print_function

from itertools import count
from os.path import basename

from attrdict import AttrDict

from numpy import concatenate

from ..external.qt import (QtCore, QtGui)
from ..core.model3d import (Model3D, read_stellar_table)
from ..core.region_mask import RegionMask
from ..core.meshes import (make_triangles, reflect_triangles)
from ..util.logger import make_logger
from . import signaldb
from .qt4.items import (Regions, Textures, Clusters, Stars)

# Shortcuts
QStandardItemModel = QtGui.QStandardItemModel
Qt = QtCore.Qt


__all__ = ['Model']


class Model(QStandardItemModel):
    """Data model"""

    # Sequence for unique id creation
    _sequence = count(1)

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
            'textures': True,
            'spiral_galaxy': True,
            'double_sided': True
        })

        # Signals related to item modification
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

    def read_image(self, pathname):
        """Read image from pathname"""
        try:
            m = Model3D.from_rgb(pathname)
        except:
            m = Model3D.from_fits(pathname)
        self.image = m.data_original

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
        stars = read_stellar_table(pathname, 'stars')
        self.stars_catalogs.add(stars, id)

    def read_cluster_catalog(self, pathname):
        """Read in a star cluster catalog"""
        id = basename(pathname)
        cluster = read_stellar_table(pathname, 'star_clusters')
        self.cluster_catalogs.add(cluster, id)

    def process(self):
        """Create the 3D model.

        Returns
        -------
        (triset, model3d):
            A tuple of the 3D mesh and the model3d it was based on.
        """
        self.logger.debug('Starting processing...')

        model3d = self.create_model3d()
        model3d.make(
            intensity=self.stages.intensity,
            textures=self.stages.textures,
            double_sided=self.stages.double_sided,
            spiral_galaxy=self.stages.spiral_galaxy
        )

        triset = make_triangles(model3d.data)
        if self.stages.double_sided:
            triset = concatenate((triset, reflect_triangles(triset)))
        self.triset = triset
        return (triset, model3d)

    def create_model3d(self, exclude_regions=None):
        """Set the Model3d parameters.
        Create and add all information to the astro3d model

        Parameters
        ----------
        exclude_regions: (str, )
            List of region types to exclude from the model.

        Returns
        -------
        model3d: An Model3D
        """

        # Really need an image.
        if self.image is None:
            raise(RuntimeError, 'Cannot created Model3D. No image defined')
        model3d = Model3D(self.image)

        if exclude_regions is None:
            exclude_regions = []

        for region in self.regions.regions:
            if region.mask_type not in exclude_regions:
                try:
                    model3d.add_mask(region)
                except AttributeError:
                    """Not a RegionMask, ignore"""
                    pass

        for catalog in self.cluster_catalogs.available():
            model3d.add_stellar_table(catalog.value, 'star_clusters')

        for catalog in self.stars_catalogs.available():
            model3d.add_stellar_table(catalog.value, 'stars')

        return model3d

    def create_gas_spiral_masks(
            self,
            smooth_size=11,
            gas_percentile=55.,
            spiral_percentile=75.
    ):
        """Create the gas and spiral masks

        Parameters
        ----------
        smooth_size : float or tuple, optional
            The shape of smoothing filter window.  If ``size`` is an
            `int`, then then ``size`` will be used for both dimensions.

        gas_percentile : float, optional
            The percentile of pixel values in the weighted data above
            which (and below ``spiral_percentile``) to assign to the
            "gas" mask.  ``gas_percentile`` must be lower than
            ``spiral_percentile``.

        spiral_percentile : float, optional
            The percentile of pixel values in the weighted data above
            which to assign to the "spiral arms" mask.
        """
        model3d = self.create_model3d(
            exclude_regions=['gas', 'spiral']
        )
        new_regions = model3d.make_spiral_galaxy_masks(
            smooth_size=smooth_size,
            gas_percentile=gas_percentile,
            spiral_percentile=spiral_percentile
        )
        id_count = str(next(self._sequence))
        for region in new_regions:
            id = 'auto' + region.mask_type + '@' + id_count
            self.regions.add_mask(mask=region, id=id)

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
