"""Data objects for GUI."""
from __future__ import division, print_function

# STDLIB
from collections import defaultdict
from copy import deepcopy

# Anaconda
import numpy as np
from astropy import log
from astropy.io import ascii
from matplotlib.path import Path
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from scipy import ndimage

# LOCAL
from ..utils.imageprep import combine_masks, make_model
from ..utils.meshcreator import to_mesh
from ..utils.imageutils import split_image


class File(object):
    """A class that stores all the information necessary to
    construct the STL file from the original Numpy array.

    Parameters
    ----------
    data : ndarray
        Original data from FITS or JPEG file.

    image : QPixmap
        Scaled image for display purposes.

    Attributes
    ----------
    data, image
        Same as inputs.

    regions : dict
        A dictionary that maps each texture to a list of `Region`.

    peaks : dict
        A dictionary that maps each texture to a `astropy.table.Table`.

    """
    _smooth_keys = ['smooth', 'remove_star']

    def __init__(self, data, image):
        super(File, self).__init__()
        self.data = data
        self.image = image
        self.regions = defaultdict(list)
        self.peaks = {}
        self._orig_shape = self.data.shape

    def scale(self):
        """Return the ratio by which the image has been scaled."""
        return self.image.height() / self.data.shape[0]

    def orig_scale(self):
        """Return the ratio between display and original size."""
        return self.image.height() / self._orig_shape[0]

    def texture_names(self):
        """Return region texture names, except for the one used
        for smoothing.

        .. note::

            This is targeted at textures with dots and lines,
            where lines belong in the foreground layer by default,
            hence listed first.

        """
        names = set()

        for key in self.regions:
            if key in self._smooth_keys:
                continue
            if len(self.regions[key]) < 1:
                continue
            names.add(key)

        return sorted(names, reverse=True)

    def scaleMasks(self):
        """Create masks scaled to actual data.

        Scale the coordinates of all regions from the display
        image to the corresponding locations on the actual data.
        Then, create masks for them. For dots and lines, their
        respective regions are combined to a single mask.

        Returns
        -------
        scaled_masks : dict
            A dictionary that maps each texture type to a list of
            corresponding boolean masks.

        """
        scaled_masks = defaultdict(list)

        for key, reglist in self.regions.iteritems():
            masklist = [reg.scaledRegion(self).to_mask(self.data)
                        for reg in reglist]

            if key not in self._smooth_keys:
                scaled_masks[key] = [combine_masks(masklist)]
            else:  # To be smoothed
                scaled_masks[key] = masklist

        return scaled_masks

    def save_regions(self, prefix):
        """Save ``regions`` to text files.

        Coordinates are transformed to match original image.
        One output file per region, each name ``<prefix>_<type>_<n>.reg``.

        Parameters
        ----------
        prefix : str
            Prefix of output files.

        """
        for key, reglist in self.regions.iteritems():
            i = 1
            for reg in reglist:
                rname = '{0}_{1}_{2}.reg'.format(prefix, key, i)
                i += 1
                reg.save(rname, _file=self)
                log.info('{0} saved'.format(rname))

    def save_peaks(self, prefix):
        """Save ``peaks`` to text files.

        Coordinates already match original image.
        One output file per table, each named ``<prefix>_<type>.txt``.

        """
        scale = self._orig_shape[0] / self.data.shape[0]

        for key, tab in self.peaks.iteritems():
            if len(tab) < 1:
                continue

            out_tab = deepcopy(tab)
            out_tab['xcen'] *= scale
            out_tab['ycen'] *= scale

            tname = '{0}_{1}.txt'.format(prefix, key)
            out_tab.write(tname, format='ascii')
            log.info('{0} saved'.format(tname))

    def make_3d(self, fname, height=150.0, depth=10, clus_r_fac_add=15,
                clus_r_fac_mul=1, star_r_fac_add=15, star_r_fac_mul=1,
                layer_order=['lines', 'dots', 'dots_small'], double=False,
                _ascii=False, has_texture=True, has_intensity=True,
                is_spiralgal=False, split_halves=True):
        """Generate STL file.

        #. Scale regions.
        #. Create boolean masks for the regions.
        #. Use :func:`~astro3d.utils.imageprep.make_model` to perform
           transformations on the array and obtains a new array ready
           for STL creator.
        #. Split array into two halves (optional).
        #. Use :func:`~astro3d.utils.meshcreator.to_mesh` to make STL file(s).

        Parameters
        ----------
        fname : str or QString
            Filename prefix. The ``.stl`` extension is automatically
            added. Also see ``split_halves``.

        height : float
            Height of the model.

        depth : int
            Depth of back plate.

        clus_r_fac_add, clus_r_fac_mul, star_r_fac_add, star_r_fac_mul : float
            Crater radius scaling factors for star clusters and stars,
            respectively.

        layer_order : list
            Order of texture layers (dots, lines) to apply.
            Top/foreground layer overwrites the bottom/background. This
            is only used if ``is_spiralgal=False`` and ``has_texture=True``.

        double : bool
            Double- or single-sided.

        _ascii : bool
            ASCII or binary format.

        has_texture : bool
            Apply textures.

        has_intensity : bool
            Generate intensity map.

        is_spiralgal : bool
            Special processing for a single spiral galaxy.

        split_halves : bool
            If `True`, image is split into two halves and the files
            will have ``1.stl`` or ``2.stl`` added to ``fname``,
            respectively.

        """
        image = deepcopy(np.flipud(self.data))
        regions = self.scaleMasks()

        model = make_model(
            image, region_masks=regions, peaks=self.peaks, height=height,
            base_thickness=depth, clus_r_fac_add=clus_r_fac_add,
            clus_r_fac_mul=clus_r_fac_mul, star_r_fac_add=star_r_fac_add,
            star_r_fac_mul=star_r_fac_mul, layer_order=layer_order,
            double=double, has_texture=has_texture, has_intensity=has_intensity,
            is_spiralgal=is_spiralgal)

        # Input filename might be QString.
        # Remove any .stl suffix because it is added by to_mesh()
        fname = str(fname)
        if fname.lower().endswith('.stl'):
            fname = fname[:-4]

        # Depth is set to 1 here because it is accounted for in make_model()
        if split_halves:
            model1, model2 = split_image(model, axis='horizontal')
            to_mesh(model1, fname + '_1', 1, double, _ascii)
            to_mesh(model2, fname + '_2', 1, double, _ascii)
        else:
            to_mesh(model, fname, 1, double, _ascii)


class Region(object):
    """This class defines a selected region in GUI.

    Parameters
    ----------
    name : str
        Region type.

    region : QPolygonF
        Shape of the region.

    Attributes
    ----------
    name, region
        Same as inputs.

    visible : bool
        Visibility (shown/hidden) of the region.

    """
    def __init__(self, name, region):
        super(Region, self).__init__()
        self.name = name
        self.region = region
        self.visible = False

    def points(self, scale=1):
        """Return a list of all points that are part of the polygon.

        Parameters
        ----------
        scale : float
            Not used.

        Returns
        -------
        points : list of QPointF

        """
        i = 0
        p = None
        points = []
        if self.region is not None:
            while p != self.region.last():
                p = self.region.at(i)
                points.append(p)
                i += 1
        return points

    def contains(self, x, y, scaled=1):
        """Check whether a certain point is inside the region.

        .. note::

            Maybe this is no longer needed because its capability
            is replaced by :meth:`to_mask`.

        Parameters
        ----------
        x, y : float
            Location of the point.

        scaled : float
            Scale the location to match region.

        Returns
        -------
        is_within : bool

        """
        x *= scaled
        y *= scaled
        p = QPointF(x, y)
        return QGraphicsPolygonItem(self.region).contains(p)

    def get_bounding_box(self):
        """Return the min/max X and Y coordinates of all
        points in polygon.

        .. note::

            Maybe this is no longer needed.

        Returns
        -------
        xmin, ymin, xmax, ymax : int

        """
        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')
        for p in self.points():
            if p.x() < xmin:
                xmin = p.x()
            elif p.x() > xmax:
                xmax = p.x()
            if p.y() < ymin:
                ymin = p.y()
            elif p.y() > ymax:
                ymax = p.y()
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def scaledRegion(self, _file):
        """Return a new region scaled to match the `File` data.

        Parameters
        ----------
        _file : `File`

        Returns
        -------
        region : `Region`

        """
        scale = 1.0 / _file.scale()
        region = QPolygonF()
        for point in self.points():
            p = QPointF(point.x() * scale, point.y() * scale)
            region << p
        return Region(self.name, region)

    def to_mask(self, image, interpolate=True, fil_size=3):
        """Uses `matplotlib.path.Path` to generate a
        Numpy boolean array, which can then be used as
        a mask for the region.

        Parameters
        ----------
        image : ndarray
            Image to apply mask to.

        interpolate : `True`, number, or tuple
            For filter used in mask generation.

        fil_size : int
            Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.

        Returns
        -------
        mask : ndarray
            Boolean mask for the region.

        """
        y, x = np.indices(image.shape)
        y, x = y.flatten(), x.flatten()
        points = np.vstack((x, y)).T
        polygon = Path([(p.x(), p.y()) for p in self.points()])
        mask = polygon.contains_points(points).reshape(image.shape)

        if interpolate:
            if interpolate == True:  # Magic?
                interpolate = (np.percentile(image[mask], 50),
                               np.percentile(image[mask], 75))
            elif np.isscalar(interpolate):
                interpolate = (np.percentile(image[mask], 0),
                               np.percentile(image[mask], interpolate))
            else:
                interpolate = (np.percentile(image[mask], interpolate[0]),
                               np.percentile(image[mask], interpolate[1]))

            nmin, nmax = interpolate
            filtered = np.zeros(mask.shape)
            filtered[mask] = 1

            # Cannot have data type to be numpy.int64
            # https://github.com/scipy/scipy/issues/4106
            radius = int(min(axis.ptp() for axis in np.where(mask)))

            filtered = ndimage.filters.maximum_filter(
                filtered, min(radius, image.shape[0] / 33))  # Magic?
            filtered = image * filtered
            mask = mask | ((filtered > nmin) & (filtered < nmax))
            maxfilt = ndimage.filters.maximum_filter(mask.astype(int), fil_size)
            mask = maxfilt != 0

        return mask

    def save(self, filename, _file=None):
        """Save the region to a text file.

        Parameters
        ----------
        filename : str
            Output filename.

        _file : `File`
            This is used to store region in coordinates
            of unscaled image.

        """
        if _file is not None:
            scale = 1.0 / _file.orig_scale()
        else:
            scale = 1.0

        with open(filename, 'w') as f:
            f.write(unicode('{0}\n'.format(self.name)))
            for p in self.points():
                f.write(unicode('{0} {1}\n'.format(p.x() * scale,
                                                   p.y() * scale)))

    @classmethod
    def fromfile(cls, filename, _file=None):
        """Read a region from file generated by :meth:`save`.

        Parameters
        ----------
        filename : str
            Input filename.

        _file : `File`
            This is used to convert native coordinates to
            match scaled image.

        Returns
        -------
        region : `Region`

        """
        region = QPolygonF()
        if _file is not None:
            scale = _file.orig_scale()
        else:
            scale = 1.0
        name = ''
        with open(filename) as f:
            name = f.readline().strip()
            coords = ascii.read(f, data_start=1, names=['x', 'y'])
            scaled_x = coords['x'].astype(np.float32) * scale
            scaled_y = coords['y'].astype(np.float32) * scale
            for x, y in zip(scaled_x, scaled_y):
                region << QPointF(x, y)
        return cls(name, region)


class MergedRegion(Region):
    """Class to represent a merging of two or more regions.

    .. note::

        This does not currently work, and not supported by GUI.

    """
    def __init__(self, list_of_regions):
        super(MergedRegion, self).__init__()  # Will raise error, no arguments
        self.name = list_of_regions[0].name
        self.region = [region.region for region in list_of_regions]
        self.originals = list_of_regions
        self.visible = False

    def contains(self, x, y, scaled=False):
        return any(map(lambda reg: reg.contains(x, y, scaled), self.region))

    def split(self):
        return self.originals
