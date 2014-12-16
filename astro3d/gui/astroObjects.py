"""Data objects for GUI."""
from __future__ import division, print_function

# STDLIB
import os
from collections import defaultdict
from copy import deepcopy

# Anaconda
import numpy as np
from astropy import log

# LOCAL
from ..utils.imageprep import combine_masks, make_model
from ..utils.meshcreator import to_mesh
from ..utils.imageutils import resize_image, split_image


class _File(object):
    """A class that stores all the information necessary to
    construct the STL file from the original Numpy array.

    .. note:: NOT USED.

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

    @property
    def scaled_height(self):
        """Height of data that is scaled for processing (not display)."""
        return self.data.shape[0]

    @property
    def scaled_width(self):
        """Width of data that is scaled for processing (not display)."""
        return self.data.shape[1]

    @property
    def original_height(self):
        """Height of data originally read from file."""
        return self._orig_shape[0]

    @property
    def original_width(self):
        """Width of data originally read from file."""
        return self._orig_shape[1]

    @property
    def displayed_height(self):
        """Height of displayed data."""
        return self.image.height()

    @property
    def displayed_width(self):
        """Width of displayed data."""
        return self.image.width()

    def scale(self):
        """Return the ratio by which the image has been scaled."""
        return self.displayed_height / self.scaled_height

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

        Scale the masks of all regions from the display
        image to the dimension of actual data.
        For dots and lines, their respective regions are
        combined to a single mask.

        Returns
        -------
        scaled_masks : dict
            A dictionary that maps each texture type to a list of
            corresponding boolean masks.

        """
        scaled_masks = defaultdict(list)

        for key, reglist in self.regions.iteritems():
            masklist = [reg.scaled_mask(self) for reg in reglist]

            if key not in self._smooth_keys:
                scaled_masks[key] = [combine_masks(masklist)]
            else:  # To be smoothed
                scaled_masks[key] = masklist

        return scaled_masks

    def save_regions(self, prefix):
        """Save ``regions`` to files using :meth:`Region.save`.

        Coordinates are transformed to match original image.
        One output file per region, each named
        ``<prefixpath>/<type>/<prefixname>_<n>_<description>.npz``.

        Parameters
        ----------
        prefix : str
            Prefix of output files.

        """
        prefixpath, prefixname = os.path.split(prefix)
        for key, reglist in self.regions.iteritems():
            rpath = os.path.join(prefixpath, '_'.join(['region', key]))
            if not os.path.exists(rpath):
                os.mkdir(rpath)
            for i, reg in enumerate(reglist, 1):
                rname = os.path.join(rpath, '_'.join(
                    map(str, [prefixname, i, reg.description])) + '.npz')
                reg.save(rname, _file=self)

    def save_peaks(self, prefix):
        """Save ``peaks`` to text files.

        Coordinates already match original image.
        One output file per table, each named ``<prefix>_<type>.txt``.

        """
        scale = self.original_height / self.scaled_height

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

        #. Scale regions and their boolean masks.
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

    region : array_like
        Boolean mask of the region.

    Attributes
    ----------
    name, region
        Same as inputs.

    description
        Metadata to describe the region. This is for informational purpose only.

    visible : bool
        Visibility (shown/hidden) of the region.

    """
    def __init__(self, name, region):
        super(Region, self).__init__()
        self.name = name
        self.description = name
        self.region = region
        self.visible = False

    def scaled_mask(self, shape):
        """Return mask of the region scaled to given shape.

        Parameters
        ----------
        shape : tuple
            New shape of the mask.

        Returns
        -------
        mask : array_like
            Boolean mask.

        """
        return resize_image(self.region, shape[0], width=shape[1])

    def save(self, filename, image_shape=None):
        """Save the region using :func:`numpy.savez`.

        Parameters
        ----------
        filename : str
            Output filename.

        image_shape : tuple
            This is used to store region in coordinates
            of unscaled image.

        """
        if image_shape is not None:
            mask = resize_image(
                self.region, image_shape[0], width=image_shape[1])
        else:
            mask = self.region

        np.savez(filename, data=mask, name=self.name, descrip=self.description)
        log.info('{0} saved'.format(filename))

    @classmethod
    def fromfile(cls, filename, image_shape=None):
        """Read a region from file generated by :meth:`save`.

        Parameters
        ----------
        filename : str
            Input filename.

        image_shape : tuple
            This is used to convert native coordinates to
            match scaled image for display.

        Returns
        -------
        newreg : `Region`

        """
        dat = np.load(filename)
        orig_mask = dat['data']

        if image_shape is not None:
            mask = resize_image(orig_mask, image_shape[0], width=image_shape[1])
        else:
            mask = orig_mask

        newreg = cls(dat['name'].tostring(), mask)
        newreg.description = dat['descrip'].tostring()
        return newreg


def _regions_old2new(oldfiles, image_shape):
    """Convert old-style regions that store polygon data to new-style
    that store boolean masks.

    This is for interim use only when existing test data already have
    old-style region files saved.

    Parameters
    ----------
    oldfiles : list
        List of ``.reg`` files to convert.

    image_shape : tuple
        Shape of the corresponding image.

    """
    from astropy.io import ascii
    from matplotlib.path import Path

    y, x = np.indices(image_shape)
    y, x = y.flatten(), x.flatten()
    points = np.vstack((x, y)).T

    for oldfile in oldfiles:
        prefix = oldfile.split('.')[0]
        old_points = []

        # Read from old file
        with open(oldfile) as f:
            header = f.readline().strip()
            s = header.split(',')
            name = s[0]
            if len(s) > 1:
                description = s[1]
            else:
                description = name
            coords = ascii.read(f, data_start=1, names=['x', 'y'])
            in_x = coords['x'].astype(np.float32)
            in_y = coords['y'].astype(np.float32)
            for xx, yy in zip(in_x, in_y):
                old_points.append((xx, yy))

        polygon = Path(old_points)
        mask = polygon.contains_points(points).reshape(image_shape)

        newregion = Region(name, mask)
        newregion.description = description
        newregion.save(prefix)
