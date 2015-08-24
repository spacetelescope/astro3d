"""
This module provides tools create a 3D model from an astronomical image.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
from copy import deepcopy
import warnings
from functools import partial

from scipy import ndimage
import numpy as np
from PIL import Image

from astropy import log
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
import photutils

from . import textures
from . import image_utils
from .meshes import write_mesh
from .region_mask import RegionMask


__all__ = ['Model3D']


class Model3D(object):
    """
    Class to create a 3D model from an astronomical image.

    Examples
    --------
    >>> # initialize the model
    >>> model = Model3D(data)
    >>> model = Model3D.from_fits('myimage.fits')    # or from FITS file

    >>> # define the type of 3D model
    >>> model.has_textures = True
    >>> model.has_intensity = True
    >>> model.double_sided = True
    >>> model.spiral_galaxy = True

    >>> # read the region/texture masks
    >>> model.read_mask('mask_file.fits')      # can read one by one
    >>> model.read_all_masks('*.fits')         # or several at once

    >>> # read the "stars" table
    >>> model.read_stellar_table('object_stars.txt', 'stars')
    >>> model.read_stars('object_stars.txt')     # same as above

    >>> # read the "star clusters" table
    >>> model.read_stellar_table('object_star_clusters.txt', 'star_clusters')
    >>> model.read_star_clusters('object_star_clusters.txt')   # same as above

    >>> # make the model
    >>> model.make()

    >>> # write the model to a STL file
    >>> filename_prefix = 'myobject'
    >>> model.write_stl(prefix, split_model=True)

    >>> # write the texture and region masks to FITS files
    >>> model.write_all_masks(filename_prefix)

    >>> # write stellar tables (separately or all at once)
    >>> model.write_stellar_table(filename_prefix, stellar_type='stars')
    >>> model.write_stellar_table(filename_prefix,
    ...                           stellar_type='star_clusters')
    >>> model.write_all_stellar_tables(filename_prefix)    # all at once
    """

    def __init__(self, data, resize_xsize=1000):
        """
        Parameters
        ----------
        data : array-like
            The input 2D array from which to create a 3D model.

        resize_xsize : int, optional
            The size of the x axis of the resized image.

        Notes
        -----
        A ``height`` of 250 corresponds to a physical height of 68.6 mm
        on the MakerBot 2 printer.  This assumes 0.14 mm per pixel and a
        uniform maximum scaling factor of 1.96 (which assumes
        ``resize_xsize = 1000``).

        The ``height`` is the height of the intensity map *before* the
        textures, including the spiral galaxy central cusp, are applied.

        A ``base_height`` of 10 corresponds to 2.74 mm.
        """

        self.data_original = np.asanyarray(data)
        self.resize_scale_factor = float(resize_xsize /
                                         self.data_original.shape[1])
        self.data_original_resized = image_utils.resize_image(
            image_utils.remove_nonfinite(self.data_original),
            self.resize_scale_factor)

        self._model_complete = False
        self._has_textures = True
        self._has_intensity = True
        self._double_sided = False
        self._spiral_galaxy = False
        self.height = 250.         # total model height of *intensity* model
        self.base_height = 10.     # total base height

        self.texture_order = ['small_dots', 'dots', 'lines']
        self.region_mask_types = ['smooth', 'remove_star']
        self.allowed_stellar_types = ['stars', 'star_clusters']

        self.translate_texture = {}
        self.translate_texture['small_dots'] = ['gas']
        self.translate_texture['dots'] = ['spiral']
        self.translate_texture['lines'] = ['bulge', 'disk']
        self.translate_texture['smooth'] = []
        self.translate_texture['remove_star'] = []

        self.texture_masks_original = defaultdict(list)
        self.region_masks_original = defaultdict(list)
        self.stellar_tables_original = {}

        self.textures = {}
        self.textures['small_dots'] = partial(
            textures.dots_texture_map, profile='spherical', diameter=9.0,
            height=4.0, grid_func=textures.hexagonal_grid, grid_spacing=5.0)
        self.textures['dots'] = partial(
            textures.dots_texture_map, profile='spherical', diameter=9.0,
            height=8.0, grid_func=textures.hexagonal_grid, grid_spacing=9.0)
        self.textures['lines'] = partial(
            textures.lines_texture_map, profile='linear', thickness=13,
            height=7.8, spacing=20, orientation=0)

    @property
    def has_textures(self):
        """
        Property to determine if the 3D model has textures.
        `True` or `False`.
        """
        return self._has_textures

    @has_textures.setter
    def has_textures(self, value):
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean.')
        if not value and not self.has_intensity:
            raise ValueError('3D Model must have textures and/or intensity.')
        self._has_textures = value

    @property
    def has_intensity(self):
        """
        Property to determine if the 3D model has intensities.
        """
        return self._has_intensity

    @has_intensity.setter
    def has_intensity(self, value):
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean.')
        if not value and not self.has_textures:
            raise ValueError('3D Model must have textures and/or intensity.')
        self._has_intensity = value

    @property
    def double_sided(self):
        """
        Property to determine if the 3D model is double sided (simple
        reflection).
        """
        return self._double_sided

    @double_sided.setter
    def double_sided(self, value):
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean.')
        self._double_sided = value

    @property
    def spiral_galaxy(self):
        """
        Property to determine if the 3D model is a spiral galaxy, which
        uses special processing.
        """
        return self._spiral_galaxy

    @spiral_galaxy.setter
    def spiral_galaxy(self, value):
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean.')
        self._spiral_galaxy = value

    @classmethod
    def from_fits(cls, filename):
        """
        Create a `Model3D` instance from a FITS file.

        Parameters
        ----------
        filename : str
            The name of the FITS file.
        """

        data = fits.getdata(filename)
        if data is None:
            raise ValueError('data not found in the FITS file')

        if data.ndim not in [2, 3]:
            raise ValueError('data is not a 2D image or a 3D RGB cube')

        if data.ndim == 3:    # RGB cube
            # TODO: interpolate over non-finite values?
            # TODO: improve RGB to grayscale conversion
            data[~np.isfinite(data)] = 0.
            data = data.sum(axis=0)

        return cls(data)

    @classmethod
    def from_rgb(cls, filename):
        """
        Create a `Model3D` instance from a RGB file (e.g. JPG, PNG,
        TIFF).

        Parameters
        ----------
        filename : str
            The name of the RGB file.
        """

        data = np.array(Image.open(filename).convert('L'),
                        dtype=np.float32)[::-1]
        return cls(data)

    def _translate_mask_type(self, mask_type):
        """
        Translate mask type into the texture or region type.

        The valid texture types are defined in ``.texture_order``.  The
        valid region types are defined in ``.region_mask_types``.

        Parameters
        ----------
        mask_type : str
            The mask type

        Returns
        -------
        texture_type: str
            The mask type translated to the texture type (e.g.
            'small_dots', 'dots', or 'lines') or region type (e.g.
            'smooth').
        """

        if mask_type in self.translate_texture:
            return mask_type
        else:
            tx_type = None
            for tx_type, mask_types in self.translate_texture.iteritems():
                if mask_type in mask_types:
                    return tx_type
            if tx_type is None:
                warnings.warn('"{0}" is not a valid mask '
                              'type.'.format(mask_type), AstropyUserWarning)
            return

    def read_mask(self, filename):
        """
        Read a region mask from a FITS file.

        The mask is read into a `RegionMask` object and then stored in
        the `region_masks_original` or `texture_masks_original`
        dictionary, keyed by the mask type.

        The mask data must have the same shape as the input ``data``.

        Parameters
        ----------
        filename : str
            The name of the FITS file.  The mask data must be in the
            primary FITS extension and the header must have a 'MASKTYPE'
            keyword defining the mask type.
        """

        region_mask = RegionMask.from_fits(
            filename, required_shape=self.data_original.shape)
        mask_type = region_mask.mask_type
        mtype = self._translate_mask_type(mask_type)
        if mtype in self.region_mask_types:
            self.region_masks_original[mtype].append(region_mask)
        else:
            self.texture_masks_original[mtype].append(region_mask)

        log.info('Mask type "{0}" loaded from "{1}"'.format(mask_type,
                                                            filename))

    def read_all_masks(self, pathname):
        """
        Read all region masks (FITS files) matching the specified
        ``pathname``.

        Parameters
        ----------
        pathname : str
            The pathname pattern of mask FITS files to read.  Wildcards
            are allowed.

        Examples
        --------
        >>> model3d = Model3D()
        >>> model3d.read_all_masks('*.fits')
        >>> model3d.read_all_masks('masks/*.fits')
        """

        import glob
        for filename in glob.iglob(pathname):
            self.read_mask(filename)

    def write_all_masks(self, filename_prefix):
        """
        Write all region masks as FITS files.  The files are saved to
        the current directory.

        The ouput masks will have the same shape as the original input
        image.

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output filenames.  The output filenames
            will be '<filename_prefix>_<mask_type>.fits'.  If there is
            more than one mask for a given mask type then they will be
            numbered with consecutive integers starting at 1:
            '<filename_prefix>_<mask_type>_<num>.fits'.
        """

        for mask_type, masks in self.region_masks_original.iteritems():
            nmasks = len(masks)
            for i, mask in enumerate(masks, 1):
                if nmasks > 1:
                    filename = '{0}_{1}_{2}.fits'.format(filename_prefix,
                                                         mask_type, i)
                else:
                    filename = '{0}_{1}.fits'.format(filename_prefix,
                                                     mask_type)
                mask.write(filename, shape=self.data_original.shape)

    def read_stellar_table(self, filename, stellar_type):
        """
        Read a table of stars or star clusters from a file.

        The table must have ``'xcentroid``, ``'ycentroid'``, and
        ``'flux'`` columns.

        Parameters
        ----------
        filename : str
            The filename containing an `~astropy.Table` in ASCII format.

        stellar_type : {'stars', 'star_clusters'}
            The type of the table.
        """

        table = Table.read(filename, format='ascii')
        table.keep_columns(['xcentroid', 'ycentroid', 'flux'])
        self.stellar_tables_original[stellar_type] = table
        log.info('Read "{0}" table from "{1}"'.format(stellar_type, filename))

    def read_star_clusters(self, filename):
        """Read star clusters table from an ASCII file."""
        self.read_stellar_table(filename, 'star_clusters')

    def read_stars(self, filename):
        """Read stars table from an ASCII file."""
        self.read_stellar_table(filename, 'stars')

    def write_stellar_table(self, filename_prefix, stellar_type):
        """
        Write a table of stars or star clusters to an ASCII file.

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output filenames.  The output filenames
            will be '<filename_prefix>_<stellar_type>.txt'.

        stellar_type : {'stars', 'star_clusters'}
            The type of the table.
        """

        filename = '{0}_{1}.txt'.format(filename_prefix, stellar_type)
        table = self.stellar_tables_original[stellar_type]
        if table is not None:
            table.write(filename, format='ascii')
            log.info('Saved "{0}" table to "{1}"'.format(stellar_type,
                                                         filename))
        else:
            log.info('"{0}" table was empty and not '
                     'saved.'.format(stellar_type))

    def write_all_stellar_tables(self, filename_prefix):
        """
        Write all tables of stars and star clusters to ASCII files.

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output filenames.  The output filenames
            will be '<filename_prefix>_<stellar_type>.txt'.
        """

        for stellar_type in self.allowed_stellar_types:
            self.write_stellar_table(filename_prefix, stellar_type)

    def write_stl(self, filename_prefix, split_model=True,
                  stl_format='binary', clobber=False):
        """
        Write the 3D model to a STL file(s).

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output filenames.  The output filename
            will be '<filename_prefix>.stl'.  If ``split_image=True``,
            then the filename will be '<filename_prefix>_[1|2].stl'.

        split_model : bool, optional
            If `True`, then split the model into two halves, a bottom
            and top part.

        stl_format : {'binary', 'ascii'}, optional
            Format for the output STL file.  The default is 'binary'.
            The binary STL file is harder to debug, but takes up less
            storage space.

        clobber : bool, optional
            Set to `True` to overwrite any existing file(s).
        """

        if not self._model_complete:
            warnings.warn('The model has not been constructed yet. '
                          'Please run the .make() method before saving '
                          'the STL file.', AstropyUserWarning)
            return

        if split_model:
            model1, model2 = image_utils.split_image(self.data, axis=0)
            write_mesh(model1, filename_prefix + '_1',
                       double_sided=self.double_sided, stl_format=stl_format,
                       clobber=clobber)
            write_mesh(model2, filename_prefix + '_2',
                       double_sided=self.double_sided, stl_format=stl_format,
                       clobber=clobber)
        else:
            write_mesh(self.data, filename_prefix,
                       double_sided=self.double_sided, stl_format=stl_format,
                       clobber=clobber)

    def _prepare_masks(self):
        """
        Prepare texture and region masks.

        Texture masks are combined and resized.  Region masks are
        resized, but not combined.
        """

        self.texture_masks = {}
        self.region_masks = {}

        # combine and resize texture_masks
        for mask_type, masks in self.texture_masks_original.iteritems():
            prepared_mask = image_utils.resize_image(
                image_utils.combine_region_masks(masks),
                self.resize_scale_factor)
            self.texture_masks[mask_type] = prepared_mask   # ndarray

        # resize but do not combine region_masks
        for mask_type, masks in self.region_masks_original.iteritems():
            resized_masks = [image_utils.resize_image(
                mask.mask, self.resize_scale_factor) for mask in masks]
            self.region_masks[mask_type] = resized_masks   # list of ndarrays

    @staticmethod
    def _scale_table_positions(table, resize_scale):
        """
        Scale the ``(x, y)`` positions in a stellar table.

        The image resize scale factor is applied to the ``xcentroid``
        and ``ycentroid`` columns.

        Parameters
        ----------
        table : `~astropy.table.Table`
            A table of stellar-like sources.

        resize_scale : float
            The desired scaling factor to apply to the position columns.

        Returns
        -------
        result : `~astropy.table.Table`
            The table with scaled positions.
        """

        result = deepcopy(table)
        result['xcentroid'] *= resize_scale
        result['ycentroid'] *= resize_scale
        return result

    def _scale_stellar_table_positions(self, stellar_tables, resize_scale):
        """
        Scale the ``(x, y)`` positions in the stellar tables.

        The image resize factor is applied to the ``xcentroid`` and
        ``ycentroid`` columns.  The output dictionary of tables is
        stored in ``self.stellar_tables``.

        Parameters
        ----------
        stellar_tables : dict
            Dictionary of stellar tables.

        resize_scale : float
            The desired scaling factor to apply to the position columns.
        """

        self.stellar_tables = deepcopy(stellar_tables)
        for stellar_type, table in self.stellar_tables.iteritems():
            if table is not None:
                tbl = self._scale_table_positions(table, resize_scale)
                self.stellar_tables[stellar_type] = tbl

    def _remove_stars(self):
        """
        Remove stars by patching the regions defined by ``remove_stars``
        region masks.
        """

        if 'remove_star' not in self.region_masks:
            return

        for mask in self.region_masks['remove_star']:
            y, x = mask.nonzero()
            xsize, ysize = x.ptp(), y.ptp()

            # Four shifted masked regions (bottom, top, left, right)
            # immediately adjacent to the masked region
            regions_x = [x, x, x - xsize, x + xsize]
            regions_y = [y - ysize, y + ysize, y, y]

            nearest_regions = []
            warn_msg = []
            for x, y in zip(regions_x, regions_y):
                try:
                    values = self.data[y, x]
                except IndexError as err:
                    # don't include regions outside of the image
                    warn_msg.append('\t{0}'.format(err))
                else:
                    nearest_regions.append(values)

            if len(nearest_regions) == 0:
                warnings.warn('The _remove_stars() method '
                              'failed:\n{0}'.format(
                                  '\n'.join(warn_msg)), AstropyUserWarning)
                continue

            # regions_median = [np.median(region) for region in
            #                   nearest_regions]
            # self.data[mask] = nearest_regions[np.argmax(regions_median)]
            self.data[mask] = np.array(nearest_regions).mean(axis=0)

    def _spiralgalaxy_compress_bulge(self, percentile=0., factor=0.05):
        """
        Compress the image values in the bulge region of a spiral
        galaxy.

        This is needed to compress the large dynamic range of
        intensities in spiral galaxy images.

        A base value is first taken a the ``percentile`` of data values
        within the bulge mask.  Values above this are then compressed by
        ``factor``.

        Parameters
        ----------
        percentile: float in range of [0, 100], optional
            The percentile of pixel values within the bulge mask to use
            as the base level.

        factor : float, optional
            The scale factor to apply to the region above the base
            level.

        Returns
        -------
        result : float
            The base level above which values are compressed.
        """

        if not self.spiral_galaxy:
            return None

        log.info('Compressing the bulge.')
        texture_type = self._translate_mask_type('bulge')
        bulge_mask = self.texture_masks[texture_type]
        if bulge_mask is not None:
            base_level = np.percentile(self.data[bulge_mask],
                                       percentile)
            compress_mask = self.data > base_level
            new_values = (base_level + (self.data[compress_mask] -
                                        base_level) * factor)
            self.data[compress_mask] = new_values
        else:
            warnings.warn('A "bulge" mask must be input.')
        return base_level

    def _suppress_background(self, percentile=90., factor=0.2,
                             floor_percentile=10.):
        """
        Suppress image values in regions that are not within any texture
        mask.

        This is used to suppress the "background".

        A base value is first taken a the ``percentile`` of data values
        in regions that are not in any texture mask.  Values below this
        are then compressed by ``factor``.  Values below ``floor`` are
        then set to zero.

        Parameters
        ----------
        percentile: float in range of [0, 100], optional
            The percentile of pixel values outside of the masked regions
            to use as the background level.

        factor : float, optional
            The scale factor to apply to the region below the
            background level.

        floor_percentile : float, optional
            The percentile of image values equal to and below which to
            set to zero after the initial background suppression.

        Returns
        -------
        result : float
            The background level below which values are suppressed.
        """

        if not self.texture_masks:
            return None

        log.info('Suppressing the background.')
        texture_masks = [self.texture_masks[i] for i in self.texture_masks]
        mask = image_utils.combine_masks(texture_masks)
        background_level = np.percentile(self.data[~mask], percentile)
        bkgrd_mask = self.data < background_level
        self.data[bkgrd_mask] = self.data[bkgrd_mask] * factor
        floor = np.percentile(self.data, floor_percentile)
        self.data[self.data < floor] = 0.
        return background_level

    def _smooth_image(self, size=11):
        """
        Smooth the image using a 2D median filter.

        Parameters
        ----------
        size : float or tuple, optional
            The shape of filter window.  If ``size`` is an `int`, then
            then ``size`` will be used for both dimensions.
        """

        log.info('Smoothing the image.')
        self.data = ndimage.filters.median_filter(self.data, size=size)

    def _normalize_image(self, max_value=1.0):
        """
        Normalize the image.

        Parameters
        ----------
        max_value : float, optional
            The maximum value of the normalized array.
        """

        log.info('Normalizing the image.')
        self.data = image_utils.normalize_data(self.data, max_value=max_value)

    def _minvalue_to_zero(self, min_value=0.02):
        """Set values below a certain value to zero."""
        self.data[self.data < min_value] = 0.0

    def _crop_data(self, threshold=0.0, resize=True):
        """
        Crop the image, masks, and stellar tables.

        If ``resize=True`, they are then resized to have the same size
        as ``self.data_original_resized`` to ensure a consistent 3D
        scaling of the model in MakerBot.

        Parameters
        ----------
        threshold : float, optional
            The values equal to and below which to crop from the data.

        resize : bool, optional
            Set to `True` to resize the data, masks, and stellar tables
            back to the original size of ``self.data_original_resized``.
        """

        log.info('Cropping the data values equal to or below a threshold of '
                 '"{0}"'.format(threshold))
        slc = image_utils.crop_below_threshold(self.data, threshold=threshold)
        self.data = self.data[slc]

        for mask_type, mask in self.texture_masks.iteritems():
            log.info('Cropping masks')
            self.texture_masks[mask_type] = mask[slc]

        for stellar_type, table in self.stellar_tables.iteritems():
            idx = ((table['xcentroid'] > slc[1].start) &
                   (table['xcentroid'] < slc[1].stop) &
                   (table['ycentroid'] > slc[0].start) &
                   (table['ycentroid'] < slc[0].stop))
            table = table[idx]
            table['xcentroid'] -= slc[1].start
            table['ycentroid'] -= slc[0].start
            self.stellar_tables[stellar_type] = table

        if resize:
            scale_factor = float(self.data_original_resized.shape[1] /
                                 self.data.shape[1])
            self.data = image_utils.resize_image(
                self.data, scale_factor)
            for mask_type, mask in self.texture_masks.iteritems():
                log.info('Resizing masks')
                self.texture_masks[mask_type] = image_utils.resize_image(
                    mask, scale_factor)
            self._scale_stellar_table_positions(self.stellar_tables,
                                                scale_factor)
        return slc

    def _make_model_height(self):
        """
        Scale the image to the final model height prior to adding the
        textures.

        To give consistent texture height (and "feel"), no scaling of
        the image should happen after this step!
        """

        # clip the image at the cusp base_height
        if self.spiral_galaxy:
            base_height = self._apply_spiral_central_cusp(
                base_height_only=True)
            # NOTE: this will also clip values outside of the bulge
            self.data[self.data > base_height] = base_height

        if self.double_sided:
            height = self.height / 2.
        else:
            height = self.height
        self._normalize_image(max_value=height)

    def _add_masked_textures(self):
        """
        Add masked textures (e.g. small dots, dots, lines) to the
        image.

        The masked textures are added in order, specified by
        ``.texture_order``.  Masked areas in subsequent textures will
        override earlier textures for pixels masked in more than one
        texture (i.e. a given pixel has only one texture applied).
        """

        self._texture_layer = np.zeros_like(self.data)
        for texture_type in self.texture_order:
            if texture_type not in self.texture_masks:
                continue

            log.info('Adding "{0}" textures.'.format(texture_type))
            mask = self.texture_masks[texture_type]
            texture_data = self.textures[texture_type](mask)
            self._texture_layer[mask] = texture_data[mask]
        self.data += self._texture_layer

    def _apply_stellar_textures(self, radius_a=10., radius_b=5.):
        """
        Apply stellar textures (stars and star clusters) to the image.

        The radius of the star (used in both `StarTexture` and
        `StarCluster` textures) for each source is linearly scaled by
        the source flux as:

            .. math:: radius = radius_a + (radius_b * flux / max_flux)

        where ``max_flux`` is the maximum ``flux`` value of all the
        input ``sources``.

        Parameters
        ----------
        radius_a : float
            The intercept term in calculating the star radius (see above).

        radius_b : float
            The slope term in calculating the star radius (see above).
        """

        if self.has_intensity:
            base_percentile = 75.
            depth = 5.
        else:
            base_percentile = None
            depth = 10.

        if not self.stellar_tables:
            return

        log.info('Adding stellar-like textures.')
        self._stellar_texture_layer = textures.make_starlike_textures(
            self.data, self.stellar_tables, radius_a=radius_a,
            radius_b=radius_b, depth=depth, base_percentile=base_percentile)
        self.data = textures.apply_textures(self.data,
                                            self._stellar_texture_layer)

    def _find_galaxy_center(self, mask=None):
        """
        Find the position of a galaxy center simply as the location of
        the maximum value in the image.

        If there are multiple pixels with the maximum value, then the
        center will be calculated as the average of those pixel
        positions.

        If a ``mask`` is input, then only those regions will be
        considered.

        Parameters
        ----------
        mask : bool `~numpy.ndarray`, optional
            Boolean mask where to search for the maximum value.
        """

        # use np.where instead of np.argmax in case of multiple
        # occurrences of the maximum value
        if mask is None:
            y, x = np.where(self.data == self.data.max())
        else:
            data = np.ma.array(self.data, mask=~mask)
            y, x = np.where(data == data.max())
        y_center = y.mean()
        x_center = x.mean()
        log.info('Center of galaxy at x={0}, y={1}'.format(x_center, y_center))
        return x_center, y_center

    def _apply_spiral_central_cusp(self, radius=25., depth=8.,
                                   base_height_only=False):
        """
        Add a central cusp for spiral galaxies.

        If ``base_only=True`` then simply return the base height of the
        texture model instead of adding the central cusp.

        Add this texture last, especially after adding the "lines"
        texture for the central bulge.

        Parameters
        ----------
        radius : float, optional
            The circular radius of the star texture.

        depth : float, optional
            The maximum depth of the crater-like bowl of the star texture.

        base_height_only : bool, optional
            If `True`, then simply return the base height of the texture
            model, i.e. do not actually add the central cusp.

        Returns
        -------
        result : float
            The base height of the texture model.
        """

        if self.spiral_galaxy:
            if self.has_intensity:
                base_percentile = 0.
            else:
                base_percentile = None

            texture_type = self._translate_mask_type('bulge')
            bulge_mask = self.texture_masks[texture_type]
            if bulge_mask is not None:
                x, y = self._find_galaxy_center(bulge_mask)

                if base_height_only:
                    base_height = textures.starlike_model_base_height(
                        self.data, 'stars', x, y, radius, depth,
                        base_percentile=base_percentile)
                else:
                    cusp_model = textures.make_cusp_model(
                        self.data, x, y, radius=radius, depth=depth,
                        base_percentile=base_percentile)
                    base_height = cusp_model.base_height

                    yy, xx = np.indices(self.data.shape)
                    self.data = textures.apply_textures(
                        self.data, cusp_model(xx, yy))
                    log.info('Placed cusp texture at the galaxy center.')

                return base_height

    def _apply_textures(self):
        """Apply all textures to the model."""

        if not self.has_intensity:
            self.data = 0.

        if self.has_textures:
            self._add_masked_textures()
            self._apply_stellar_textures()
            self._apply_spiral_central_cusp()

    def _make_model_base(self, filter_size=75, min_value=1.):
        """
        Make a structural base for the model and replace zeros with
        ``min_value``.

        For two-sided models, this is used to create a stronger base,
        which prevents the model from shaking back and forth due to
        printer vibrations.  These structures will have a *total* width
        of ``self.base_height``.

        Parameters
        ----------
        filter_size : int, optional
            The size of the binary dilation filter.
        """

        log.info('Making model base.')

        if not self.has_intensity:
            self._base_layer = self.base_height
        else:
            if not self.double_sided:
                self._base_layer = self.base_height
            else:
                selem = np.ones((filter_size, filter_size))
                img = ndimage.binary_dilation(self.data, structure=selem)
                self._base_layer = np.where(
                    img == 0, self.base_height / 2., 0)
        self.data += self._base_layer
        self.data[self.data == 0.] = min_value

    def make(self, compress_bulge_percentile=0., compress_bulge_factor=0.05,
             suppress_background_percentile=90.,
             suppress_background_factor=0.2, smooth_size=11,
             minvalue_to_zero=0.02, crop_data_threshold=0.,
             model_base_filter_size=75):
        """
        Make the model.

        A series of steps are performed to prepare the intensity image
        and/or add textures to the model.
        """

        self.data = deepcopy(self.data_original_resized)    # start fresh
        self._prepare_masks()
        self._scale_stellar_table_positions(
            self.stellar_tables_original, self.resize_scale_factor)
        self._remove_stars()
        self._spiralgalaxy_compress_bulge(
            percentile=compress_bulge_percentile,
            factor=compress_bulge_factor)
        self._suppress_background(percentile=suppress_background_percentile,
                                  factor=suppress_background_factor)
        self._smooth_image(size=smooth_size)
        self._normalize_image()
        self._minvalue_to_zero(min_value=minvalue_to_zero)
        # TODO: add a step here to remove "islands" using segmentation?
        self._crop_data(threshold=0., resize=True)
        self._make_model_height()
        self.data_intensity = deepcopy(self.data)
        self._apply_textures()
        self._make_model_base(filter_size=model_base_filter_size,
                              min_value=1.)
        self._model_complete = True
        log.info('Make complete!')

    def find_peaks(self, snr=10, snr_min=5, npixels=10, min_count=25,
                   max_count=50, sigclip_iters=10,
                   stellar_type='star_clusters'):
        """
        Find the brightest "sources" above a threshold in an image.

        For complex images, such as an image of a bright galaxy, manual
        editting of the final source list will likely be necessary.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Image in which to find sources.

        snr : float, optional
            The signal-to-noise ratio per pixel above the "background"
            for which to consider a pixel as possibly being part of a
            source.

        snr_min : float, optional
            The minimum signal-to-noise ratio per above to consider.
            See ``min_count``.

        npixels : int, optional
            The number of connected pixels, each greater than the
            threshold values (calculated from ``snr``), that an object
            must have to be detected.  ``npixels`` must be a positive
            integer.

        min_count : int, optional
            The minimum number of "sources" to try to find.  If at least
            ``min_count`` sources are not found at the input ``snr``,
            then ``snr`` is incrementally lowered.  The ``snr`` will not
            go below ``snr_min``, so the returned number of sources may
            be less than ``min_count``.

        max_count : int, optional
            The maximum number of "sources" to find.  The brightest
            ``max_count`` sources are returned.

        sigclip_iters : float, optional
           The number of iterations to perform sigma clipping, or `None`
           to clip until convergence is achieved (i.e., continue until
           the last iteration clips nothing) when calculating the image
           background statistics.

        stellar_type : int, optional
            The type of sources.  The source table is stored in
            ``self.stellar_tables[stellar_type]`` and the version
            resized to the original image size is stored in
            ``self.stellar_tables_original[stellar_type]``.

        Returns
        -------
        result : `~astropy.table.Table`
            A table of the found "sources".

        Notes
        -----
        The background and background noise estimated by this routine
        are based simple sigma-clipped statistics.  For complex images,
        such as an image of a bright galaxy, the "background" will not
        be accurate.
        """

        if min_count > max_count:
            raise ValueError('min_count must be <= max_count')

        self.data = deepcopy(self.data_original_resized)

        columns = ['id', 'xcentroid', 'ycentroid', 'segment_sum']
        while snr >= snr_min:
            threshold = photutils.detect_threshold(
                self.data, snr=snr, mask_value=0.0,
                sigclip_iters=sigclip_iters)
            segm_img = photutils.detect_sources(self.data, threshold,
                                                npixels=npixels)
            segm_props = photutils.segment_properties(self.data, segm_img)
            tbl = photutils.properties_table(segm_props, columns=columns)

            if len(tbl) >= min_count:
                break
            else:
                snr -= 1.

        tbl.sort('segment_sum')
        tbl.reverse()
        if len(tbl) > max_count:
            tbl = tbl[:max_count]
        tbl.rename_column('segment_sum', 'flux')

        scaled_tbl = self._scale_table_positions(
            tbl, 1. / self.resize_scale_factor)
        self.stellar_tables_original[stellar_type] = scaled_tbl

        self.stellar_tables = deepcopy(self.stellar_tables_original)
        self.stellar_tables[stellar_type] = tbl

        return self.stellar_tables_original

    def make_spiral_galaxy_masks(self, smooth_size=11, gas_percentile=55.,
                                 spiral_percentile=75.):
        """
        For a spiral galaxy image, automatically generate texture masks
        for spiral arms and gas.

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

        if gas_percentile >= spiral_percentile:
            raise ValueError('gas_percentile must be less than '
                             'spiral_percentile.')

        texture_type = self._translate_mask_type('bulge')
        if texture_type not in self.texture_masks_original:
            warnings.warn('You must first define the bulge mask.',
                          AstropyUserWarning)
            return

        self.data = deepcopy(self.data_original_resized)
        self._prepare_masks()
        self._remove_stars()
        self._smooth_image(size=smooth_size)

        bulge_mask = self.texture_masks[texture_type]
        x, y = self._find_galaxy_center(bulge_mask)
        data = self.data * image_utils.radial_weight_map(self.data.shape,
                                                         (y, x))
        data[bulge_mask] = 0.    # exclude the bulge mask region

        # define the "spiral arms" mask
        spiral_threshold = np.percentile(data, spiral_percentile)
        spiral_mask = (data > spiral_threshold)

        # define the "gas" mask
        gas_threshold = np.percentile(data, gas_percentile)
        gas_mask = np.logical_and(data > gas_threshold, ~spiral_mask)

        for mask_type, mask in zip(['spiral', 'gas'],
                                   [spiral_mask, gas_mask]):
            texture_type = self._translate_mask_type(mask_type)
            if texture_type in self.texture_masks_original:
                warnings.warn('Overwriting existing "{0}" texture mask',
                              AstropyUserWarning)

            mask = image_utils.resize_image(mask,
                                            1. / self.resize_scale_factor)
            region_mask = RegionMask(mask, mask_type)
            self.texture_masks_original[texture_type] = [region_mask]

        log.info('Automatically generated "spiral" and "gas" masks for '
                 'spiral galaxy.')
