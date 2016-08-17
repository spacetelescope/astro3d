"""
This module provides tools to create a 3D model from an astronomical image.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
from copy import deepcopy, copy
import glob
import warnings
from functools import partial
from distutils.version import LooseVersion

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


if LooseVersion(photutils.__version__) < LooseVersion('0.3'):
    PHOTUTILS_LT_0P3 = True
else:
    PHOTUTILS_LT_0P3 = False


__all__ = ['Model3D']
__doctest_skip__ = ['Model3D', 'Model3D.read_all_masks']


class Model3D(object):
    """
    Class to create a 3D model from an astronomical image.

    Parameters
    ----------
    data : array-like
        The input 2D array from which to create a 3D model.

    Examples
    --------
    >>> # initialize the model
    >>> model = Model3D(data)     # from an array
    >>> model = Model3D.from_fits('myimage.fits')    # or from a FITS file
    >>> model = Model3D.from_rgb('myimage.png')      # or from a bitmap image

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
    >>> model.make(intensity=True, textures=True, double_sided=True,
    ...            spiral_galaxy=True)

    >>> # write the model to a STL file
    >>> filename_prefix = 'myobject'
    >>> model.write_stl(filename_prefix, split_model=True)

    >>> # write the texture and region masks to FITS files
    >>> model.write_all_masks(filename_prefix)

    >>> # write stellar tables (separately or all at once)
    >>> model.write_stellar_table(filename_prefix, stellar_type='stars')
    >>> model.write_stellar_table(filename_prefix,
    ...                           stellar_type='star_clusters')
    >>> model.write_all_stellar_tables(filename_prefix)    # all at once
    """

    def __init__(self, data):
        self.data_original = np.asanyarray(data)
        self._model_complete = False

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
            textures.dots_texture_image, profile='spherical', diameter=9.0,
            height=4.0, grid_func=textures.hexagonal_grid, grid_spacing=7.0)
        self.textures['dots'] = partial(
            textures.dots_texture_image, profile='spherical', diameter=9.0,
            height=4.0, grid_func=textures.hexagonal_grid, grid_spacing=11.0)
        self.textures['lines'] = partial(
            textures.lines_texture_image, profile='linear', thickness=13,
            height=7.8, spacing=20, orientation=0)

    @classmethod
    def from_fits(cls, filename):
        """
        Create a `Model3D` instance from a FITS file.

        Parameters
        ----------
        filename : str
            The name of the FITS file.
        """

        log.info('Reading FITS data from "{0}"'.format(filename))
        data = fits.getdata(filename)
        if data is None:
            raise ValueError('data not found in the FITS file')

        if data.ndim not in [2, 3]:
            raise ValueError('data is not a 2D image or a 3D RGB cube')

        if data.ndim == 3:    # RGB cube
            log.info('Converting RGB FITS cube to 2D data array')
            data = data[0] * 0.299 + data[1] * 0.587 + data[2] * 0.144
            data = image_utils.remove_nonfinite(data)

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

        log.info('Reading RGB data from "{0}"'.format(filename))
        data = np.array(Image.open(filename).convert('L'),
                        dtype=np.float32)[::-1]
        return cls(data)

    def _translate_mask_type(self, mask_type):
        """
        Translate mask type into the texture or region type.

        The valid texture types are defined in ``self.texture_order``.
        The valid region types are defined in
        ``self.region_mask_types``.

        Parameters
        ----------
        mask_type : str
            The mask type.

        Returns
        -------
        texture_type: str
            The texture type (e.g. 'small_dots', 'dots', or 'lines') or
            region type (e.g. 'smooth') translated from the mask type.
        """

        if mask_type in self.translate_texture:
            return mask_type
        else:
            tx_type = None
            for tx_type, mask_types in self.translate_texture.items():
                if mask_type in mask_types:
                    return tx_type
            if tx_type is None:
                warnings.warn('"{0}" is not a valid mask type.'
                              .format(mask_type), AstropyUserWarning)
            return

    def add_mask(self, mask):
        """
        Add a region mask to the `region_masks_original` or
        `texture_masks_original` dictionary, keyed by the mask type.

        Parameters
        ----------
        mask : `RegionMask`
            A `RegionMask` object.

        Returns
        -------
        mask_type : str
            The mask type of the read FITS file.
        """

        mask_type = mask.mask_type
        mtype = self._translate_mask_type(mask_type)
        if mtype in self.region_mask_types:
            self.region_masks_original[mtype].append(mask)
        else:
            self.texture_masks_original[mtype].append(mask)
        return mask_type

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

        Returns
        -------
        mask_type : str
            The mask type of the read FITS file.
        """

        region_mask = RegionMask.from_fits(
            filename, required_shape=self.data_original.shape)
        mask_type = self.add_mask(region_mask)
        log.info('Loaded "{0}" mask from "{1}"'.format(mask_type, filename))
        return mask_type

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
        >>> model3d = Model3D(data)
        >>> model3d.read_all_masks('*.fits')
        >>> model3d.read_all_masks('masks/*.fits')
        """

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

        for mask_type, masks in self.region_masks_original.items():
            nmasks = len(masks)
            for i, mask in enumerate(masks, 1):
                if nmasks > 1:
                    filename = '{0}_{1}_{2}.fits'.format(filename_prefix,
                                                         mask_type, i)
                else:
                    filename = '{0}_{1}.fits'.format(filename_prefix,
                                                     mask_type)
                mask.write(filename)
        for texture_type, masks in self.texture_masks_original.items():
            nmasks = len(masks)
            for i, mask in enumerate(masks, 1):
                mask_type = mask.mask_type
                if nmasks > 1:
                    filename = '{0}_{1}_{2}.fits'.format(filename_prefix,
                                                         mask_type, i)
                else:
                    filename = '{0}_{1}.fits'.format(filename_prefix,
                                                     mask_type)
                mask.write(filename)

    def add_stellar_table(self, table, stellar_type):
        """
        Add a table of stars or star clusters from a table.

        The table must contain ``'xcentroid'`` and ``'ycentroid'``
        columns and a ``'flux'`` and/or ``'magnitude'`` column.

        Parameters
        ----------
        table : ~astropy.Table
            The table

        stellar_type : {'stars', 'star_clusters'}
            The type of the table.
        """

        self.stellar_tables_original[stellar_type] = table

    def read_stellar_table(self, filename, stellar_type):
        """
        Read a table of stars or star clusters from a file.

        The table must contain ``'xcentroid'`` and ``'ycentroid'``
        columns and a ``'flux'`` and/or ``'magnitude'`` column.

        Parameters
        ----------
        filename : str
            The filename containing an `~astropy.Table` in ASCII format.

        stellar_type : {'stars', 'star_clusters'}
            The type of the table.
        """

        table = read_stellar_table(filename, stellar_type)
        self.add_stellar_table(table, stellar_type)

    def read_star_clusters(self, filename):
        """
        Read star clusters table from an ASCII file.

        Parameters
        ----------
        filename : str
            The filename containing an `~astropy.Table` in ASCII format.
        """

        self.read_stellar_table(filename, 'star_clusters')

    def read_stars(self, filename):
        """
        Read stars table from an ASCII file.

        Parameters
        ----------
        filename : str
            The filename containing an `~astropy.Table` in ASCII format.
        """

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
            log.info('"{0}" table was empty and not saved.'
                     .format(stellar_type))

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
            try:
                self.write_stellar_table(filename_prefix, stellar_type)
            except KeyError:
                pass

    def write_stl(self, filename_prefix, x_size_mm=275, split_model=True,
                  stl_format='binary', clobber=False):
        """
        Write the 3D model to a STL file(s).

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output filenames.  The output filename
            will be '<filename_prefix>.stl'.  If ``split_image=True``,
            then the filename will be '<filename_prefix>_part[1|2].stl'.

        x_size_mm : int, optional
            The x size of the model in mm.

        split_model : bool, optional
            If `True`, then split the model into two halves, a bottom
            and top part.

        stl_format : {'binary', 'ascii'}, optional
            Format for the output STL file.  The default is 'binary'.
            The binary STL file is harder to debug, but requires less
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
            write_mesh(model1, filename_prefix + '_part1',
                       x_size_mm=x_size_mm, double_sided=self._double_sided,
                       stl_format=stl_format, clobber=clobber)
            write_mesh(model2, filename_prefix + '_part2',
                       x_size_mm=x_size_mm, double_sided=self._double_sided,
                       stl_format=stl_format, clobber=clobber)
        else:
            write_mesh(self.data, filename_prefix, x_size_mm=x_size_mm,
                       double_sided=self._double_sided, stl_format=stl_format,
                       clobber=clobber)

    def _prepare_data(self):
        """
        Resize the input data.
        """

        log.info('Preparing data (resizing).')

        self.data_original_resized = image_utils.resize_image(
            image_utils.remove_nonfinite(self.data_original),
            self._resize_scale_factor)
        self.data = deepcopy(self.data_original_resized)

    def _prepare_masks(self):
        """
        Prepare texture and region masks.

        Texture masks are combined and resized.  Region masks are
        resized, but not combined.
        """

        log.info('Preparing masks (combining and resizing).')

        self.texture_masks = {}
        self.region_masks = {}

        # combine and resize texture_masks
        for mask_type, masks in self.texture_masks_original.items():
            prepared_mask = image_utils.resize_image(
                image_utils.combine_region_masks(masks),
                self._resize_scale_factor)
            self.texture_masks[mask_type] = prepared_mask   # ndarray

        # resize but do not combine region_masks
        for mask_type, masks in self.region_masks_original.items():
            resized_masks = [image_utils.resize_image(
                mask.mask, self._resize_scale_factor) for mask in masks]
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

        result = table.copy()
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

        log.info('Scaling stellar table positions.')

        self.stellar_tables = copy(stellar_tables)
        for stellar_type, table in self.stellar_tables.items():
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

        log.info('Removing bright stars identified in "remove_star" mask(s).')
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

        A base value is first taken as the ``percentile`` of data values
        within the bulge mask.  Values above this are then compressed by
        ``factor``.

        Parameters
        ----------
        percentile: float in range of [0, 100], optional
            The percentile of pixel values within the bulge mask to use
            as the base level.

        factor : float, optional
            The scale factor to apply to the pixel values above the base
            level.

        Returns
        -------
        result : float
            The base level above which pixel values are compressed.
        """

        if not self._spiral_galaxy:
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

        A base value is first taken as the ``percentile`` of data values
        in regions that are not in any texture mask.  Values below this
        are then compressed by ``factor``.  Values below
        ``floor_percentile`` are then set to zero.

        Parameters
        ----------
        percentile: float in range of [0, 100], optional
            The percentile of pixel values outside of the masked regions
            to use as the background level.

        factor : float, optional
            The scale factor to apply to the pixel values below the
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
        zero_mask = (self.data == 0.)
        mask = np.logical_or(mask, zero_mask)
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
        size : int or tuple of int, optional
            The shape of filter window.  If ``size`` is an `int`, then
            then ``size`` will be used for both dimensions.
        """

        log.info('Smoothing the image with a 2D median filter of size '
                 '{0} pixels.'.format(size))
        self.data = ndimage.filters.median_filter(self.data, size=size)

    def _normalize_image(self, max_value=1.0):
        """
        Normalize the image.

        Parameters
        ----------
        max_value : float, optional
            The maximum value of the normalized array.
        """

        log.info('Normalizing the image values to [0, {0}].'
                 .format(max_value))
        self.data = image_utils.normalize_data(self.data, max_value=max_value)

    def _minvalue_to_zero(self, min_value=0.02):
        """
        Set values below a certain value to zero.

        Parameters
        ----------
        min_value : float, optional
            The image threshold value below which pixels are set to zero.
        """

        log.info('Setting image values below {0} to zero.'.format(min_value))
        self.data[self.data < min_value] = 0.0

    def _extract_galaxy(self):
        if not self._spiral_galaxy:
            return None

        log.info('Extracting the galaxy using source segmentation.')
        segm = photutils.detect_sources(self.data, 0., 1)

        if PHOTUTILS_LT_0P3:
            props = photutils.source_properties(self.data, segm)
            areas = [prop['area'].value for prop in props]
            label = np.argmax(areas) + 1
        else:
            label = np.argmax(segm.areas[1:]) + 1    # exclude label=0

        log.info('Extracting source label: {0}'.format(label))
        segm.keep_labels(label)
        segm_mask = segm.data.astype(bool)
        self.data *= segm_mask

        for mask_type, mask in self.texture_masks.items():
            log.info('Masking the texture masks for the extracted galaxy.')
            mask[~segm_mask] = 0
            self.texture_masks[mask_type] = mask

        log.info('Pruning the stellar tables for the extracted galaxy.')
        for stellar_type, table in self.stellar_tables.items():
            values = []
            for row in table:
                x = int(round(row['xcentroid']))
                y = int(round(row['ycentroid']))
                values.append(self.data[y, x])
            values = np.array(values)
            idx = np.where(values == 0.)
            table.remove_rows(idx)
            self.stellar_tables[stellar_type] = table

    def _crop_data(self, threshold=0.0, pad_width=20, resize=True):
        """
        Crop the image, masks, and stellar tables.

        The image(s) and table(s) are cropped to the minimal bounding
        box containing all pixels greater than the input ``threshold``.

        If ``resize=True``, they are then resized to have the same ``x``
        size as ``self.data_original_resized`` to ensure a consistent
        height/width ratio of the model in MakerBot.

        Parameters
        ----------
        threshold : float, optional
            Pixels greater than the threshold define the minimal bounding box
            of the cropped region.

        pad_width : int, optional
            The number of pixels used to pad the array after cropping to
            the minimal bounding box.

        resize : bool, optional
            Set to `True` to resize the data, masks, and stellar tables
            back to the original ``x`` size of
            ``self.data_original_resized``.

        Returns
        -------
        result : tuple of slice objects
            The slice tuple used to crop the images(s).
        """

        log.info('Cropping the image using a threshold of {0} to define '
                 'the minimal bounding box'.format(threshold))
        slc = image_utils.bbox_threshold(self.data, threshold=threshold)
        self.data = self.data[slc]
        if pad_width != 0:
            log.info('Padding the image by {0} pixels.'.format(pad_width))
            self.data = np.pad(self.data, pad_width, mode=str('constant'))

        for mask_type, mask in self.texture_masks.items():
            log.info('Cropping "{0}" mask.'.format(mask_type))
            self.texture_masks[mask_type] = mask[slc]
            if pad_width != 0:
                log.info('Padding the mask by {0} pixels.'.format(pad_width))
                self.texture_masks[mask_type] = np.pad(
                    self.texture_masks[mask_type], pad_width,
                    mode=str('constant'))

        for stellar_type, table in self.stellar_tables.items():
            idx = ((table['xcentroid'] > slc[1].start) &
                   (table['xcentroid'] < slc[1].stop) &
                   (table['ycentroid'] > slc[0].start) &
                   (table['ycentroid'] < slc[0].stop))
            table = table[idx]
            table['xcentroid'] -= slc[1].start - pad_width
            table['ycentroid'] -= slc[0].start - pad_width
            self.stellar_tables[stellar_type] = table

        if resize:
            scale_factor = float(self.data_original_resized.shape[1] /
                                 self.data.shape[1])
            self.data = image_utils.resize_image(
                self.data, scale_factor)
            for mask_type, mask in self.texture_masks.items():
                log.info('Resizing "{0}" mask.'.format(mask_type))
                self.texture_masks[mask_type] = image_utils.resize_image(
                    mask, scale_factor)
            self._scale_stellar_table_positions(self.stellar_tables,
                                                scale_factor)
        return slc

    def _make_model_height(self, model_height=200):
        """
        Scale the image to the final model height prior to adding the
        textures.

        To give a consistent model height/width ratio and texture height
        (and "feel"), no scaling of the image should happen after this
        step!

        Parameters
        ----------
        model_height : float, optional
            The maximum value in the intensity image, which controls the
            final model height.  This is the height of the intensity map
            *before* the textures, including the spiral galaxy central
            cusp, are applied.
        """

        # clip the image at the cusp base_height
        if self._spiral_galaxy:
            if self._has_intensity:
                base_height = self._apply_spiral_central_cusp(
                    base_height_only=True)
                # NOTE: this will also clip values outside of the bulge
                log.info('Clipping image values at {0} (for central '
                         'cusp).'.format(base_height))
                self.data[self.data > base_height] = base_height

        if self._double_sided:
            model_height = model_height / 2.
        self._normalize_image(max_value=model_height)

    def _add_masked_textures(self):
        """
        Add masked textures (e.g. small dots, dots, lines) to the
        image.

        The masked textures are added in order, specified by
        ``self.texture_order``.  Masked areas in subsequent textures
        will override earlier textures for pixels masked in more than
        one texture (i.e. a given pixel has only one texture applied).
        """

        self._texture_layer = np.zeros_like(self.data)
        for texture_type in self.texture_order:
            if texture_type not in self.texture_masks:
                continue

            log.info('Adding "{0}" textures.'.format(texture_type))
            mask = self.texture_masks[texture_type]
            texture_data = self.textures[texture_type](mask.shape, mask=mask)
            self._texture_layer[mask] = texture_data[mask]
        self.data += self._texture_layer

    def _apply_stellar_textures(self, radius_a=10., radius_b=5., slope=1.0,
                                depth=3., base_percentile=0.):
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
        radius_a : float, optional
            The intercept term in calculating the star radius (see above).

        radius_b : float, optional
            The slope term in calculating the star radius (see above).

        slope : float, optional
            The slope of the star texture sides.

        depth : float
            The maximum depth of the crater-like bowl of the star
            texture.

        base_percentile : float in the range of [0, 100], optional
            The percentile of the image data values within the source
            texture (where the texture is non-zero) used to define the
            base height of the model texture.  If `None`, then the model
            base_height will be zero.
        """

        if self._has_intensity:
            data = self.data
        else:
            data = np.zeros_like(self.data)

        if len(self.stellar_tables) == 0:
            if not self._has_intensity:
                log.info('Discarding data intensity')
                self.data = self._texture_layer
        else:
            log.info('Adding stellar-like textures.')
            self._stellar_texture_layer = textures.make_starlike_textures(
                data, self.stellar_tables, radius_a=radius_a,
                radius_b=radius_b, depth=depth, slope=slope,
                base_percentile=base_percentile)

            if self._has_intensity:
                self.data = textures.apply_textures(
                    self.data, self._stellar_texture_layer)
            else:
                log.info('Discarding data intensity')
                self.data = textures.apply_textures(
                    self._texture_layer, self._stellar_texture_layer)

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
        log.info('Found center of galaxy at x={0}, y={1}.'
                 .format(x_center, y_center))
        return x_center, y_center

    def _apply_spiral_central_cusp(self, radius=25., depth=8., slope=1.0,
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

        slope : float, optional
            The slope of the star texture sides.

        base_height_only : bool, optional
            If `True`, then simply return the base height of the texture
            model, i.e. do not actually add the central cusp.

        Returns
        -------
        result : float
            The base height of the texture model.
        """

        if self._spiral_galaxy:
            if self._has_intensity:
                base_percentile = 0.
            else:
                base_percentile = None

            texture_type = self._translate_mask_type('bulge')
            bulge_mask = self.texture_masks[texture_type]
            if bulge_mask is not None:
                x, y = self._find_galaxy_center(bulge_mask)

                if base_height_only:
                    base_height = textures.starlike_model_base_height(
                        self.data, 'stars', x, y, radius, depth, slope,
                        base_percentile=base_percentile)
                else:
                    cusp_model = textures.make_cusp_model(
                        self.data, x, y, radius=radius, depth=depth,
                        slope=slope, base_percentile=base_percentile)
                    base_height = cusp_model.base_height

                    yy, xx = np.indices(self.data.shape)
                    self._cusp_texture_layer = cusp_model(xx, yy)
                    self.data = textures.apply_textures(
                        self.data, self._cusp_texture_layer)
                    log.info('Placed cusp texture at the galaxy center.')

                return base_height

    def _apply_textures(self, star_texture_depth=3.,
                        star_texture_base_percentile=0.)
        """
        Apply all textures to the model.

        Parameters
        ----------
        star_texture_depth : float, optional
            The maximum depth of the crater-like bowl of the star
            texture.

        star_texture_base_percentile : float in the range of [0, 100], optional
            The percentile of the image data values within the stellar
            texture (where the texture is non-zero) used to define the
            base height of the model texture.
        """

        if self._has_textures:
            self._add_masked_textures()
            self._apply_stellar_textures(
                depth=star_texture_depth,
                base_percentile=star_texture_base_percentile)
            self._apply_spiral_central_cusp()

    def _make_model_base(self, base_height=18.18, filter_size=10,
                         min_value=1.83, fill_holes=True):
        """
        Make a structural base for the model and replace zeros with
        ``min_value``.

        For two-sided models, this is used to create a stronger base,
        which prevents the model from shaking back and forth due to
        printer vibrations.  These structures will have a *total* width
        of ``self.base_height`` (i.e. it is not doubled for the
        double-sided model).

        Parameters
        ----------
        base_height : float, optional
            The height of the model structural base.

        filter_size : int, optional
            The size of the binary dilation filter.

        min_value : float, optional
            The minimum value that the final image can have, e.g.
            prevents printing zeros.  The default value of 1.83
            corresponds to 0.5 mm (which will be doubled to 1 mm for a
            double-sided model).

        fill_holes : bool, optional
            Whether to fill interior holes (e.g. between spiral galaxy
            arms) with the ``base_height``.  Otherwise a "thin" region
            of height ``min_value`` will be placed around the interior
            of the hole.
        """

        log.info('Making model base.')
        if self._double_sided and self._has_intensity:
            data_mask = self.data.astype(bool)
            selem = np.ones((filter_size, filter_size))
            dilation_mask = ndimage.binary_dilation(data_mask,
                                                    structure=selem)
            self._base_layer = np.where(dilation_mask == 0,
                                        base_height / 2., 0)
            if fill_holes:
                galaxy_mask = ndimage.binary_fill_holes(data_mask)
                holes_mask = galaxy_mask * ~data_mask
                self._base_layer[holes_mask] = base_height / 2.
        else:
            self._base_layer = base_height
        self.data += self._base_layer
        self.data[self.data < min_value] = min_value

    def make(self, intensity=True, textures=True, double_sided=False,
             spiral_galaxy=False, model_xsize=1000,
             compress_bulge_percentile=0., compress_bulge_factor=0.05,
             suppress_background_percentile=90.,
             suppress_background_factor=0.2, smooth_size1=11,
             smooth_size2=15, minvalue_to_zero=0.02, crop_data_threshold=0.,
             crop_data_pad_width=20, model_height=200,
             star_texture_depth=3., star_texture_base_percentile=0.,
             model_base_height=18.18, model_base_filter_size=10,
             model_base_min_value=1.83, model_base_fill_holes=True):
        """
        Make the model.

        A series of steps are performed to prepare the intensity image
        and/or add textures to the model.

        Parameters
        ----------
        intensity : bool
            Whether the 3D model has intensities.  At least one of
            ``intensity`` and ``textures`` must be `True`.

        textures : bool
            Whether the 3D model has textures.  At least one of
            ``intensity`` and ``textures`` must be `True`.

        double_sided : bool
            Whether the 3D model is double sided.  Double-sided models
            are generated using a simple reflection.

        spiral_galaxy : bool
            Whether the 3D model is a spiral galaxy, which uses special
            processing.

        model_xsize : int, optional
            The size of the x axis of the model image.

        compress_bulge_percentile : float in range of [0, 100], optional
            The percentile of pixel values within the bulge mask to use
            as the base level when compressing the bulge.  See
            `_spiralgalaxy_compress_bulge`.

        compress_bulge_factor : float, optional
            The scale factor to apply to the region above the base level
            in the bulge compression step.  See
            `_spiralgalaxy_compress_bulge`.

        suppress_background_percentile : float in range of [0, 100], optional
            The percentile of pixel values outside of the masked regions
            to use as the background level in the suppress background
            step.  See `_suppress_background`.

        suppress_background_factor : float, optional
            The scale factor to apply to the region below the background
            level in the suppress background step.  See
            `_suppress_background`.

        smooth_size1 : float or tuple optional
            The shape of filter window for the first image smoothing
            step.  If ``size`` is an `int`, then then ``size`` will be
            used for both dimensions.  Set to `None` to skip the first
            smoothing step.  See `_smooth_image`.

        smooth_size2 : float or tuple optional
            The shape of filter window for the second image smoothing
            step.  If ``size`` is an `int`, then then ``size`` will be
            used for both dimensions.  Set to `None` to skip the second
            smoothing step.  See `_smooth_image`.

        minvalue_to_zero : float, optional
            The image threshold value below which pixels are set to
            zero.  See `_minvalue_to_zero`.

        crop_data_threshold : float, optional
            The values equal to and below which to crop from the data.
            See `_crop_data`.

        crop_data_pad_width : int, optional
            The number of pixels used to pad the array after cropping to
            the minimal bounding box.  See `_crop_data`.

        model_height : float, optional
            The maximum value in the intensity image, which controls the
            final model height.  This is the height of the intensity map
            *before* the textures, including the spiral galaxy central
            cusp, are applied.

        star_texture_depth : float, optional
            The maximum depth of the crater-like bowl of the star
            texture.

        star_texture_base_percentile : float in the range of [0, 100], optional
            The percentile of the image data values within the stellar
            texture (where the texture is non-zero) used to define the
            base height of the model texture.

        model_base_height : float, optional
            The height of the model structural base.  See
            `_make_model_base`.

        model_base_filter_size : int, optional
            The size of the binary dilation filter used in making the
            model base.  See `_make_model_base`.

        model_base_min_value : float, optional
            The minimum value that the final image can have, e.g.
            prevents printing zeros.  The default value of 1.83
            corresponds to 0.5 mm (which will be doubled to 1 mm for a
            double-sided model).  See `_make_model_base`.

        model_base_fill_holes : bool, optional
            Whether to fill interior holes (e.g. between spiral galaxy
            arms) with the ``model_base_height``.  Otherwise a "thin"
            region of height ``min_value`` will be placed around the
            interior of the hole.  See `_make_model_base`.

        Notes
        -----
        The maximum model sizes for the MakerBot 2 printer are:
            ``x``: 275 mm
            ``y``: 143 mm
            ``z``: 150 mm

        The maximum model sizes for the MakerBot 5 printer are:
            ``x``: 242 mm
            ``y``: 189 mm
            ``z``: 143 mm

        The model physical scale (mm/pixel) depends on two numbers: the
        input ``model_xsize`` (default 1000) and the ``x_size_mm``
        (default 275) parameter to `write_stl`.  The model scale is
        simply ``x_size_mm`` / ``model_xsize``.  The default is
        275/1000. = 0.275 mm/pixel.

        With the defaults, a ``model_base_height`` of 18.18 corresponds
        to 5.0 mm.  Note that the ``model_base_height`` is the base
        height for both single- and double-sided models (it is not
        doubled for two-sided models).

        With the defaults, a ``model_height`` of 200 corresponds to a
        physical height of 55.0 mm.  This is the height of the intensity
        map *before* the textures, including the spiral galaxy central
        cusp, are applied.

        .. note::

            The physical model sizes above are for the output model
            **before** any subsequent scaling in the MakerBot Desktop or
            any other software.
        """

        if not textures and not intensity:
            raise ValueError('The 3D model must have textures and/or '
                             'intensity.')
        self._has_intensity = intensity
        self._has_textures = textures
        self._double_sided = double_sided
        self._spiral_galaxy = spiral_galaxy

        self._resize_scale_factor = (float(model_xsize) /
                                     self.data_original.shape[1])

        self._prepare_data()
        self._prepare_masks()
        self._scale_stellar_table_positions(
            self.stellar_tables_original, self._resize_scale_factor)
        self._remove_stars()
        self._spiralgalaxy_compress_bulge(
            percentile=compress_bulge_percentile,
            factor=compress_bulge_factor)
        self._suppress_background(percentile=suppress_background_percentile,
                                  factor=suppress_background_factor)
        if smooth_size1 is not None:
            self._smooth_image(size=smooth_size1)
        self._normalize_image()
        self._minvalue_to_zero(min_value=minvalue_to_zero)
        self._extract_galaxy()

        if smooth_size2 is not None:
            # smooth the image again (to prevent printing issues)
            self._smooth_image(size=smooth_size2)

        self._crop_data(threshold=crop_data_threshold,
                        pad_width=crop_data_pad_width, resize=True)
        self._make_model_height(model_height=model_height)
        self.data_intensity = deepcopy(self.data)

        self._apply_textures(
            star_texture_depth=star_texture_depth,
            star_texture_base_percentile=star_texture_base_percentile)
        self._make_model_base(base_height=model_base_height,
                              filter_size=model_base_filter_size,
                              min_value=model_base_min_value,
                              fill_holes=model_base_fill_holes)
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
            segm_props = photutils.source_properties(self.data, segm_img)
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
            tbl, 1. / self._resize_scale_factor)
        self.stellar_tables_original[stellar_type] = scaled_tbl

        self.stellar_tables = deepcopy(self.stellar_tables_original)
        self.stellar_tables[stellar_type] = tbl

        return self.stellar_tables_original

    def make_spiral_galaxy_masks(self, smooth_size=11, model_xsize=1000,
                                 gas_percentile=55., spiral_percentile=75.):
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

        self._resize_scale_factor = (float(model_xsize) /
                                     self.data_original.shape[1])

        self._prepare_data()
        self.data = deepcopy(self.data_original_resized)
        self._prepare_masks()
        self._remove_stars()
        self._smooth_image(size=smooth_size)

        bulge_mask = self.texture_masks[texture_type]
        x, y = self._find_galaxy_center(bulge_mask)
        rwm = image_utils.radial_weight_map(self.data.shape, (y, x))
        minval = rwm[~bulge_mask].min()
        rwm[bulge_mask] = 0.     # exclude the bulge mask region
        rwm /= minval      # min weight value outside of bulge is now 1.
        data = self.data * rwm

        # mask zeros to handle case where image is largely zero (e.g.
        # outside coverage area due to rotation or mosaic)
        mask = (data != 0)

        # define the "spiral arms" mask
        spiral_threshold = np.percentile(data[mask], spiral_percentile)
        spiral_mask = (data > spiral_threshold)

        # define the "gas" mask
        gas_threshold = np.percentile(data[mask], gas_percentile)
        gas_mask = np.logical_and(data > gas_threshold, ~spiral_mask)

        new_regions = []
        for mask_type, mask in zip(['spiral', 'gas'],
                                   [spiral_mask, gas_mask]):
            texture_type = self._translate_mask_type(mask_type)
            if texture_type in self.texture_masks_original:
                warnings.warn('Overwriting existing "{0}" texture mask',
                              AstropyUserWarning)

            mask = image_utils.resize_image(mask,
                                            1. / self._resize_scale_factor)
            region_mask = RegionMask(mask, mask_type)
            self.texture_masks_original[texture_type] = [region_mask]
            new_regions.append(region_mask)

        log.info('Automatically generated "spiral" and "gas" masks for '
                 'spiral galaxy.')
        return new_regions


def read_stellar_table(filename, stellar_type):
    """
    Read a table of stars or star clusters from a file.

    The table must contain ``'xcentroid'`` and ``'ycentroid'`` columns
    and a ``'flux'`` and/or ``'magnitude'`` column.

    Parameters
    ----------
    filename : str
        The filename containing an `~astropy.Table` in ASCII format.

    stellar_type : {'stars', 'star_clusters'}
        The type of the table.

    Returns
    -------
    result : `~astropy.table.Table`
        A table of stellar-like sources.

    Notes
    -----
    This function is called only by the GUI.
    """

    table = Table.read(filename, format='ascii')
    table.keep_columns(['xcentroid', 'ycentroid', 'flux'])
    log.info('Loaded "{0}" table from "{1}"'.format(stellar_type, filename))
    return table
