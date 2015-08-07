"""
This module provides tools to apply textures to an image and to create a
3D model.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from astropy import log
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from PIL import Image
from scipy import ndimage
import photutils

from . import image_utils
from .meshes import to_mesh
from .region_mask import RegionMask
from .textures import (apply_textures, make_starlike_textures,
                       make_cusp_texture, DOTS, SMALL_DOTS, LINES)


class Model3D(object):
    """
    Class to create a 3D model from an astronomical image.

    Examples
    --------
    >>> model = Model3D.from_fits('myimage.fits')

    >>> # define the type of 3D model
    >>> model.has_textures = True
    >>> model.has_intensity = True
    >>> model.double_sided = True
    >>> model.spiral_galaxy = True

    >>> # read the region/texture masks
    >>> model.read_mask('mask_file.fits')      # can read one by one
    >>> model.read_all_masks('*.fits')         # or several at once

    >>> # read the stars and/or star clusters
    >>> model.read_stellar_table('object_stars.txt', 'stars')
    >>> model.read_stars('object_stars.txt')     # same as above
    >>> model.read_stellar_table('object_star_clusters.txt', 'star_clusters')
    >>> model.read_star_clusters('object_star_clusters.txt')   # same as above


    >>> model.make()
    >>> preview_intensity = model.preview_intensity
    >>> preview_dots_mask = model.get_preview_mask(model.dots_key)
    >>> preview_clusters = model.get_final_clusters()
    >>> preview_stars = model.get_final_stars()
    >>> image_for_stl = model.out_image


    >>> filename_prefix = 'myobject'
    >>> model.write_stl(prefix, split_model=True)
    >>> model.write_all_masks(filename_prefix)

    >>> # write stellar tables separately or all at once
    >>> model.write_stellar_table(filename_prefix, stellar_type='stars')
    >>> model.write_all_stellar_tables(filename_prefix)
    """

    def __init__(self, data, resize_xsize=1000):
        """
        Parameters
        ----------
        data : array-like
            The input 2D array from which to create a 3D model.

        resize_xsize : int, optional
            The size of the x axis of the resized image.
        """

        self.input_data = np.asanyarray(data)
        self.data = image_utils.resize_image(
            image_utils.remove_nonfinite(self.input_data),
            x_size=resize_xsize)

        self._model_complete = False
        self._has_textures = True
        self._has_intensity = True
        self._double_sided = False
        self._spiral_galaxy = False

        self.texture_mask_types = ['gas', 'dots_small', 'spiral', 'dots',
                                   'disk', 'lines']
        self.region_mask_types = ['smooth', 'remove_star']
        self.allowed_mask_types = (self.texture_mask_types +
                                   self.region_mask_types)
        self.allowed_stellar_types = ['stars', 'star_clusters']

        self.texture_masks = defaultdict(list)
        self.region_masks = defaultdict(list)
        self.stellar_tables = {}
        for key in self.allowed_stellar_types:
            self.stellar_tables[key] = None


        self._layer_order = [self.lines_key, self.dots_key,
                             self.small_dots_key]

        spiral_translation = {}
        spiral_translation['remove_star'] = 'smooth'
        spiral_translation['gas'] = 'dots_small'
        spiral_translation['spiral'] = 'dots'
        spiral_translation['disk'] = 'lines'

        self.height = 150.0
        self.base_thickness = 20

        self.clus_r_fac_add = 10
        self.clus_r_fac_mul = 5
        self.star_r_fac_add = 10
        self.star_r_fac_mul = 5



        # Results
        self._preview_intensity = None
        self._out_image = None
        self._texture_layer = None
        self._preview_masks = None
        self._final_peaks = {}

        # Image is now ready for the rest of processing when user
        # provides the rest of the info
        self._preproc_img = self.resize(data)

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

    def read_mask(self, filename):
        """
        Read a region mask from a FITS file.

        The mask is read into a `RegionMask` object and then stored in
        the `region_masks` or `texture_masks` dictionary, keyed by the
        mask type.

        The mask data must have the same shape as the input ``data``.
        The masks are resized to have the same shape as ``self.data``.

        Parameters
        ----------
        filename : str
            The name of the FITS file.  The mask data must be in the
            primary FITS extension and the header must have a 'MASKTYPE'
            keyword defining the mask type.
        """

        region_mask = RegionMask.from_fits(
            filename, required_shape=self.input_data.shape,
            shape=self.data.shape)
        mask_type = region_mask.mask_type
        if mask_type not in self.allowed_mask_types:
            warnings.warn('"{0}" is not a valid mask '
                          'type'.format(mask_type), AstropyUserWarning)
            return

        if mask_type in self.region_mask_types:
            self.region_masks[mask_type].append(region_mask)
            region_type = 'Region'
        else:
            self.texture_masks[mask_type].append(region_mask)
            region_type = 'Texture'

        log.info('{0} type "{1}" loaded from {2}'.format(region_type,
                                                         mask_type,
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

        for mask_type, masks in self.region_masks.iteritems():
            nmasks = len(masks)
            for i, mask in enumerate(masks, 1):
                if nmasks > 1:
                    filename = '{0}_{1}_{2}.fits'.format(filename_prefix,
                                                         mask_type, i)
                else:
                    filename = '{0}_{1}.fits'.format(filename_prefix,
                                                     mask_type)
                mask.write(filename, shape=self.input_data.shape)

    def read_stellar_table(self, filename, stellar_type):
        """
        Read a table of stars or star clusters from a file.

        The table must have ``'x_center'``, ``'y_center'``, and
        ``'flux'`` columns.

        Parameters
        ----------
        filename : str
            The filename containing an `~astropy.Table` in ASCII format.

        stellar_type : {'stars', 'star_clusters'}
            The type of the table.
        """

        table = Table.read(filename, format='ascii')
        # TODO: check for required columns
        # TODO: rename input columns, e.g. xcen/x_cen/xcenter -> x_center
        table.keep_columns(['x_center', 'y_center', 'flux'])
        self.stellar_tables[stellar_type] = table

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
        self.stellar_tables[stellar_type].write(filename, format='ascii')
        log.info('Saved {0} table to {1}'.format(stellar_type, filename))

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
                  stl_format='binary'):
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
        """

        if not self._model_complete:
            warn.warnings('The model has not been constructed yet. '
                          'Please run the .make() method before saving '
                          'the STL file.', AstropyUserWarning)
            return

        if split_model:
            model1, model2 = image_utils.split_image(self.data,
                                                     axis='horizontal')
            write_mesh(model1, filename_prefix + '_1',
                       double_sided=self.double_sided, stl_format=stl_format)
            write_mesh(model2, filename_prefix + '_2',
                       double_sided=self.double_sided, stl_format=stl_format)
        else:
            write_mesh(model, filename_prefix,
                       double_sided=self.double_sided, stl_format=stl_format)




    def get_preview_mask(self, key):
        """Boolean mask for given texture key for GUI preview."""
        if self._preview_masks is None:
            raise ValueError('Run make() first')
        return self._preview_masks == key

    def get_final_clusters(self):
        """Star clusters for GUI preview (not in native coords)."""
        if self.clusters_key not in self._final_peaks:
            raise ValueError('Run make() first')
        return self._final_peaks[self.clusters_key]

    def get_final_stars(self):
        """Stars for GUI preview (not in native coords)."""
        if self.stars_key not in self._final_peaks:
            raise ValueError('Run make() first')
        return self._final_peaks[self.stars_key]

    def _process_masks(self):
        """Scale and combine masks."""
        scaled_masks = defaultdict(list)
        disk = None
        spiralarms = None

        for key, reglist in self.region_masks.iteritems():
            masklist = [reg.resize(self._preproc_img.shape)
                        for reg in reglist]

            if key != self.smooth_key:
                scaled_masks[key] = [combine_masks(masklist)]
            else:  # To be smoothed
                scaled_masks[key] = masklist

        if self.is_spiralgal:
            if len(scaled_masks[self.lines_key]) > 0:
                disk = scaled_masks[self.lines_key][0]
            if len(scaled_masks[self.dots_key]) > 0:
                spiralarms = scaled_masks[self.dots_key][0]

        return scaled_masks, disk, spiralarms

    def _crop_masks(self, scaled_masks, ix1, ix2, iy1, iy2):
        """Crop masks."""
        croppedmasks = defaultdict(list)
        disk = None
        spiralarms = None

        for key, mlist in scaled_masks.iteritems():
            if key == self.smooth_key:  # Smoothing already done
                continue
            for mask in mlist:
                croppedmasks[key].append(mask[iy1:iy2, ix1:ix2])

        if self.is_spiralgal:
            if len(croppedmasks[self.lines_key]) > 0:
                disk = croppedmasks[self.lines_key][0]
            if len(croppedmasks[self.dots_key]) > 0:
                spiralarms = croppedmasks[self.dots_key][0]

        return croppedmasks, disk, spiralarms

    def _process_peaks(self):
        """Scale peaks."""
        scaled_peaks = deepcopy(self.peaks)
        fac = self._preproc_img.shape[0] / self.input_data.shape[0]

        for peaks in scaled_peaks.itervalues():  # clusters and stars
            peaks['xcen'] *= fac
            peaks['ycen'] *= fac

        return scaled_peaks

    def _crop_peaks(self, scaled_peaks, key, ix1, ix2, iy1, iy2):
        """Crop peaks."""
        if key in scaled_peaks:
            cropped_peak = deepcopy(scaled_peaks[key])
            cropped_peak = cropped_peak[(cropped_peak['xcen'] > ix1) &
                                        (cropped_peak['xcen'] < ix2 - 1) &
                                        (cropped_peak['ycen'] > iy1) &
                                        (cropped_peak['ycen'] < iy2 - 1)]
            cropped_peak['xcen'] -= ix1
            cropped_peak['ycen'] -= iy1
            log.info('{0} before and after cropping: {1} -> {2}'.format(
                key, len(scaled_peaks[key]), len(cropped_peak)))
        else:
            cropped_peak = []

        return cropped_peak

    def make(self):
        """Make the model."""

        # Don't want to change input for repeated calls
        image = deepcopy(self._preproc_img)

        scaled_masks, disk, spiralarms, scaled_peaks = self.resize_masks()
        image = self.remove_stars(image, scaled_masks)
        image = self.filter_image1(image, size=10)
        image = image_utils.normalize(image, True)
        image = self.spiralgalaxy_scale_top(image, disk, percent=90)
        image = image_utils.normalize(image, True)
        # central cusp orginally applied here
        image = self.emphasize_regs(image, scaled_masks)
        (image, croppedmasks, disk, spiralarms,
         clusters, markstars) = self.crop_image(image, scaled_masks,
                                                scaled_peaks)
        image = self.filter_image2(image)
        image = image_utils.normalize(image, True, self.height)
        # Renormalize again so that height is more predictable
        image = image_utils.normalize(image, True, self.height)

        # Generate monochrome intensity for GUI preview
        self._preview_intensity = deepcopy(image.data)

        # To store non-overlapping key-coded texture info
        self._preview_masks = np.zeros(
            self._preview_intensity.shape, dtype='S10')

        # add central cusp and textures
        if self.has_texture:
            image = self.add_textures(image, croppedmasks)

            # apply stars and star clusters
            if self.has_intensity:
                base_percentile = 75
                depth = 5
            else:
                base_percentile = None
                depth = 10
            starlike_textures = make_starlike_textures(
                image, markstars, clusters, radius_a=self.clus_r_fac_add,
                radius_b=self.clus_r_fac_mul, depth=depth,
                base_percentile=base_percentile)
            # if h_percentile is not None:
            #     filt = ndimage.filters.maximum_filter(array, fil_size)
            #     mask = (filt > 0) & (image > filt) & (array == 0)
            #     array[mask] = filt[mask]
            image = apply_textures(image, starlike_textures)

            # add central cusp for spiral galaxies (do this last,
            # particular after adding the lines texture for the disk
            # bulge)
            if self.is_spiralgal:
                if self.has_intensity:
                    cusp_depth = 20
                    cusp_percentile = 0.
                else:
                    cusp_depth = 20
                    cusp_percentile = None
                bulge_mask = disk
                image = self.spiralgalaxy_central_cusp(
                    image, bulge_mask, radius=25, depth=cusp_depth,
                    base_percentile=cusp_percentile)

        if isinstance(image, np.ma.core.MaskedArray):
            image = image.data

        self.make_model_base(image)
        return

    def resize_masks(self):
        # Scale and combine masks
        scaled_masks, disk, spiralarms = self._process_masks()
        scaled_peaks = self._process_peaks()
        return scaled_masks, disk, spiralarms, scaled_peaks

    def remove_stars(self, image, scaled_masks):
        log.info('Smoothing {0} region(s)'.format(
                len(scaled_masks[self.smooth_key])))
        image = remove_stars(image, scaled_masks[self.smooth_key])
        return image

    def filter_image1(self, image, size=10):
        # size=10 for 1k image
        log.info('Filtering image (first pass)')
        image = ndimage.filters.median_filter(image, size=size)
        image = np.ma.masked_equal(image, 0.0)
        return image

    def spiralgalaxy_scale_top(self, image, disk, percent=90):
        # LDB: should use disk mask
        if self.is_spiralgal and disk is not None:
            log.info('Scaling top')

            # Use a mask that covers high SNR region
            rgrid = self._rad_from_center(
                image.shape, image.shape[1] // 2, image.shape[0] // 2)
            rlim = rgrid.max() / 2
            bigdisk = rgrid < rlim

            image = scale_top(image, mask=bigdisk, percent=percent)
        return image

    def spiralgalaxy_central_cusp(self, image, bulge_mask, radius=25,
                                  depth=40, base_percentile=None):
        """
        Find the galaxy center and then apply the cusp texture.
        """

        x, y = find_galaxy_center(image, bulge_mask)
        cusp_texture = make_cusp_texture(image, x, y, radius=radius,
                                         depth=depth,
                                         base_percentile=base_percentile)
        log.info('Placed cusp texture at the galaxy center.')
        return apply_textures(image, cusp_texture)

    def emphasize_regs(self, image, scaled_masks):
        log.info('Emphasizing regions')
        image = emphasize_regions(
            image, scaled_masks[self.small_dots_key] +
            scaled_masks[self.dots_key] + scaled_masks[self.lines_key])
        return image

    def crop_image(self, image, scaled_masks, scaled_peaks):
        image, iy1, iy2, ix1, ix2 = image_utils.crop_image(image, _max=1.0)
        log.info('Cropped image shape: {0}'.format(image.shape))

        # Also crop masks and lists
        croppedmasks, disk, spiralarms = self._crop_masks(
            scaled_masks, ix1, ix2, iy1, iy2)

        clusters = self._crop_peaks(
            scaled_peaks, self.clusters_key, ix1, ix2, iy1, iy2)
        markstars = self._crop_peaks(
            scaled_peaks, self.stars_key, ix1, ix2, iy1, iy2)

        # Generate list of peaks for GUI preview
        self._final_peaks = {
            self.clusters_key: clusters,
            self.stars_key: markstars}

        return (image, croppedmasks, disk, spiralarms, clusters, markstars)

    def filter_image2(self, image):
        log.info(
            'Filtering image (second pass, height={0})'.format(self.height))
        image = ndimage.filters.median_filter(image, 10)  # For 1k image
        image = ndimage.filters.gaussian_filter(image, 3)  # Magic?
        image = np.ma.masked_equal(image, 0)

        return image

    def add_textures(self, image, croppedmasks):
        # Texture layers

        # Dots and lines

        self._texture_layer = np.zeros(image.shape)

        # Apply layers from bottom up
        for layer_key in self.layer_order[::-1]:
            if layer_key == self.dots_key:
                texture_func = DOTS
            elif layer_key == self.small_dots_key:
                texture_func = SMALL_DOTS
            elif layer_key == self.lines_key:
                texture_func = LINES
            else:
                warnings.warn('{0} is not a valid texture, skipping...'
                              ''.format(layer_key), AstropyUserWarning)
                continue

            log.info('Adding {0}'.format(layer_key))
            for mask in croppedmasks[layer_key]:
                # cur_texture = texture_func(image, mask)
                cur_texture = texture_func(mask)
                self._texture_layer[mask] = cur_texture[mask]
                self._preview_masks[mask] = layer_key

        image += self._texture_layer
        return image

    def make_model_base(self, image):
        log.info('Making base')
        if self.double_sided:
            base_dist = 100  # Magic? Was 60. Widened for nibbler.
            base_height = self.base_thickness / 2  # Doubles in mesh creator
            base = make_base(image, dist=base_dist, height=base_height,
                             snapoff=True)
        else:
            base = make_base(image, height=self.base_thickness, snapoff=False)

        if self.has_intensity:
            self._out_image = image + base
        else:
            self._out_image = self._texture_layer + base
        return

    # For spiral galaxy only
    def _find_galaxy_center(self, image, diskmask):
        """Find center of galaxy."""
        if not self.is_spiralgal:
            raise ValueError('Image is not a spiral galaxy')
        dat = deepcopy(image)
        dat[~diskmask] = 0
        i = np.where(dat == dat.max())
        ycen = i[0].mean()
        xcen = i[1].mean()
        return xcen, ycen

    def _rad_from_center(self, shape, xcen, ycen):
        """Calculate radius of center of galaxy for image pixels."""
        ygrid, xgrid = np.mgrid[0:shape[0], 0:shape[1]]
        dx = xgrid - xcen
        dy = ygrid - ycen
        r = np.sqrt(dx * dx + dy * dy)
        return r

    def _radial_map(self, r, alpha=0.8, r_min=100, r_max=450, dr=5,
                    fillval=0.1):
        """Generate radial map to enhance faint spiral arms in the outskirts.

        **Notes from Perry Greenfield**

        Simple improvement by adding a radial weighting function.
        That is to say, take the median filtered image, and adjust all values
        by a radially increasing function (linear is probably too strong,
        we could compute a radial average so that becomes the adjustment
        (divide by the azimuthally averaged values). This way the spiral
        structure reaches out further. This weighting has to be truncated
        at some point at a certain radius.

        Parameters
        ----------
        r : ndarray
            Output from :meth:`_rad_from_center`.

        alpha : float
            Weight is calculated as :math:`r^{\\alpha}`.

        r_min, rmax : int
            Min and max radii where the weights are flattened using mean
            values from annuli around them.

        dr : int
            Half width of the annuli for ``r_min`` and ``r_max``.

        fillval : float
            Zero weights are replaced with this value.

        """
        r2 = r ** alpha  # some kind of scaling
        r2[r > r_max] = np.mean(r2[(r > (r_max - dr)) & (r < (r_max + dr))])
        r2[r < r_min] = np.mean(r2[(r > (r_min - dr)) & (r < (r_min + dr))])
        r2 /= r2.max()  # normalized from 0 to 1
        r2[r2 == 0] = fillval
        return r2

    def auto_spiralarms(self, shape=None, percentile_hi=75, percentile_lo=55):
        """Automatically generate texture masks for spiral arms
        and gas in a spiral galaxy, and store them in ``self.region_masks``.

        """
        # Don't want to change input
        image = deepcopy(self._preproc_img)

        # Scale and combine masks
        scaled_masks, disk, spiralarms = self._process_masks()
        if disk is None:
            raise ValueError('You must define the disk first')

        log.info('Smoothing {0} region(s)'.format(
                len(scaled_masks[self.smooth_key])))
        image = remove_stars(image, scaled_masks[self.smooth_key])

        # Smooth image
        image = ndimage.filters.median_filter(image, size=10)  # For 1k image

        # Apply weighted radial map
        xcen, ycen = self._find_galaxy_center(image, disk)
        r = self._rad_from_center(image.shape, xcen, ycen)
        r_weigh = self._radial_map(r)
        image *= r_weigh

        # Remove disk from consideration
        image[disk] = 0

        # Dots (spiral arms)
        dmask_thres = np.percentile(image, percentile_hi)
        dmask = image > dmask_thres

        # Small dots (gas)
        sdmask_thres = np.percentile(image, percentile_lo)
        sdmask = (image > sdmask_thres) & (~dmask)

        # Resize and save them as RegionMask
        if shape is not None:
            dmask = image_utils.resize_image(dmask, shape[0], width=shape[1])
            sdmask = image_utils.resize_image(sdmask, shape[0], width=shape[1])
        self.region_masks[self.dots_key] = [RegionMask(dmask, self.dots_key)]
        self.region_masks[self.small_dots_key] = [
            RegionMask(sdmask, self.small_dots_key)]

        log.info('auto find min dthres sdthres max: {0} {1} {2} {3}'.format(
            image.min(), sdmask_thres, dmask_thres, image.max()))


def remove_stars(input_image, starmasks):
    """Patches all bright/foreground stars marked as such by the user.

    Parameters
    ----------
    input_image : ndimage

    starmasks : list
        List of boolean masks of foreground stars that need to be patched.

    Returns
    -------
    image : ndimage

    """
    image = deepcopy(input_image)

    for mask in starmasks:
        ypoints, xpoints = np.where(mask)
        dist = max(ypoints.ptp(), xpoints.ptp())
        xx = [xpoints, xpoints, xpoints + dist, xpoints - dist]
        yy = [ypoints + dist, ypoints - dist, ypoints, ypoints]
        newmasks = []
        warn_msg = []

        for x, y in zip(xx, yy):
            try:
                pts = image[y, x]
            except IndexError as e:
                warn_msg.append('\t{0}'.format(e))
            else:
                newmasks.append(pts)

        if len(newmasks) == 0:
            warnings.warn('remove_stars() failed:\n{0}'.format(
                '\n'.join(warn_msg)), AstropyUserWarning)
            continue

        medians = [newmask.mean() for newmask in newmasks]
        index = np.argmax(medians)
        image[mask] = newmasks[index]

    return image


def scale_top(input_image, mask=None, percent=30, factor=10.0):
    """Linear scale of very high values of image.

    LDB:
        - bigdisk mask (input) is centered on the image, not centered
          on the actual nucleus
        - percent=90 is suppressing spiral arms in ngc3344
        - the normalization in this function needs to be fixed
        - ``factor`` is really the final height above a threshold

    Parameters
    ----------
    input_image : ndarray
        Image array.

    mask : ndarray
        Mask of region with very high values. E.g., disk.

    percent : float
        Percentile between 0 and 100, inclusive.
        Only used if ``mask`` is given.

    factor : float
        Scaling factor.

    Returns
    -------
    image : ndarray
        Scaled image.

    """
    image = deepcopy(input_image)

    if mask is None:
        top = image.mean() + image.std()
    else:
        top = np.percentile(image[mask], percent)

    topmask = image > top
    image[topmask] = top + (image[topmask] - top) * factor / image.max()

    return image


def find_galaxy_center(image, mask=None):
    """
    Find the position of a galaxy center simply as the location of the
    maximum value in the image.

    If a ``mask`` is input, then only those regions will be considered.
    """

    # NOTE:  use np.where instead of np.argmax in case of multiple
    # occurrences of the maximum value
    if mask is None:
        y, x = np.where(image == image.max())
    else:
        data = np.ma.array(image, mask=~mask)
        y, x = np.where(data == data.max())

    y_center = y.mean()
    x_center = x.mean()

    log.info('Center of galaxy at x={0}, y={1}'.format(x_center, y_center))

    return x_center, y_center


def emphasize_regions(input_image, masks, threshold=20, niter=2):
    """Emphasize science data and suppress background.

    Parameters
    ----------
    input_image : ndarray

    masks : list
        List of masks that mark areas of interest.
        If no mask provided (empty list), entire
        image is used for calculations.

    threshold : float
        After regions are emphasized, values less than
        this are set to zero.

    niter : int
        Number of iterations.

    Returns
    -------
    image : ndarray

    """
    image = deepcopy(input_image)
    n_masks = len(masks)

    for i in range(niter):
        if n_masks < 1:
            _min = image.mean()
        else:
            _min = min([image[mask].mean() for mask in masks])
        _min -= image.std() * 0.5
        minmask = image < _min
        image[minmask] = image[minmask] * (image[minmask] / _min)

    # Remove low bound
    boolarray = image < threshold
    log.debug('# background pix set to '
              'zero: {0}'.format(len(image[boolarray])))
    image[boolarray] = 0

    return image


def make_star(radius, height):
    """Creates a crater-like depression that can be used
    to represent a star.

    Similar to :func:`astro3d.utils.texture.make_star`.

    """
    a = np.arange(radius * 2 + 1)
    x, y = np.meshgrid(a, a)
    r = np.sqrt((x - radius)**2 + (y - radius)**2)
    star = height / radius ** 2 * r ** 2
    star[r > radius] = -1
    return star


def make_base(image, dist=60, height=10, snapoff=True):
    """Used to create a stronger base for printing.
    Prevents model from shaking back and forth due to printer vibration.

    .. note::

        Raft can be added using Makerware during printing process.

    Parameters
    ----------
    image : ndarray

    dist : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.
        Only used if ``snapoff=True``.

    height : int
        Height of the base.

    snapoff : bool
        If `True`, base is thin around object border so it
        can be snapped off. Set this to `False` for flat
        texture map or one sided prints.

    Returns
    -------
    max_filt : ndarray
        Array containing base values.

    """
    if snapoff:
        max_filt = ndimage.filters.maximum_filter(image, dist)
        max_filt[max_filt < 1] = -5  # Magic?
        max_filt[max_filt > 1] = 0
        max_filt[max_filt < 0] = height
    else:
        max_filt = np.zeros(image.shape) + height

    return max_filt


def combine_masks(masks):
    """
    Combine boolean masks into a single mask.
    NOTE:  also used by GUI
    """
    if len(masks) == 0:
        return masks

    return reduce(lambda m1, m2: m1 | m2, masks)


def find_peaks(image, remove=0, num=None, threshold=8, npix=10, minpeaks=35):
    """
    Identifies the brightest point sources in an image.

    NOTE:  also called by GUI.

    Parameters
    ----------
    image : ndarray
        Image to find.

    remove : int
        Number of brightest point sources to remove.

    num : int
        Number of unrejected brightest point sources to return.

    threshold, npix : int
        Parameters for ``photutils.detect_sources()``.

    minpeaks : int
        This is the minimum number of peaks that has to be found,
        if possible.

    Returns
    -------
    peaks : list
        Point sources.

    """
    columns = ['id', 'xcentroid', 'ycentroid', 'segment_sum']

    while threshold >= 4:
        threshold = photutils.detect_threshold(
            image, snr=threshold, mask_val=0.0)
        segm_img = photutils.detect_sources(image, threshold, npixels=npix)
        segm_props = photutils.segment_properties(image, segm_img)
        isophot = photutils.properties_table(segm_props, columns=columns)
        isophot.rename_column('xcentroid', 'xcen')
        isophot.rename_column('ycentroid', 'ycen')
        isophot.rename_column('segment_sum', 'flux')
        if len(isophot['xcen']) >= minpeaks:
            break
        else:
            threshold -= 1

    isophot.sort('flux')
    isophot.reverse()

    if remove > 0:
        isophot.remove_rows(range(remove))

    if num is not None:
        peaks = isophot[:num]
    else:
        peaks = isophot

    return peaks
