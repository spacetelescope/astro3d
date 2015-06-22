"""
This module provides tools to apply textures to an image and to create a
3D model.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from astropy import log
from astropy.io import ascii, fits
from astropy.table import Column, Table
from astropy.utils.exceptions import AstropyUserWarning
from PIL import Image
from scipy import ndimage
import photutils

from . import image_utils
from .meshcreator import to_mesh
from .textures import (TextureMask, apply_starlike_textures,
                       DOTS, SMALL_DOTS, LINES)


class Model3D(object):
    """
    Class to create a 3D model from an image.

    This class also supports image previews for the GUI.

    Examples
    --------
    >>> model = Model3D.from_fits('myimage.fits')
    >>> model.is_spiralgal = True
    >>> model.double_sided = True
    >>> model.read_texture_masks('*.fits')
    >>> model.load_peaks(model.clusters_key, 'myclusters.txt')
    >>> model.make()
    >>> preview_intensity = model.preview_intensity
    >>> preview_dots_mask = model.get_preview_mask(model.dots_key)
    >>> preview_clusters = model.get_final_clusters()
    >>> preview_stars = model.get_final_stars()
    >>> image_for_stl = model.out_image
    >>> prefix = 'myprefix'
    >>> model.save_stl(prefix)
    >>> model.save_texture_masks(prefix)
    >>> model.save_peaks(prefix)
    """

    _MIN_NPIXELS = 8.1e5    # 900 x 900
    _MAX_NPIXELS = 1.69e6    # 1300 x 1300
    _RESIZE_AXIS_LEN = 1000

    def __init__(self, image):
        """
        Parameters
        ----------
        image : array_like
            The input image from which to create a 3D model.
        """

        self.input_image = image

        # This can be set from GUI
        self.region_masks = defaultdict(list)
        self.peaks = {
            self.clusters_key: Table(names=['xcen', 'ycen', 'flux']),
            self.stars_key: Table(names=['xcen', 'ycen', 'flux'])}
        self.height = 150.0
        self.base_thickness = 20
        self.clus_r_fac_add = 10
        self.clus_r_fac_mul = 5
        self.star_r_fac_add = 10
        self.star_r_fac_mul = 5
        self.double_sided = False
        self._has_texture = True
        self._has_intensity = True
        self.is_spiralgal = False  # Also initialize layer order

        # Results
        self._preview_intensity = None
        self._out_image = None
        self._texture_layer = None
        self._preview_masks = None
        self._final_peaks = {}

        # Image is now ready for the rest of processing when user
        # provides the rest of the info
        self._preproc_img = np.flipud(self.resize(image))

    def resize(self, image):
        """
        Resize an image such that the longest axes has _RESIZE_AXIS_LEN
        pixels, preserving the image aspect ratio.

        The image is resized only if it contains less than
        _MIN_NPIXELS or more than _MAX_NPIXELS.
        """

        orig_h, orig_w = image.shape
        ny, nx = image.shape
        log.info('Input image is {0}x{1} (ny, nx)'.format(ny, nx))

        if (image.size < self._MIN_NPIXELS or
                image.size > self._MAX_NPIXELS):
            aspect_ratio = float(ny) / nx
            if nx <= ny:
                nx_new = self._RESIZE_AXIS_LEN
                ny_new = int(nx_new * aspect_ratio)
            else:
                ny_new = self._RESIZE_AXIS_LEN
                nx_new = int(ny_new / aspect_ratio)

            image = np.array(Image.fromarray(image).resize(
                (nx_new, ny_new)), dtype=np.float64)
            log.info('Input image was resized from {0}x{1} to {2}x{3}'.format(
                ny, nx, ny_new, nx_new))
        else:
            image = image.astype(np.float64)
            log.info('Input image was not resized.')

        return image

    @classmethod
    def from_fits(cls, filename):
        """Create class instance from FITS file."""
        data = fits.getdata(filename)
        if data is None:
            raise ValueError('FITS file does not have image data')
        elif data.ndim == 3:  # RGB cube from HLA
            data[~np.isfinite(data)] = 0  # Replace NaNs
            data = data.sum(axis=0)

        return cls(data)

    @classmethod
    def from_rgb(cls, filename):
        """Create class instance from RGB images like JPEG and TIFF."""
        data = np.array(
            Image.open(filename), dtype=np.float32)[::-1, :, :].sum(axis=2)
        return cls(data)

    def read_texture_masks(self, search_string):
        """
        Read texture masks from FITS files and save directly into
        ``self.texture_masks``.

        This method should not be used with the GUI.
        """

        import glob

        for filename in glob.iglob(search_string):
            texture_mask = TextureMask.read(filename)
            texture_type = texture_mask.texture_type
            if texture_type not in self.allowed_textures():
                warnings.warn('{0} is not a valid texture type, '
                              'skipping {1}'.format(texture_type, filename),
                              AstropyUserWarning)
                continue
            self.region_masks[texture_type].append(texture_mask)
            log.info('{0} loaded from {1}'.format(texture_type, filename))

    def save_texture_masks(self, prefix):
        """
        Save (uncropped) texture masks to FITS files.

        The texture masks are resized to match the original image
        size.
        """

        prefixpath, prefixname = os.path.split(prefix)
        for key, reglist in self.region_masks.iteritems():
            #rpath = os.path.join(prefixpath, '_'.join(['region', key]))
            #if not os.path.exists(rpath):
            #    os.mkdir(rpath)
            for i, texture_mask in enumerate(reglist, 1):
                #rname = os.path.join(rpath, '_'.join(
                #    map(str, [prefixname, reg.description, i])) + '.fits')
                filename = '{0}_{1}_{2}.fits'.format(prefixname,
                                                     texture_mask.texture_type,
                                                     i)
                texture_mask.save(filename, self.input_image.shape)

    def _store_peaks(self, key, tab):
        """Store peaks in attribute."""
        tab.keep_columns(['xcen', 'ycen', 'flux'])
        self.peaks[key] = tab

    def find_peaks(self, key, n):
        """Find point sources and store them in ``self.peaks[key]``.

        .. note:: Overwrites :meth:`load_peaks`.

        Parameters
        ----------
        key : {self.clusters_key, self.stars_key}
            Stars or star clusters.

        n : int
            Maximum number of sources allowed.

        """
        tab = find_peaks(np.flipud(self.input_image))[:n]
        self._store_peaks(key, tab)

    def load_peaks(self, key, filename):
        """Load existing point sources and store them in ``self.peaks[key]``.

        .. note:: Overwrites :meth:`find_peaks`.

        Parameters
        ----------
        key : {self.clusters_key, self.stars_key}
            Stars or star clusters.

        filename : str
            ASCII table generated by ``photutils``.

        """
        tab = ascii.read(filename, data_start=1)
        self._store_peaks(key, tab)

    def save_peaks(self, prefix):
        """Save stars and star clusters to text files.

        Coordinates already match original image.
        One output file per table, each named ``<prefix>_<type>.txt``.

        """
        for key, tab in self.peaks.iteritems():
            if len(tab) < 1:
                continue
            tname = '{0}_{1}.txt'.format(prefix, key)
            tab.write(tname, format='ascii')
            log.info('{0} saved'.format(tname))

    @property
    def is_spiralgal(self):
        """Does the model represent a spiral galaxy?"""
        return self._is_spiralgal

    @is_spiralgal.setter
    def is_spiralgal(self, val):
        """Set spiral galaxy property. Also reset layer order."""
        if not isinstance(val, bool):
            raise ValueError('Must be a boolean')
        self._is_spiralgal = val
        self._layer_order = [self.lines_key, self.dots_key, self.small_dots_key]

    @property
    def has_texture(self):
        """Apply textures."""
        return self._has_texture

    @has_texture.setter
    def has_texture(self, value):
        """Set to `True` or `False`."""
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean')
        if not self.has_intensity and not value:
            raise ValueError('Model must have textures or intensity!')
        self._has_texture = value

    @property
    def has_intensity(self):
        """Generate intensity map."""
        return self._has_intensity

    @has_intensity.setter
    def has_intensity(self, value):
        """Set to `True` or `False`."""
        if not isinstance(value, bool):
            raise ValueError('Must be a boolean')
        if not self.has_texture and not value:
            raise ValueError('Model must have textures or intensity!')
        self._has_intensity = value

    @property
    def smooth_key(self):
        """Key identifying regions to smooth."""
        if self.is_spiralgal:
            key = 'remove_star'
        else:
            key = 'smooth'
        return key

    @property
    def small_dots_key(self):
        """Key identifying regions to mark with small dots."""
        if self.is_spiralgal:
            key = 'gas'
        else:
            key = 'dots_small'
        return key

    @property
    def dots_key(self):
        """Key identifying regions to mark with dots."""
        if self.is_spiralgal:
            key = 'spiral'
        else:
            key = 'dots'
        return key

    @property
    def lines_key(self):
        """Key identifying regions to mark with lines."""
        if self.is_spiralgal:
            key = 'disk'
        else:
            key = 'lines'
        return key

    @property
    def clusters_key(self):
        """Key identifying star clusters to be marked."""
        return 'clusters'

    @property
    def stars_key(self):
        """Key identifying stars to be marked."""
        return 'stars'

    @property
    def layer_order(self):
        """Layer ordering, listed by highest priority first."""
        return self._layer_order

    @layer_order.setter
    def layer_order(self, value):
        if self.is_spiralgal:
            raise ValueError('Layer order is fixed for spiral galaxy')
        if set(value) != set(self.layer_order):
            raise ValueError(
                'Layers can be reordered but cannot be added or removed.')
        self._layer_order = value

    def allowed_textures(self):
        """Return a list of allowed texture names."""
        return [self.dots_key, self.small_dots_key, self.lines_key,
                self.smooth_key]

    def texture_names(self):
        """Return existing region texture names, except for the one used
        for smoothing.

        .. note::

            This is targeted at textures with dots and lines,
            where lines belong in the foreground layer by default,
            hence listed first.

        """
        names = sorted(
            self.region_masks, key=lambda x: self.layer_order.index(x)
            if x in self.layer_order else 99, reverse=True)
        if self.smooth_key in names:
            names.remove(self.smooth_key)
        for key in names:
            if len(self.region_masks[key]) < 1:
                names.remove(key)
        return names

    @property
    def preview_intensity(self):
        """Monochrome intensity for GUI preview."""
        if self._preview_intensity is None:
            raise ValueError('Run make() first')
        return np.flipud(self._preview_intensity)

    @property
    def out_image(self):
        """Final result for STL generator."""
        if self._out_image is None:
            raise ValueError('Run make() first')
        return self._out_image

    def save_stl(self, fname, split_halves=True, _ascii=False):
        """Save 3D model to STL file(s)."""
        model = self.out_image

        # Remove any .stl suffix because it is added by to_mesh()
        if fname.lower().endswith('.stl'):
            fname = fname[:-4]

        # Depth is set to 1 here because it is accounted for in make()
        depth = 1
        if split_halves:
            model1, model2 = image_utils.split_image(model, axis='horizontal')
            to_mesh(model1, fname + '_1', depth, self.double_sided, _ascii)
            to_mesh(model2, fname + '_2', depth, self.double_sided, _ascii)
        else:
            to_mesh(model, fname, depth, self.double_sided, _ascii)

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
            masklist = [reg.scaled_mask(self._preproc_img.shape)
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
        fac = self._preproc_img.shape[0] / self.input_image.shape[0]

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
        image, cusp_mask, cusp_texture_flat = self.spiralgalaxy_central_cusp(
            image, disk, cusp_radius=25)
        image = self.emphasize_regs(image, scaled_masks)

        (image, croppedmasks, disk, spiralarms, cusp_mask,
         cusp_texture_flat, clusters, markstars) = self.crop_image(
             image, scaled_masks, scaled_peaks, cusp_mask, cusp_texture_flat)

        image = self.filter_image2(image)

        image = image_utils.normalize(image, True, self.height)
        # Renormalize again so that height is more predictable
        image = image_utils.normalize(image, True, self.height)

        # Generate monochrome intensity for GUI preview
        self._preview_intensity = deepcopy(image.data)

        # To store non-overlapping key-coded texture info
        self._preview_masks = np.zeros(
            self._preview_intensity.shape, dtype='S10')

        if self.has_texture:
            image = self.add_textures(image, croppedmasks, cusp_mask)
            #image = self.add_stars_clusters(image, clusters, markstars)

            # apply stars and star clusters
            if self.has_intensity:
                base_percentile = 75
                depth = 5
            else:
                base_percentile = None
                depth = 10
            image = apply_starlike_textures(
                image, markstars, clusters, depth=depth,
                radius_a=self.clus_r_fac_add, radius_b=self.clus_r_fac_mul,
                base_percentile=base_percentile)

            # For texture-only model, need to add cusp to texture layer
            if not self.has_intensity and cusp_mask is not None:
                self._texture_layer[cusp_mask] = cusp_texture_flat[cusp_mask]

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

    def spiralgalaxy_central_cusp(self, image, disk, cusp_radius=25,
                                  cusp_height=None, cusp_percent=None):
        # Only works for single-disk image.
        # Do this even for smooth intensity map to avoid sharp peak in model.
        #    cusp_radius = 25  # For 1k image

        image = image.copy()
        cusp_mask = None
        cusp_texture_flat = None
        if disk is not None:
            if cusp_height is None:
                cusp_height = 40
            if cusp_percent is None:
                cusp_percent = 10
            log.info('Replacing cusp')
            cusp_texture = replace_cusp(
                image, mask=disk, radius=cusp_radius, height=cusp_height,
                percent=cusp_percent)
            cusp_mask = cusp_texture > 0

            if not self.has_intensity:
                if cusp_height is None:
                    cusp_height = 20
                cusp_texture_flat = replace_cusp(
                    image, mask=disk, radius=cusp_radius, height=cusp_height,
                    percent=cusp_percent)

            image[cusp_mask] = cusp_texture[cusp_mask]
        return image, cusp_mask, cusp_texture_flat

    def emphasize_regs(self, image, scaled_masks):
        log.info('Emphasizing regions')
        image = emphasize_regions(
            image, scaled_masks[self.small_dots_key] +
            scaled_masks[self.dots_key] + scaled_masks[self.lines_key])
        return image

    def crop_image(self, image, scaled_masks, scaled_peaks, cusp_mask,
                   cusp_texture_flat):
        image, iy1, iy2, ix1, ix2 = image_utils.crop_image(image, _max=1.0)
        log.info('Cropped image shape: {0}'.format(image.shape))

        # Also crop masks and lists
        croppedmasks, disk, spiralarms = self._crop_masks(
            scaled_masks, ix1, ix2, iy1, iy2)
        if cusp_mask is not None:
            cusp_mask = cusp_mask[iy1:iy2, ix1:ix2]
        if cusp_texture_flat is not None:
            cusp_texture_flat = cusp_texture_flat[iy1:iy2, ix1:ix2]

        clusters = self._crop_peaks(
            scaled_peaks, self.clusters_key, ix1, ix2, iy1, iy2)
        markstars = self._crop_peaks(
            scaled_peaks, self.stars_key, ix1, ix2, iy1, iy2)

        # Generate list of peaks for GUI preview
        self._final_peaks = {
            self.clusters_key: clusters,
            self.stars_key: markstars}

        return (image, croppedmasks, disk, spiralarms, cusp_mask,
                cusp_texture_flat, clusters, markstars)

    def filter_image2(self, image):
        log.info(
            'Filtering image (second pass, height={0})'.format(self.height))
        image = ndimage.filters.median_filter(image, 10)  # For 1k image
        image = ndimage.filters.gaussian_filter(image, 3)  # Magic?
        image = np.ma.masked_equal(image, 0)

        return image

    def add_textures(self, image, croppedmasks, cusp_mask):
        # Texture layers

        # Dots and lines

        self._texture_layer = np.zeros(image.shape)

        # Apply layers from bottom up
        for layer_key in self.layer_order[::-1]:
            if layer_key == self.dots_key:
                texture_func = DOTS
            elif layer_key == self.small_dots_key:
                texture_func = SMALL_DOTS
                #texture_func = NO_TEXTURE  # Disable small dots
            elif layer_key == self.lines_key:
                texture_func = LINES
            else:
                warnings.warn('{0} is not a valid texture, skipping...'
                              ''.format(layer_key), AstropyUserWarning)
                continue

            log.info('Adding {0}'.format(layer_key))
            for mask in croppedmasks[layer_key]:
                #cur_texture = texture_func(image, mask)
                cur_texture = texture_func(mask)
                self._texture_layer[mask] = cur_texture[mask]
                self._preview_masks[mask] = layer_key

        # Remove cusp from texture and preview
        if cusp_mask is not None:
            self._texture_layer[cusp_mask] = 0
            self._preview_masks[cusp_mask] = ''

        image += self._texture_layer
        return image


    def OLD_add_stars_clusters(self, image, clusters, markstars):
        # Stars and star clusters

        clustexarr = None
        order_w = 10
        order_dw = order_w // 2

        if self.has_intensity:
            h_percentile = 75
            s_height = 5
        else:
            h_percentile = None
            s_height = 10

        # Add star clusters

        n_clus_added = 0

        if len(clusters) > 0:
            maxclusflux = max(clusters['flux'])

            # Sort so that lower cluster height added first
            order_dat = []
            for cluster in clusters:
                ix1 = cluster['xcen'] - order_dw
                ix2 = ix1 + order_w
                iy1 = cluster['ycen'] - order_dw
                iy2 = iy1 + order_w
                order_dat.append(image[iy1:iy2, ix1:ix2].mean())
            clusters.add_column(Column(name='order', data=order_dat))
            clusters.sort('order')

        for cluster in clusters:
            c1 = make_star_cluster(
                image, cluster,  maxclusflux, height=s_height,
                h_percentile=h_percentile, r_fac_add=self.clus_r_fac_add,
                r_fac_mul=self.clus_r_fac_mul, n_craters=3)
            if not np.any(c1):
                continue
            if clustexarr is None:
                clustexarr = c1
            else:
                clustexarr = add_clusters(clustexarr, c1)
            n_clus_added += 1

        log.info('Displaying {0} clusters'.format(n_clus_added))

        # Add individual stars

        n_star_added = 0

        if len(markstars) > 0:
            maxstarflux = max(markstars['flux'])

            # Sort so that lower star height added first
            order_dat = []
            for mstar in markstars:
                ix1 = mstar['xcen'] - order_dw
                ix2 = ix1 + order_w
                iy1 = mstar['ycen'] - order_dw
                iy2 = iy1 + order_w
                order_dat.append(image[iy1:iy2, ix1:ix2].mean())
            markstars.add_column(Column(name='order', data=order_dat))
            markstars.sort('order')

        for mstar in markstars:
            s1 = make_star_cluster(
                image, mstar, maxstarflux, height=s_height,
                h_percentile=h_percentile, r_fac_add=self.star_r_fac_add,
                r_fac_mul=self.star_r_fac_mul, n_craters=1)
            if not np.any(s1):
                continue
            if clustexarr is None:
                clustexarr = s1
            else:
                clustexarr = add_clusters(clustexarr, s1)
            n_star_added += 1

        log.info('Displaying {0} stars'.format(n_star_added))

        # Both stars and star clusters share the same mask

        if clustexarr is not None:
            clustermask = clustexarr != 0
            if self.has_intensity:
                image[clustermask] = clustexarr[clustermask]
            else:
                self._texture_layer[clustermask] = clustexarr[clustermask]

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

        # Resize and save them as TextureMask
        if shape is not None:
            dmask = image_utils.resize_image(dmask, shape[0], width=shape[1])
            sdmask = image_utils.resize_image(sdmask, shape[0], width=shape[1])
        self.region_masks[self.dots_key] = [TextureMask(dmask, self.dots_key)]
        self.region_masks[self.small_dots_key] = [
            TextureMask(sdmask, self.small_dots_key)]

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


def replace_cusp(image, mask=None, radius=20, height=40, percent=10):
    """Replaces the center of the galaxy, which would be
    a sharp point, with a crater.

    Parameters
    ----------
    image : ndarray
        Image array.

    mask : ndarray
        Mask of the disk.

    radius : int
        Radius of the crater in pixels.

    height : int
        Height of the crater.

    percent : float or `None`
        Percentile between 0 and 100, inclusive, used to
        re-adjust height of marker.
        If `None` is given, then no readjustment is done.

    Returns
    -------
    cusp_texture : ndarray
        Crater values to be added.

    """
    cusp_texture = np.zeros(image.shape)

    if mask is None:
        y, x = np.where(image == image.max())
    else:
        a = np.ma.array(image.data, mask=~mask)
        y, x = np.where(a == a.max())

    # LDB:
    # x, y is the index of the central pixel -- not necessarily the
    # center position - round instead?
    x = int(np.mean(x))
    y = int(np.mean(y))

    log.info('\tCenter of galaxy at X={0} Y={1}'.format(x, y))

    # LDB:  handle edges better
    # this will not work because star array does not get trimmed
    ymin = max(y - radius, 0)
    ymax = min(y + radius, image.shape[0])
    xmin = max(x - radius, 0)
    xmax = min(x + radius, image.shape[1])

    if percent is None:
        top = 0.0
    else:
        top = np.percentile(image[ymin:ymax, xmin:xmax], percent)

    # LDB: should check region around center for places where intensity
    # plus disk texture are comparable to cusp height and adjust star
    # height as necessary - NGC3344 star height seems too small
    star = make_star(radius, height)
    smask = star != -1

    # LDB: this is redundant with above
    diam = 2 * radius + 1
    ymax = min(ymin + diam, image.shape[0])
    xmax = min(xmin + diam, image.shape[1])
    # this will not work at edge because star array does not get trimmed
    cusp_texture[ymin:ymax, xmin:xmax][smask] = top + star[smask]

    return cusp_texture


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
        image[minmask] =  image[minmask] * (image[minmask] / _min)

    # Remove low bound
    boolarray = image < threshold
    log.debug('# background pix set to zero: {0}'.format(len(image[boolarray])))
    image[boolarray] = 0

    return image


def make_star(radius, height):
    """Creates a crater-like depression that can be used
    to represent a star.

    Similar to :func:`astro3d.utils.texture.make_star`.

    """
    a = np.arange(radius * 2 + 1)
    x, y = np.meshgrid(a, a)
    r = np.sqrt((x - radius) ** 2 + (y - radius) **2)
    star = height / radius ** 2 * r ** 2
    star[r > radius] = -1
    return star


def make_star_cluster(image, peak, max_intensity, r_fac_add=15, r_fac_mul=5,
                      height=5, h_percentile=75, fil_size=10, n_craters=3):
    """Mark star or star cluster for given position.

    Parameters
    ----------
    image : ndarray

    peak : `astropy.table.Table` row
        One star or star cluster entry.

    max_intensity : float
        Max intensity for all the stars or star clusters.

    r_fac_add, r_fac_mul : number
        Scaling factors to be added and multiplied to
        intensity ratio to determine marker radius.

    height : number
        Height of the marker for :func:`make_star`.

    h_percentile : float or `None`
        Percentile between 0 and 100, inclusive, used to
        re-adjust height of marker.
        If `None` is given, then no readjustment is done.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.

    n_craters : {1, 3}
        Star cluster is marked with ``3``. For single star, use ``1``.

    Returns
    -------
    array : ndarray

    """
    array = np.zeros(image.shape)

    x, y, intensity = peak['xcen'], peak['ycen'], peak['flux']
    radius = r_fac_add + r_fac_mul * intensity / float(max_intensity)
    #log.info('\tcluster radius = {0}'.format(radius, r_fac_add, r_fac_mul))
    star = make_star(radius, height)
    diam = 2 * radius
    r = star.shape[0]
    dr = r / 2
    star_mask = star != -1
    imx1 = max(int(x - diam), 0)
    imx2 = min(int(x + diam), image.shape[1])
    imy1 = max(int(y - diam), 0)
    imy2 = min(int(y + diam), image.shape[0])

    if n_craters == 1:
        centers = [(y, x)]
    else:  # 3
        dy = 0.5 * radius * np.sqrt(3)
        centers = [(y + dy, x), (y - dy, x + radius), (y - dy, x - radius)]

    if h_percentile is None:
        _max = 0.0
    else:
        try:
            _max = np.percentile(image[imy1:imy2, imx1:imx2], h_percentile)
        except ValueError as e:
            warnings.warn('Make star/cluster failed: {0}\n\timage[{1}:{2},'
                          '{3}:{4}]'.format(e, imy1, imy2, imx1, imx2),
                          AstropyUserWarning)
            return array

    for (cy, cx) in centers:
        xx1, xx2, yy1, yy2, sx1, sx2, sy1, sy2 = image_utils.calc_insertion_pos(
            array, star, int(cx - dr), int(cy - dr))
        cur_smask = star_mask[sy1:sy2, sx1:sx2]
        cur_star = star[sy1:sy2, sx1:sx2]
        array[yy1:yy2, xx1:xx2][cur_smask] = _max + cur_star[cur_smask]

    if h_percentile is not None:
        filt = ndimage.filters.maximum_filter(array, fil_size)
        mask = (filt > 0) & (image > filt) & (array == 0)
        array[mask] = filt[mask]

    return array


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
