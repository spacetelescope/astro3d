"""This module provides image (2D array) utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
import warnings
from PIL import Image
import numpy as np
from astropy.convolution import convolve
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning


def remove_nonfinite(data):
    """
    Remove non-finite values (e.g. NaN, inf, etc.) from an array.

    Parameters
    ----------
    data : array-like
        The input data array.

    Returns
    -------
    result : `~numpy.ndarray`
        The array with non-finite values removed.
    """

    mask = ~np.isfinite(data)

    if np.any(mask):
        # use astropy's convolve as a 5x5 mean filter that ignores nans
        # (in regions that are smaller than 5x5)
        data_out = deepcopy(np.asanyarray(data))
        data_out[mask] = np.nan
        filt = np.ones((5, 5))
        data_conv = convolve(data_out, filt) / filt.sum()
        data_out[mask] = data_conv[mask]

        # if there any non-finite values left (e.g. contiguous non-finite
        # regions larger than the filter size), then simply set them to zero.
        # For example, RGB FITS files appear to assign nan to large regions
        # of zero weight (as a coverage mask).
        data_out[~np.isfinite(data_out)] = 0.
        return data_out
    else:
        return data


def resize_image(data, scale_factor):
    """
    Resize a 2D array by the given scale factor.

    The array is resized by the same factor in each dimension,
    preserving the original aspect ratio.

    Given that 3D printing cannot handle fine resolution, any loss of
    resolution is ultimately unimportant.

    Parameters
    ----------
    data : array-like
        The 2D array to be resized.

    scale_factor : float
        The scale factor to apply to the image.

    Returns
    -------
    result : `~numpy.ndarray`
        The resized array.
    """

    data = np.asanyarray(data)
    ny, nx = data.shape

    if scale_factor == 1:
        log.info('The array (ny * nx) = ({0}x{1}) was not '
                 'resized.'.format(ny, nx))
        return data

    ny, nx = data.shape
    if (float(ny) / nx) >= 1.5:
        warnings.warn('The image is >= 1.5x taller than wide.  For 3D '
                      'printing, it should be rotated such that the longest '
                      'axis is in the x direction.', AstropyUserWarning)

    y_size = int(round(ny * scale_factor))
    x_size = int(round(nx * scale_factor))
    data = np.array(Image.fromarray(data.astype(float)).resize(
        (x_size, y_size)), dtype=data.dtype)
    # from scipy.misc import imresize
    # data = imresize(data, (y_size, x_size)).astype(data.dtype)

    log.info('The array was resized from {0}x{1} to {2}x{3} '
             '(ny * nx).'.format(ny, nx, y_size, x_size))

    return data


def normalize_data(data, max_value=1.):
    """
    Normalize an array such that its values range from 0 to
    ``max_value``.

    Parameters
    ----------
    data : array-like
        The input data array.

    max_value : float, optional
        The maximum value of the normalized array.

    Returns
    -------
    result : `~numpy.ndarray`
        The normalized array.
    """

    data = np.asanyarray(data)
    minval, maxval = np.min(data), np.max(data)
    if (maxval - minval) == 0:
        return (data / maxval) * max_value
    else:
        return (data - minval) / (maxval - minval) * max_value


def crop_below_threshold(data, threshold=0):
    """
    Calculate a slice tuple to crop an array where its values
    are less than ``threshold``.

    Parameters
    ----------
    data : array-like
        The input data array.

    threshold : float, optional
        The values equal to and below which to crop from the array.

    Returns
    -------
    result : tuple of slice objects
        The slice tuple that can be used to crop the array.

    Examples
    --------
    >>> data = np.zeros((100, 100))
    >>> data[40:50, 40:50] = 100
    >>> slc = crop_below_threshold(data, 10)
    >>> slc
    (slice(40, 50, None), slice(40, 50, None))
    >>> data_cropped = data[slc]
    """

    idx = np.where(data > threshold)
    y0, y1 = min(idx[0]), max(idx[0]) + 1
    x0, x1 = min(idx[1]), max(idx[1]) + 1
    return (slice(y0, y1), slice(x0, x1))


def combine_masks(masks):
    """
    Combine boolean masks into a single mask.

    Parameters
    ----------
    masks : list of boolean `~numpy.ndarray`
        A list of boolean `~numpy.ndarray` masks.

    Returns
    -------
    mask : bool `~numpy.ndarray`
        The combined mask.
    """

    nmasks = len(masks)
    if nmasks == 0:
        return None
    elif nmasks == 1:
        return masks[0]
    else:
        return reduce(lambda mask1, mask2: np.logical_or(mask1, mask2), masks)


def combine_region_masks(region_masks):
    """
    Combine a list of `~astro3d.region_mask.RegionMask` into a single
    mask.

    Parameters
    ----------
    region_masks : list of `~astro3d.region_mask.RegionMask`
        A list of boolean `~astro3d.region_mask.RegionMask` masks.

    Returns
    -------
    mask : bool `~numpy.ndarray`
        The combined mask.
    """

    nmasks = len(region_masks)
    if nmasks == 0:
        return region_masks
    else:
        return reduce(
            lambda mask, regm2: np.logical_or(mask, regm2.mask),
            region_masks[1:],
            region_masks[0].mask
        )


def radial_distance(shape, position):
    """
    Return an array where the pixel values are the Euclidean distance of
    the pixel from a given position.

    Parameters
    ----------
    shape : tuple
        The ``(ny, nx)`` shape of the output array.

    position : tuple
        The ``(y, x)`` position corresponding to zero distance.

    Returns
    -------
    result : `~numpy.ndarray`
        A 2D array of given ``shape`` representing the radial distance
        map.
    """

    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)


def radial_weight_map(shape, position, alpha=0.8, r_min=100, r_max=450,
                      fill_value=0.1):
    """
    Return a radial weight map used to enhance the faint spiral arms in
    the outskirts of a galaxy image.

    Parameters
    ----------
    shape : tuple
        The ``(ny, nx)`` shape of the output array.

    position : tuple
        The ``(y, x)`` position corresponding to zero distance.

    alpha : float, optional
        The power scaling factor applied to the radial distance.

    r_min : int, optional
        Minimum pixel radius below which the weights are truncated.

    r_max : int, optional
        Maximum pixel radius above which the weights are truncated.

    fill_value : float, optional
       Value to replace weights that were calculated to be zero.

    Returns
    -------
    result : `~numpy.ndarray`
        A 2D array of given ``shape`` representing the radial weight
        map.
    """

    r = radial_distance(shape, position)
    r2 = r ** alpha
    min_mask = (r < r_min)
    max_mask = (r > r_max)
    r2[min_mask] = r2[min_mask].min()
    r2[max_mask] = r2[max_mask].max()
    r2 /= r2.max()
    r2[r2 == 0] = fill_value
    return r2


def split_image(data, axis=0):
    """
    Split an image into two (nearly-equal) halves.

    If the split axis has an even number of elements, then the image
    will be split into two equal halves.

    Parameters
    ----------
    data : array-like
        The input data array.

    axis : int, optional
        The axis to split (e.g. ``axis=0`` splits the y axis).

    Returns
    -------
    result1, result2 : `~numpy.ndarray`
        The split arrays.  For ``axis=0`` the returned order is
        ``(bottom, top)``.  For ``axis=1`` the returned order is
        ``(left, right)``.
    """

    ny, nx = data.shape
    if axis == 0:
        hy = int(ny / 2.)
        data1 = data[:hy, :]
        data2 = data[hy:, :]
    elif axis == 1:
        hx = int(nx / 2.)
        data1 = data[:, :hx]
        data2 = data[:, hx:]
    else:
        raise ValueError('Invalid axis={0}'.format(axis))

    return data1, data2
