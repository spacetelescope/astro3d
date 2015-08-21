"""
Utility functions for image manipulation.
These functions are used only by the GUI.
"""
from __future__ import division, print_function
import numpy as np
from PIL import Image
from PyQt4.QtCore import Qt
from astropy import log

import qimage2ndarray as q2a
from astropy.visualization import (PercentileInterval, LinearStretch,
                                   LogStretch, SqrtStretch)


scale_linear = LinearStretch() + PercentileInterval(99.)
scale_log = LogStretch() + PercentileInterval(99.)
scale_sqrt = SqrtStretch() + PercentileInterval(99.)


def makeqimage(nparray, transformation, size):
    """Performs various transformations (linear, log, sqrt, etc.)
    on the image. Clips and scales pixel values between 0 and 255.
    Scales and inverts the image. All transformations are
    non-destructive (performed on a copy of the input array).

    Parameters
    ----------
    nparray : ndarray

    transformation : func or `None`

    size : QSize

    Returns
    -------
    qimage : QImage

    """
    npimage = nparray.copy()
    npimage[npimage < 0] = 0

    if transformation is not None:
        npimage = q2a._normalize255(transformation(npimage), True)
        qimage = q2a.array2qimage(npimage, (0, 255))
    else:
        qimage = q2a.array2qimage(npimage, True)

    qimage = qimage.scaled(size, Qt.KeepAspectRatio)
    qimage = qimage.mirrored(False, True)

    return qimage


def scale_rgb(image, rgb_scaling=None):
    """Scale RGB values in given image cube.

    Parameters
    ----------
    image : array_like
        RGB image cube.

    rgb_scaling : tuple or `None`
        Scaling factors for red, green, and blue components, in that order.
        If `None`, no change.

    Returns
    -------
    scaled_image : array_like
        Scaled RGB image cube.

    """
    rgb_ndim = 3

    if image.ndim != rgb_ndim:
        raise ValueError('Input must be RGB cube.')

    if rgb_scaling is None:
        return image
    elif len(rgb_scaling) != rgb_ndim:
        raise ValueError('Must have one scaling factor for each RGB color')

    log.info('RGB scaling is {0}'.format(rgb_scaling))
    scaled_image = np.empty_like(image)
    for i in range(rgb_ndim):
        scaled_image[:, :, i] = rgb_scaling[i] * image[:, :, i]

    return scaled_image


def img2array(filename, rgb_scaling=None):
    """Turns an image into a Numpy array.

    .. note::

        Requires PIL (Python Imaging Library) or Pillow (a PIL fork).

    Parameters
    ----------
    filename : str
        Input filename. For example, a JPEG file.

    rgb_scaling
        See :func:`scale_rgb`.

    Returns
    -------
    array : ndarray
        Image array.

    """
    img = np.array(Image.open(filename), dtype=np.float32)
    ndim = img.ndim

    if ndim == 3:
        scaled_img = scale_rgb(img, rgb_scaling=rgb_scaling)
        array = scaled_img.sum(axis=2)

    elif ndim == 2:
        array = img

    else:
        raise ValueError('Invalid image ndim={0}'.format(ndim))

    return array


def calc_insertion_pos_1d(array, subarray, x_beg):
    """Calculate indices for inserting subarray into 1D array,
    cropping at edges.

    Parameters
    ----------
    array, subarray : array_like
        Array and subarray data.

    x_beg : int
        Starting index of array for insertion.

    Returns
    -------
    xx1, xx2, sx1, sx2 : int
        Indices such that ``array[xx1:xx2] = subarray[sx1:sx2]``
        will not give ``IndexError``.

    """
    xsize = array.shape[0]
    sx1 = 0
    sx2 = subarray.shape[0]
    xx1 = x_beg
    xx2 = xx1 + sx2

    if xx1 < 0:
        sx1 = -xx1
        xx1 = 0
    if xx2 > xsize:
        sx2 -= (xx2 - xsize)
        xx2 = xsize

    return xx1, xx2, sx1, sx2


def calc_insertion_pos(image, subarray, x_beg, y_beg):
    """Calculate indices for inserting subarray into image,
    cropping at edges.

    Parameters
    ----------
    image, subarray : array_like
        Image and subarray data.

    x_beg, y_beg : int
        Indices of lower left corner of image for insertion.

    Returns
    -------
    xx1, xx2, yy1, yy2, sx1, sx2, sy1, sy2 : int
        Indices such that
        ``image[yy1:yy2, xx1:xx2] = subarray[sy1:sy2, sx1:sx2]``
        will not give ``IndexError``.

    """
    xx1, xx2, sx1, sx2 = calc_insertion_pos_1d(
        image[0, :], subarray[0, :], x_beg)
    yy1, yy2, sy1, sy2 = calc_insertion_pos_1d(
        image[:, 0], subarray[:, 0], y_beg)
    return xx1, xx2, yy1, yy2, sx1, sx2, sy1, sy2


# http://mail.scipy.org/pipermail/numpy-discussion/2011-January/054470.html
def circular_mask(arr_shape, r, xcen, ycen):
    """Generate circular mask for 2D image.

    Parameters
    ----------
    arr_shape : tuple of int
        Shape of the array to use the mask.

    r : int
        Radius of the mask in pixels.

    xcen, ycen : int
        Position of mask center.

    Returns
    -------
    mask : array_like
        Boolean array of the mask.

    """
    ny, nx = arr_shape
    xcen = np.around(xcen).astype('int')
    ycen = np.around(ycen).astype('int')

    x1, x2 = xcen - r, xcen + r
    y1, y2 = ycen - r, ycen + r

    y, x = np.ogrid[-r:r, -r:r]
    circle = (x ** 2 + y ** 2) <= (r ** 2)

    if y1 >= 0 and y2 < ny and x1 >= 0 and x2 < nx:
        # Mask contained in image.
        mask = np.zeros(arr_shape).astype('bool')

        # populate array with mask position
        mask[y1:y2, x1:x2][circle] = True

    else:
        # Mask falls outside image bounds.
        # compute number of pixels mask extends beyond current array size
        xout = np.abs(min(0 - x1, nx - x2)) + 1
        yout = np.abs(min(0 - y1, ny - y2)) + 1

        # derive size of new array that will contain entire mask
        ny_new = ny + yout * 2
        nx_new = nx + xout * 2
        # initialize larger array
        a = np.zeros((ny_new, nx_new)).astype('bool')

        # recompute positions relative to larger array size
        xcen = np.round(xcen + 0.5 * (nx_new - nx)).astype('int')
        ycen = np.round(ycen + 0.5 * (ny_new - ny)).astype('int')
        x1, x2 = xcen - r, xcen + r
        y1, y2 = ycen - r, ycen + r

        # populate array with mask position
        a[y1:y2, x1:x2][circle] = True

        mask = a[yout:-yout, xout:-xout].copy()

    return mask


def dot(v1, v2):
    """Vector dot product."""
    return (v1 * v2).sum(axis=-1)


def in_triangle(p, p1, p2, p3):
    """Test if a point is inside a triangle.
    All points are defined by ``(y, x)``.

    Parameters
    ----------
    p : tuple
        Point to test.

    p1, p2, p3 : tuple
        Points that define the triangle.

    Returns
    -------
    ans : bool
        `True` if point is inside.

    """
    v1 = np.reshape(np.array([p3[0] - p1[0], p3[1] - p1[1]]), (1, 2))
    v2 = np.reshape(np.array([p2[0] - p1[0], p2[1] - p1[1]]), (1, 2))
    v3 = p - np.reshape(np.array(p1), (1, 2))
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)
    dot13 = dot(v1, v3)
    dot22 = dot(v2, v2)
    dot23 = dot(v2, v3)
    fac = 1.0 / (dot11 * dot22 - dot12 * dot12)
    u = (dot22 * dot13 - dot12 * dot23) * fac
    v = (dot11 * dot23 - dot12 * dot13) * fac
    return (u >= 0) & (v >= 0) & ((u + v) < 1)


def in_rectangle(p, p1, p2, radius):
    """Like :func:`in_triangle` but for a quadrilateral.

    Parameters
    ----------
    p : tuple
        Point to test.

    p1, p2 : tuple
        Points that define the orientation and length of one side,
        but bisects the rectangle.

    radius : int
        Defines the other dimension as
        :math:`2 \\times \\textnormal{radius} + 1`.

    Returns
    -------
    ans : bool
        `True` if point is inside.

    """
    # Compute corners of rectangle.
    # First, determine orthogonal unit vector.
    unit_base = np.array(p2) - np.array(p1)
    denom = np.sqrt((unit_base ** 2).sum())
    if denom == 0:
        return None
    unit_base = unit_base / denom
    unit_orth = np.array([unit_base[1], -unit_base[0]])
    rad = radius + 0.5

    # define rectangle points
    r1 = p1 + rad * unit_orth
    r2 = p2 + rad * unit_orth
    r3 = p2 - rad * unit_orth
    r4 = p1 - rad * unit_orth

    return in_triangle(p, r1, r2, r3) | in_triangle(p, r1, r3, r4)
