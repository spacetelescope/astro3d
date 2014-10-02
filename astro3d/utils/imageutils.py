"""Utility functions for image manipulation."""
from __future__ import division, print_function

# Anaconda
import numpy as np
import scipy
from astropy import log
from PIL import Image


def im2file(im, filename):
    """Save image in TIFF or JPEG format.

    Parameters
    ----------
    im : ndarray
        Input image.

    filename : str
        Output filename.

    """
    im = im[::-1,:]
    cim = np.zeros(im.shape,dtype=np.uint8)
    cim[:] = (255*im/im.max()).astype(np.uint8)
    pim = Image.fromarray(cim)
    pim.save(filename)
    log.info('{0} saved'.format(filename))


def img2array(filename):
    """Turns an image into a Numpy array.

    .. note::

        Requires PIL (Python Imaging Library) or Pillow (a PIL fork).

    Parameters
    ----------
    filename : str
        Input filename. For example, a JPEG file.

    Returns
    -------
    array : ndarray
        Image array.

    """
    img = Image.open(filename)
    array = np.array(img, dtype=np.float32)
    if array.ndim == 3:
        array = array.sum(2)
    return array


def compressImage(image, height):
    """Compress the image to a given size.
    Given that 3D printing cannot handle fine resolution,
    any loss of resolution is ultimately unimportant.

    Parameters
    ----------
    image : ndarray
        Input image.

    height : int
        Desired height. Width is adjusted according to
        input aspect ratio.

    Returns
    -------
    array : ndarray
        Resized image.

    """
    h, w = image.shape
    width = int(w * height / float(h))

    array = scipy.misc.imresize(image, (height, width))

    #array = np.zeros((height, width))
    #y_step = h / float(height)
    #x_step = w / float(width)
    #for y in range(height):
    #    for x in range(width):
    #        array[y, x] = image[y * y_step, x * x_step]

    return array


def crop_image(image, _max=0):
    """Crop boundaries of image where maximum value is
    less than the given value.

    Parameters
    ----------
    image : ndarray
        Image array to process.

    _max : float
        Crop pixels below this value.

    Returns
    -------
    image : ndarray
        Cropped image.

    iy1, iy2, ix1, ix2 : int
        Indices of input image for cropping other components.

    """
    locations = np.where(image > _max)
    iy1 = min(locations[0])
    ymax = max(locations[0])
    iy2 = ymax + 1
    ix1 = min(locations[1])
    xmax = max(locations[1])
    ix2 = xmax + 1

    return image[iy1:iy2, ix1:ix2], iy1, iy2, ix1, ix2


def normalize(array, norm, height=255.):
    """Taken, with some slight modifications, from ``qimage2ndarray``.

    As ``qimage2ndarray`` is a third-party package and has
    SIP and PyQt4 dependency, it is simpler to copy this
    method.

    See http://hmeine.github.io/qimage2ndarray/ for more information.

    Parameters
    ----------
    array : ndarray
        Input array.

    norm
        Used to normalize an image to ``0..height``:
            * ``(nmin, nmax)`` - Scale and clip values from
              ``nmin..nmax`` to ``0..height``
            * ``nmax`` - Scale and clip the range
              ``0..nmax`` to ``0..height``
            * `True` - Scale image values to ``0..height``
            * `False` - No scaling

    height : float
        Max value of scaled image.

    Returns
    -------
    array : ndarray
        Scaled array.

    """
    if not norm:
        return array

    if norm is True:
        norm = array.min(), array.max()
    elif np.isscalar(norm):
        norm = (0, norm)

    nmin, nmax = norm
    array = array - nmin
    array = array * height / float(nmax - nmin)

    return array


def split_image(image):
    """Split image array into two halves.
    If image shape is a rectangle, splitting is done
    on the shorter edge.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    image1, image2 : ndarray

    """
    if image.shape[0] > image.shape[1]:
        mid = int(image.shape[1] / 2)
        image1 = image[:, :mid]

        if image.shape[1] % 2 == 0:
            image2 = image[:, mid:]
        else:
            image2 = image[:, mid:-1]

    else:
        mid = int(image.shape[0] / 2)
        image1 = image[:mid]

        if image.shape[0] % 2 == 0:
            image2 = image[mid:]
        else:
            image2 = image[mid:-1]

    return image1, image2
