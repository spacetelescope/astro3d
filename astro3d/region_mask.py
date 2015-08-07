"""This module defines a RegionMask."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy import log
from astropy.io import fits
from .image_utils import resize_image


class RegionMask(object):
    """
    This class defines a region mask.
    """

    def __init__(self, mask, mask_type):
        """
        Parameters
        ----------
        mask : array-like (bool)
            A 2D boolean image defining the region mask.  For texture
            masks, the texture will be applied where the ``mask`` is
            `True`.

        mask_type : str
            The type of mask.  Some examples include 'dots', 'small_dots',
            'lines', 'gas', 'spiral'.
        """

        self.mask = np.asanyarray(mask)
        self.mask_type = mask_type

    def resize(self, shape):
        """
        Return the region mask resized to the given ``shape``.

        Parameters
        ----------
        shape : tuple
            Desired shape of the region mask.

        Returns
        -------
        mask : `~numpy.ndarray`
            The resized region mask.
        """

        return resize_image(self.mask, shape[0], width=shape[1])

    def write(self, filename, shape=None):
        """
        Write the region mask to a FITS file.

        Mask `True` and `False` values will be saved as 1 and 0,
        respectively.

        Parameters
        ----------
        filename : str
            The output filename.

        shape : tuple
            If not `None`, then the region mask will be resized to
            ``shape``.  This is used to save the mask as the same size
            of original input image.
        """

        if shape is not None:
            mask = self.resize(shape)
        else:
            mask = self.mask

        header = fits.Header()
        header['MASKTYPE'] = self.mask_type
        hdu = fits.PrimaryHDU(data=mask.astype(np.int32), header=header)
        hdu.writeto(filename)
        log.info('Saved {0} (mask type="{1}").'.format(filename,
                                                       self.mask_type))

    @classmethod
    def from_fits(cls, filename, shape=None):
        """
        Create a `RegionMask` instance from a FITS file.

        The FITS file must have a 'MASKTYPE' header keyword defining the
        mask type.  The mask must be in the primary extension.

        The mask data should contain only ones or zeros, which will be
        converted to `True` and `False` values, respectively.

        Parameters
        ----------
        filename : str
            The input FITS filename.

        shape : tuple
            If not `None`, then the input mask will be resized to
            ``shape``.  This is used to resize the region mask to the
            same size of the (smaller) working image (e.g used in the
            GUI, etc.).

        Returns
        -------
        result : `RegionMask`
            A `RegionMask` instance.
        """

        fobj = fits.open(filename)
        header = fobj[0].header
        mask = fobj[0].data.astype(np.bool)

        if shape is not None:
            mask = resize_image(mask, shape[0], width=shape[1])

        mask_type = header['MASKTYPE']
        region_mask = cls(mask, mask_type)
        log.info('Read {0} (mask type="{1}").'.format(filename, mask_type))

        return region_mask
