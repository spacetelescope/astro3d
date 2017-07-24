"""This module defines a RegionMask."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy import log
from astropy.io import fits
from .image_utils import resize_image_absolute


class RegionMask(object):
    """
    This class defines a region mask.
    """

    def __init__(self, mask, mask_type, required_shape=None, shape=None):
        """
        Parameters
        ----------
        mask : array-like (bool)
            A 2D boolean image defining the region mask.  For texture
            masks, the texture will be applied where the ``mask`` is
            `True`.

        mask_type : str
            The type of mask.  Some examples include 'dots', 'small_dots',
            'lines', 'gas', and 'spiral'.

        required_shape : tuple, optional
            If not `None`, then the ``(ny, nx)`` shape required for the
            input mask.

        shape : tuple, optional
            If not `None`, then the input mask will be resized to
            ``shape``.
        """

        self.mask = np.asanyarray(mask)
        self.mask_type = mask_type

        if required_shape is not None:
            if self.mask.shape != required_shape:
                raise ValueError('Input mask does not have the correct '
                                 'shape.')

        if shape is not None:
            self.mask = resize_image_absolute(self.mask, x_size=shape[1],
                                              y_size=shape[0])

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
            ``shape``.  This is used to save the mask with the same size
            as the original input image.
        """

        mask = self.mask
        if shape is not None:
            mask = resize_image_absolute(self.mask, x_size=shape[1],
                                         y_size=shape[0])

        header = fits.Header()
        header['MASKTYPE'] = self.mask_type
        hdu = fits.PrimaryHDU(data=mask.astype(np.int32), header=header)
        hdu.writeto(filename)
        log.info('Saved {0} (mask type="{1}").'.format(filename,
                                                       self.mask_type))

    @classmethod
    def from_fits(cls, filename, required_shape=None, shape=None):
        """
        Create a `RegionMask` instance from a FITS file.

        The FITS file must have a 'MASKTYPE' header keyword defining the
        mask type.  This keyword must be in the primary extension.

        The mask data should contain only ones or zeros, which will be
        converted to `True` and `False` values, respectively.

        Parameters
        ----------
        filename : str
            The input FITS filename.

        required_shape : tuple, optional
            If not `None`, then the ``(ny, nx)`` shape required for the
            input mask.

        shape : tuple, optional
            If not `None`, then the input mask will be resized to
            ``shape``.

        Returns
        -------
        result : `RegionMask`
            A `RegionMask` instance.
        """

        fobj = fits.open(filename)
        header = fobj[0].header
        mask = fobj[0].data.astype(np.bool)
        mask_type = header['MASKTYPE']
        region_mask = cls(mask, mask_type, required_shape=required_shape,
                          shape=shape)
        return region_mask
