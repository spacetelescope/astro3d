"""
This module provides tools to produce texture samples.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.io import fits
from astropy import log
from . import textures


def save_image(data, filename):
    """
    Save an image to a bitmap (e.g. TIFF, JPEG, PNG) or a FITS file.

    The type of file is determined by the filename suffix.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data array to save.

    filename : str
        The output filename.
    """

    if isinstance(data, np.ma.core.MaskedArray):
        data = data.data

    suffix = filename.split('.')[-1].lower()
    if suffix in ('fit', 'fits'):
        fits.writeto(filename, data, clobber=True)
    else:
        from scipy.misc import imsave   # uses PIL/Pillow
        imsave(filename, np.flipud(data))    # flip to make origin lower-left
    log.info('Saved {0}.'.format(filename))


def textures_to_jpeg():
    """
    Generate some textures and save them to JPEG images.

    Examples
    --------
    >>> from astro3d import texture_samples
    >>> texture_samples.textures_to_jpeg()
    """

    shape = (200, 200)
    size = [15, 10, 6, 3]        # line thickness or dot diameter
    spacing = [25, 15, 10, 5]    # line spacing or dot grid spacing

    for sz, sp in zip(size, spacing):
        log.info('{0} {1}'.format(sz, sp))
        for profile in ['spherical', 'linear']:
            log.info('\t{0}'.format(profile))

            lim = textures.lines_texture(shape, profile, sz, 1., sp,
                                         orientation=0.)
            fn = ('lines_{0}_thickness{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save_image(lim, fn)

            rlim = lim.transpose()
            lim[rlim > lim] = rlim[rlim > lim]
            fn = ('hatch_{0}_thickness{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save_image(lim, fn)

            sdim = textures.dots_texture(shape, profile, sz, 1.,
                                         textures.square_grid(shape, sp))
            fn = ('dots_squaregrid_{0}_diameter{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save_image(sdim, fn)

            hdim = textures.dots_texture(shape, profile, sz, 1,
                                         textures.hexagonal_grid(shape, sp))
            fn = ('dots_hexagonalgrid_{0}_diameter{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save_image(hdim, fn)
