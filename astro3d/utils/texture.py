"""
This module defines and adds textures (repeating or random) to an image.
Spacings, thickness, and diameters are specified in pixels.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
from functools import partial
import warnings
import numpy as np
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
from scipy import ndimage
from . import imutils


def square_grid(shape, spacing, offset=0):
    """
    Generate ``(x, y)`` coordinates for a regular square grid over a
    given image shape.

    Parameters
    ----------
    shape : tuple
        The shape of the image over which to create the grid.

    spacing : float
        The spacing in pixels between the centers of adjancent squares.
        This is also the square size.

    offset : float, optional
        An optional offset to apply in both the ``x`` and ``y``
        directions from the nominal starting position of ``(0, 0)``.

    Returns
    -------
    coords : `~numpy.ndarray`
        A ``N x 2`` array where each row contains the ``x`` and ``y``
        coordinates of the square centers.
    """

    y, x = np.mgrid[offset:shape[0]:spacing, offset:shape[1]:spacing]
    return np.transpose(np.vstack([x.ravel(), y.ravel()]))


def hexagonal_grid(shape, spacing, offset=0):
    """
    Generate ``(x, y)`` coordinates for a hexagonal grid over a given
    image shape.

    Parameters
    ----------
    shape : tuple
        The shape of the image over which to create the grid.

    spacing : float
        The spacing in pixels between the centers of adjacent hexagons.
        This is also the "size" of the hexagon, as measured perpendicular
        from a side to the opposite side.

    offset : float, optional
        An optional offset to apply in both the ``x`` and ``y``
        directions from the nominal starting position of ``(0, 0)``.

    Returns
    -------
    coords : `~numpy.ndarray`
        A ``N x 2`` array where each row contains the ``x`` and ``y``
        coordinates of the hexagon centers.
    """

    x_spacing = 2. * spacing / np.sqrt(3.)
    y, x = np.mgrid[offset:shape[0]:spacing, offset:shape[1]:x_spacing]
    # shift the odd rows by half of the x_spacing
    for i in range(1, len(x), 2):
        x[i] += 0.5 * x_spacing
    return np.transpose(np.vstack([x.ravel(), y.ravel()]))


def random_points(shape, spacing):
    """
    Generate ``(x, y)`` coordinates at random positions over a given
    image shape.

    Parameters
    ----------
    shape : tuple
        The shape of the image over which to create the random points.

    spacing : float
        The "average" spacing between the random positions.
        Specifically, ``spacing`` defines the number of random positions
        as ``shape[0] * shape[1] / spacing**2``.

    Returns
    -------
    coords : `~numpy.ndarray`
        A ``N x 2`` array where each row contains the ``x`` and ``y``
        coordinates of the random positions.
    """

    npts = shape[0] * shape[1] / spacing**2
    x = np.random.random(npts) * shape[1]
    y = np.random.random(npts) * shape[0]
    return np.transpose(np.vstack([x, y]))


def lines_texture(shape, profile, thickness, spacing, scale, orientation=0.):
    """
    Create a texture consisting of regularly-spaced set of lines.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The line profile. ``'linear'`` produces a "^"-shaped line
        profile.  ``'spherical'`` produces a rounded cylindrical or
        elliptical profile.  See ``scale`` for more details.

    thickness : int
        Thickness of the line over the entire profile.

    spacing : int
        Perpendicular spacing between adjacent line centers.

    scale : float
        The scale factor applied to the line.  If ``scale`` is 1, then
        the line height is half the ``thickness``.

        For a ``'spherical'`` profile, ``scale=1`` produces a
        hemispherical profile perpendicular to the line.  If ``scale``
        is not 1, then the profile is elliptical.

    orientation : float, optional
        The counterclockwise rotation angle in degrees.  The default
        ``orientation`` of 0 degrees corresponds to horizontal lines.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the line texture.
    """

    # start in center of the image and then offset lines both ways
    xc = shape[1] / 2
    yc = shape[1] / 2
    x = np.arange(shape[1]) - xc
    y = np.arange(shape[0]) - yc
    xp, yp = np.meshgrid(x, y)

    angle = np.pi * orientation/180.
    s, c = np.sin(angle), np.cos(angle)
    # x = c*xp + s*yp    # unused
    y = -s*xp + c*yp

    # compute maximum possible offsets
    noffsets = int(np.sqrt(xc**2 + yc**2) / spacing)
    offsets = spacing * (np.arange(noffsets*2 + 1) - noffsets)

    # loop over all offsets
    data = np.zeros(shape)
    h_thick = thickness / 2.
    for offset in offsets:
        y_diff = y - offset
        idx = np.where((y_diff > -h_thick) & (y_diff < h_thick))
        if idx:
            if profile == "spherical":
                data[idx] = scale * np.sqrt(h_thick**2 - y_diff[idx]**2)
            elif profile == "linear":
                data[idx] = scale * (h_thick - np.abs(y_diff[idx]))
    return data


def dots_texture(shape, profile, diameter, scale, locations):
    """
    Create a texture consisting of dots at the given locations.

    If any dots overlap (e.g. the location separations are smaller than
    the dot size), then the greater value of the two is taken, not the
    sum.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The dot profile. ``'linear'`` produces a cone-shaped dot
        profile.  ``'spherical'`` produces a hemispherical or half
        ellipsoid dot profile.  See ``scale`` for more details.

    diameter : int
        The diameter of the dot.

    scale : float
        The scale factor applied to the dot.  If ``scale`` is 1, then
        the dot height is half the ``diameter``.

        For a ``'spherical'`` profile, ``scale=1`` produces a
        hemispherical dot.  If ``scale`` is not 1, then the dot profile
        is a half ellipsoid (circular base with a stretched height).

    locations : `~numpy.ndarray`
        A ``Nx2`` `~numpy.ndarray` where each row contains the ``x`` and
        ``y`` coordinate positions.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the dot texture.

    Examples
    --------
    >>> shape = (1000, 1000)
    >>> dots('linear', shape, 7, 3, locations=hex_grid(shape, 10))
    """

    dot_size = diameter + 1
    dot_shape = (dot_size, dot_size)
    dot = np.zeros(dot_shape)
    y, x = np.indices(dot_shape)
    radius = diameter / 2
    r = np.sqrt((x - radius)**2 + (y - radius)**2)
    idx = np.where(r < radius)

    if profile == 'spherical':
        dot[idx] = scale * np.sqrt(radius**2 - r[idx]**2)
    elif profile == 'linear':
        dot[idx] = scale * np.abs(radius - r[idx])
    else:
        raise ValueError('profile must be "spherical" or "linear"')

    data = np.zeros(shape)
    for (x, y) in locations:
        # exclude points too close to the edge
        if not (x < radius or x > (shape[1] - radius - 1) or
                y < radius or y > (shape[0] - radius - 1)):
            # replace pixels in the output image only if they are larger
            # in the dot (i.e. the pixels are not summed, but are
            # assigned the greater value of the new dot and the image)
            region = data[y-radius:y+radius+1, x-radius:x+radius+1]
            mask = (dot > region)
            region[mask] = dot[mask]
    return data


def star_texture(radius, height):
    """
    Create a texture representing a single star.

    The texture is a parabolic "bowl" with a circular base of given
    ``radius`` and given ``height`` (pictorially like "_|U|_").

    Parameters
    ----------
    radius : int
        The circular radius of the texture.

    height : int
        The height of the texture.

    Returns
    -------
    data : `~numpy.ndimage`
        An image containing the star texture.
    """

    x = np.arange(2.*radius + 1) - radius
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2, + yy**2)
    data = height * (r / radius)**2
    data[r > radius] = 0
    # data[r > radius] = -1     # currently used version
    return data


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
    # log.info('\tcluster radius = {0}'.format(radius, r_fac_add, r_fac_mul))
    # star = make_star(radius, height)
    star = star_texture(radius, height)
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
        xx1, xx2, yy1, yy2, sx1, sx2, sy1, sy2 = imutils.calc_insertion_pos(
            array, star, int(cx - dr), int(cy - dr))
        cur_smask = star_mask[sy1:sy2, sx1:sx2]
        cur_star = star[sy1:sy2, sx1:sx2]
        array[yy1:yy2, xx1:xx2][cur_smask] = _max + cur_star[cur_smask]

    if h_percentile is not None:
        filt = ndimage.filters.maximum_filter(array, fil_size)
        mask = (filt > 0) & (image > filt) & (array == 0)
        array[mask] = filt[mask]

    return array


def add_clusters(input_cluster1, cluster2):
    """Add two star clusters together.

    Parameters
    ----------
    input_cluster1, cluster2 : ndarray
        See :func:`make_star_cluster`.

    Returns
    -------
    cluster1 : ndarray

    """
    cluster1 = deepcopy(input_cluster1)
    mask = cluster2 != 0

    if cluster1[mask].min() < cluster2[mask].min():
        m = mask
    else:
        m = cluster1 == 0

    cluster1[m] = cluster2[m]
    return cluster1


def lines_texture_map(mask, profile='spherical', thickness=10,
                      spacing=20, scale=1.2, orientation=0.):
    """
    Create a lines texture map by applying the texture to the regions
    defined by the input ``mask``.

    Parameters
    ----------
    mask : `~numpy.ndarray` (bool)
        A 2D boolean mask.  The texture will be applied where the
        ``mask`` is `True`.

    profile : {'spherical', 'linear'}
        The line profile. ``'linear'`` produces a "^"-shaped line
        profile.  ``'spherical'`` produces a rounded cylindrical or
        elliptical profile (see ``scale`` for details).

    thickness : int
        Thickness of the line over the entire profile.

    spacing : int
        Perpendicular spacing between adjacent line centers.

    scale : float
        The scale factor applied to the line.  If ``scale`` is 1, then
        the line height is half the ``thickness``.

        For a ``'spherical'`` profile, ``scale=1`` produces a
        hemispherical profile perpendicular to the line.  If ``scale``
        is not 1, then the profile is elliptical.

    orientation : float, optional
        The counterclockwise rotation angle in degrees.  The default
        ``orientation`` of 0 degrees corresponds to horizontal lines.

    Returns
    -------
    data : `~numpy.ndarray`
        An image with same shape as the input ``mask`` containing the
        applied texture map.

    Examples
    --------
    Texture for NGC 602 dust region:

    >>> dust_tx = lines_texture_map(
    ...     dust_mask, profile='linear', thickness=15, spacing=25,
    ...     scale=0.7, orientation=0)
    """

    texture = lines_texture(mask.shape, profile, thickness, scale,
                            orientation)
    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture[mask]
    return data


def dots_texture_map(mask, profile='spherical', diameter=5, scale=3.2,
                     grid_func=hexagonal_grid, grid_spacing=7):
    """
    Create a dots texture map by applying the texture to the regions
    defined by the input ``mask``.

    Parameters
    ----------
    mask : `~numpy.ndarray` (bool)
        A 2D boolean mask.  The texture will be applied where the
        ``mask`` is `True`.

    profile : {'spherical', 'linear'}
        The dot profile. ``'linear'`` produces a cone-shaped dot
        profile.  ``'spherical'`` produces a hemispherical or half
        ellipsoid dot profile (see ``scale`` for details).

    diameter : int
        The dot diameter.

    scale : float
        The scale factor applied to the dot.  If ``scale`` is 1, then
        the dot height is half the ``diameter``.

        For a ``'spherical'`` profile, ``scale=1`` produces a
        hemispherical dot.  If ``scale`` is not 1, then the dot profile
        is a half ellipsoid (circular base with a stretched height).

    grid_func : callable
        The function used to generate the ``(x, y)`` positions of the
        dots.

    grid_spacing : float
        The spacing in pixels between the grid points.

    Returns
    -------
    data : `~numpy.ndarray`
        An image with same shape as the input ``mask`` containing the
        applied texture map.

    Examples
    --------
    Texture for NGC 602 dust region:

    >>> gas_tx = dots_texture_map(
    ...     gas_mask, profile='linear', diameter=7, scale=1.0,
    ...     grid_func=hexagonal_grid, grid_spacing=7)

    Texture for NGC 602 dust and gas combined region:

    >>> dustgas_tx = dots_texture_map(
    ...     dustgas_mask, profile='linear', diameter=7, scale=3.0,
    ...     grid_func=hexagonal_grid, grid_spacing=10)

    Alternate texture for NGC 602 gas region:

    >>> dust_tx = dots_texture_map(
    ...     dust_mask, profile='linear', diameter=7, scale=3.0,
    ...     grid_func=hexagonal_grid, grid_spacing=20)
    """

    texture = dots_texture(mask.shape, profile, diameter, scale,
                           grid_func(mask.shape, grid_spacing))
    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture[mask]
    return data


def textures_to_jpeg():
    """Generate some textures and save them to JPEG images."""
    from .imutils import im2file as save

    shape = (200, 200)
    size = [15, 10, 6, 3]        # line thickness or dot diameter
    spacing = [25, 15, 10, 5]    # line spacing or dot grid spacing

    for sz, sp in zip(size, spacing):
        log.info('{0} {1}'.format(sz, sp))
        for profile in ['spherical', 'linear']:
            log.info('\t{0}'.format(profile))

            lim = lines_texture(shape, profile, sz, sp, 1., orientation=0.)
            fn = ('lines_{0}_thickness{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save(lim, fn)

            rlim = lim.transpose()
            lim[rlim > lim] = rlim[rlim > lim]
            fn = ('hatch_{0}_thickness{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save(lim, fn)

            sdim = dots_texture(shape, profile, sz, 1.,
                                square_grid(shape, sp))
            fn = ('dots_squaregrid_{0}_diameter{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save(sdim, fn)

            hdim = dots_texture(shape, profile, sz, 1,
                                hexagonal_grid(shape, sp))
            fn = ('dots_hexagonalgrid_{0}_diameter{1}_spacing{2}'
                  '.jpg'.format(profile, sz, sp))
            save(hdim, fn)


# Pre-defined textures (by Perry Greenfield for NGC 602)
# This is for XSIZE=1100 YSIZE=1344
# DOTS = partial(
#    dots_texture_map, profile='linear', diameter=7, scale=3.0,
#    grid_func=hexagonal_grid, grid_spacing=10)
# SMALL_DOTS = partial(
#    dots_texture_map, profile='linear', diameter=7, scale=1.0,
#    grid_func=hexagonal_grid, grid_spacing=7)
# LINES = partial(lines_texture_map, profile='linear', thickness=15,
#                spacing=25, scale=0.7, orientation=0)

# Pre-defined textures (by Roshan Rao for NGC 3344 and NGC 1566)
# This is for roughly XSIZE=1000 YSIZE=1000
DOTS = partial(
    dots_texture_map, profile='linear', diameter=5, scale=3.2,
    grid_func=hexagonal_grid, grid_spacing=7)
SMALL_DOTS = partial(
    dots_texture_map, profile='linear', diameter=5, scale=1.8,
    grid_func=hexagonal_grid, grid_spacing=4.5)
LINES = partial(lines_texture_map, profile='linear', thickness=13, spacing=20,
                scale=1.2, orientation=0)
NO_TEXTURE = lambda mask: np.zeros_like(mask)
