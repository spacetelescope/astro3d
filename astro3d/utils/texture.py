"""
This module defines and adds textures to an image.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from operator import attrgetter
from functools import partial
import warnings
import numpy as np
from astropy import log
from astropy.modeling import Parameter, Fittable2DModel
from astropy.utils.exceptions import AstropyUserWarning
#from scipy import ndimage


def combine_textures_max(texture1, texture2):
    """
    Combine two texture images.

    The non-zero values of the texture image with the largest maximum
    replaces the other texture image.

    When combining more than two texture images, one should sort the
    textures by their maximum values and start the combinations from the
    lowest maxima.  This is necessary to properly layer the textures on
    top of each other (where applicable).

    Parameters
    ----------
    texture1 : `~numpy.ndarray`
        Data array of the first texture map.

    texture2 : `~numpy.ndarray`
        Data array of the second texture map.

    Returns
    -------
    data : `~numpy.ndarray`
        Data array of the combined texture map.
    """

    if texture2.max() >= texture1.max():
        # non-zero values of texture2 replace texture1
        data = np.copy(texture1)
        mask = (texture2 != 0)
        if not np.any(mask):
            # both textures contain only zeros
            return data
        else:
            data[mask] = texture2[mask]
    else:
        # non-zero values of texture1 replace texture2
        data = np.copy(texture2)
        mask = (texture1 != 0)
        data[mask] = texture1[mask]
    return data


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


def lines_texture(shape, profile, thickness, height, spacing, orientation=0.):
    """
    Create a texture consisting of regularly-spaced set of lines.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The line profile. ``'linear'`` produces a "^"-shaped line
        profile.  ``'spherical'`` produces a rounded cylindrical or
        elliptical profile.  See ``height`` for more details.

    thickness : int
        Thickness of the line over the entire profile.

    height : float
        The maximum height (data value) of the line.

        For a ``'spherical'`` profile, set ``height`` equal to half the
        ``thickness`` to produce a hemispherical profile perpendicular
        to the line, otherwise the profile is elliptical.

    spacing : int
        Perpendicular spacing between adjacent line centers.

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
    h_thick = (thickness - 1) / 2.
    for offset in offsets:
        y_diff = y - offset
        idx = np.where((y_diff > -h_thick) & (y_diff < h_thick))
        if idx:
            if profile == "spherical":
                data[idx] = ((height / h_thick) *
                             np.sqrt(h_thick**2 - y_diff[idx]**2))
            elif profile == "linear":
                data[idx] = ((height / h_thick) *
                             (h_thick - np.abs(y_diff[idx])))
    return data


def dots_texture(shape, profile, diameter, height, locations):
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
        ellipsoid dot profile.  See ``height`` for more details.

    diameter : int
        The diameter of the dot.

    height : float
        The maximum height (data value) of the dot.

        For a ``'spherical'`` profile, set ``height`` equal to half the
        ``diameter`` to produce a hemispherical dot, otherwise the dot
        profile is a half ellipsoid (circular base with a stretched
        height).

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
    >>> dots_texture(shape, 'linear', 7, 3,
    ...              locations=hexagonal_grid(shape, 10))
    """

    if int(diameter) != diameter:
        raise ValueError('diameter must be an integer')

    dot_shape = (diameter, diameter)
    dot = np.zeros(dot_shape)
    y, x = np.indices(dot_shape)
    radius = (diameter - 1) / 2
    r = np.sqrt((x - radius)**2 + (y - radius)**2)
    idx = np.where(r < radius)

    if profile == 'spherical':
        dot[idx] = (height / radius) * np.sqrt(radius**2 - r[idx]**2)
    elif profile == 'linear':
        dot[idx] = (height / radius) * np.abs(radius - r[idx])
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


def lines_texture_map(mask, profile='spherical', thickness=10,
                      height=6.0, spacing=20, orientation=0.):
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
        elliptical profile (see ``height`` for details).

    thickness : int
        Thickness of the line over the entire profile.

    height : float
        The maximum height (data value) of the line.

        For a ``'spherical'`` profile, set ``height`` equal to half the
        ``thickness`` to produce a hemispherical profile perpendicular
        to the line, otherwise the profile is elliptical.

    spacing : int
        Perpendicular spacing between adjacent line centers.

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
    ...     dust_mask, profile='linear', thickness=15, height=5.25,
    ...     spacing=25, orientation=0)
    """

    texture = lines_texture(mask.shape, profile, thickness, height, spacing,
                            orientation)
    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture[mask]
    return data


def dots_texture_map(mask, profile='spherical', diameter=5,
                     height=8., grid_func=hexagonal_grid, grid_spacing=7):
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
        ellipsoid dot profile (see ``height`` for details).

    diameter : int
        The dot diameter.

    height : float
        The maximum height (data value) of the dot.

        For a ``'spherical'`` profile, set ``height`` equal to half the
        ``diameter`` to produce a hemispherical dot, otherwise the dot
        profile is a half ellipsoid (circular base with a stretched
        height).

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
    ...     gas_mask, profile='linear', diameter=7, height=3.5,
    ...     grid_func=hexagonal_grid, grid_spacing=7)

    Texture for NGC 602 dust and gas combined region:

    >>> dustgas_tx = dots_texture_map(
    ...     dustgas_mask, profile='linear', diameter=7, height=10.5,
    ...     grid_func=hexagonal_grid, grid_spacing=10)

    Alternate texture for NGC 602 gas region:

    >>> dust_tx = dots_texture_map(
    ...     dust_mask, profile='linear', diameter=7, height=10.5,
    ...     grid_func=hexagonal_grid, grid_spacing=20)
    """

    texture = dots_texture(mask.shape, profile, diameter, height,
                           grid_func(mask.shape, grid_spacing))
    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture[mask]
    return data


class StarTexture(Fittable2DModel):
    """
    A 2D star texture model.

    The texture is a parabolic "bowl" of specified maximum
    ``amplitude``, bowl ``depth``, and a circular base of given
    ``radius``

    Parameters
    ----------
    x_0 : float
        x position of the center of the star texture.

    y_0 : float
        y position of the center of the star texture.

    amplitude : float
        The maximum amplitude of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    radius : float
        The circular radius of the star texture.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    amplitude = Parameter()
    depth = Parameter()
    radius = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, amplitude, depth, radius):
        """Star model function."""
        xx = x - x_0
        yy = y - y_0
        r = np.sqrt(xx**2 + yy**2)
        star = depth * (r / radius)**2 + amplitude
        star[r > radius] = 0.
        return star


class StarClusterTexture(Fittable2DModel):
    """
    A 2D star cluster texture model.

    The texture is comprised of three touching star textures
    (`StarTexture`) arranged in an equilateral triangle pattern.  Each
    individual star texture has the same ``amplitude``, ``depth``, and
    ``radius``.

    Parameters
    ----------
    x_0 : float
        x position of the center of the star cluster.

    y_0 : float
        y position of the center of the star cluster.

    amplitude : float
        The maximum amplitude of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    radius : float
        The circular radius of the star texture.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    amplitude = Parameter()
    depth = Parameter()
    radius = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, amplitude, depth, radius):
        """Star cluster model function."""
        h1 = radius / np.sqrt(3.)
        h2 = 2. * radius / np.sqrt(3.)
        y1, x1 = (y_0 - h1, x_0 - radius)
        y2, x2 = (y_0 - h1, x_0 + radius)
        y3, x3 = (y_0 + h2, x_0)
        star1 = StarTexture(x1, y1, amplitude, depth, radius)(x, y)
        star2 = StarTexture(x2, y2, amplitude, depth, radius)(x, y)
        star3 = StarTexture(x3, y3, amplitude, depth, radius)(x, y)
        return np.maximum(np.maximum(star1, star2), star3)


def starlike_models(image, model_type, sources, depth=5, radius_a=10,
                    radius_b=5, base_percentile=75):
    """
    Create the star-like (star or star cluster) texture models.

    Given the position and amplitude of each source (``sources``),
    a list of texture models is generated.  The radius of the
    star (used in both `StarTexture` and `StarCluster` textures) for
    each source is linearly scaled by the source flux as:

        .. math:: radius = radius_a + (radius_b * flux / max_flux)

    where ``max_flux`` is the maximum ``flux`` value of all the models.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    model_type : {'star', 'star_cluster'}
        The type of the star-like texture.

    sources : `~astropy.table.Table`
        A table defining the stars or star clusters.  The table must contain
        ``'x_center'``, ``'y_center'``, and ``'flux'`` columns.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    radius_a : float
        The intercept term in calculating the star radius (see above).

    radius_b : float
        The slope term in calculating the star radius (see above).

    base_percentile : float in the range of [0, 100]
        The percentile of the image data values within the source
        texture, which is used to calculate the base amplitude of the
        texture.

    Returns
    -------
    result : list
        A list of `StarTexture` or `StarClusterTexture` model objects.
    """

    if model_type == 'star':
        Texture = StarTexture
    elif model_type == 'star_cluster':
        Texture = StarClusterTexture
    else:
        raise ValueError('model_type must be "star" or "star_cluster"')

    models = []
    y, x = np.indices(image.shape)

    # NOTE:  probably should exclude any bad sources (e.g. bad position)
    #        before finding the maximum amplitude - but expensive, so skip
    #        for now
    max_flux = float(np.max(sources['flux']))

    for source in sources:
        radius = radius_a + (radius_b * source['flux'] / max_flux)
        model = Texture(source['x_center'], source['y_center'],
                        1.0, depth, radius)
        texture = model(x, y)
        mask = (texture != 0)
        if not np.any(mask):
            # texture contains only zeros (e.g. bad position), so skip model
            warnings.warn('source texture at (x, y) = ({0}, {1}) does not '
                          'overlap with the image'.format(source['x_center'],
                                                          source['y_center']),
                          AstropyUserWarning)
            continue

        if base_percentile is None:
            amplitude = 0.
        else:
            amplitude = np.percentile(image[mask], base_percentile)
        models.append(Texture(source['x_center'],
                              source['y_center'], amplitude, depth,
                              radius))

    #if h_percentile is not None:
    #    filt = ndimage.filters.maximum_filter(array, fil_size)
    #    mask = (filt > 0) & (image > filt) & (array == 0)
    #    array[mask] = filt[mask]

    return models


def sort_starlike_models(texture_models):
    """
    Sort star-like texture models by their ``amplitude`` parameter.

    Parameters
    ----------
    texture_models : list of `StarTexture` and/or `StarClusterTexture`
        A list of star-like texture models including stars
        (`StarTexture`) and/or star clusters (`StarClusterTexture`).
        Each model must contain an ``amplitude`` parameter.

    Returns
    -------
    result : list of `StarTexture` and/or `StarClusterTexture`
        A list of `StarTexture` and/or `StarClusterTexture` sorted by
        the ``amplitude`` parameter in increasing order.
    """

    return sorted(texture_models, key=attrgetter('amplitude'))


def starlike_texture_map(shape, texture_models):
    """
    Create a star and/or star cluster texture map by combining all the
    individual texture models.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    texture_models : list of `StarTexture` and/or `StarClusterTexture`
        A list of star-like texture models including stars
        (`StarTexture`) and/or star clusters (`StarClusterTexture`).

    Returns
    -------
    data : `~numpy.ndarray`
        An image with the specified ``shape`` containing the star and/or
        star cluster texture map.

    Notes
    -----
    The texture models will be sorted by their ``amplitude``s (maximum
    value) in increasing order and then added to the output texture map
    starting with the smallest ``amplitude``.
    """

    data = np.zeros(shape)
    y, x = np.indices(shape)
    for texture_model in sort_starlike_models(texture_models):
        data = combine_textures_max(data, texture_model(x, y))
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

            lim = lines_texture(shape, profile, sz, 1., sp, orientation=0.)
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
#    dots_texture_map, profile='linear', diameter=7, height=10.5,
#    grid_func=hexagonal_grid, grid_spacing=10)
# SMALL_DOTS = partial(
#    dots_texture_map, profile='linear', diameter=7, height=3.5,
#    grid_func=hexagonal_grid, grid_spacing=7)
# LINES = partial(lines_texture_map, profile='linear', thickness=15,
#                 height=5.25, spacing=25, orientation=0)

# Pre-defined textures (by Roshan Rao for NGC 3344 and NGC 1566)
# This is for roughly XSIZE=1000 YSIZE=1000
DOTS = partial(
    dots_texture_map, profile='linear', diameter=5, height=8.0,
    grid_func=hexagonal_grid, grid_spacing=7)
SMALL_DOTS = partial(
    dots_texture_map, profile='linear', diameter=5, height=4.5,
    grid_func=hexagonal_grid, grid_spacing=4.5)
LINES = partial(lines_texture_map, profile='linear', thickness=13,
                height=7.8, spacing=20, orientation=0)
NO_TEXTURE = lambda mask: np.zeros_like(mask)
