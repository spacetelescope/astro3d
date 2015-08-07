"""
This module provides tools for defining textures and applying them to an
image.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from operator import attrgetter
from functools import partial
import warnings
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.modeling import Parameter, Fittable2DModel
from astropy.utils.exceptions import AstropyUserWarning
from .image_utils import resize_image


def combine_textures_max(texture1, texture2):
    """
    Combine two texture images.

    The non-zero values of the texture image with the largest maximum
    replace the values in the other texture image.

    When sequentially using this function to combine more than two
    texture images, one should sort the textures by their maximum values
    and start combining from the lowest maxima.  This is necessary to
    properly layer the textures on top of each other (where applicable).

    If both ``texture1`` and ``texture2`` contain only zeros, then
    ``texture1`` is returned (i.e. an array of zeros).

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
        The spacing in pixels between the centers of adjacent squares.
        This is the same as the square size.

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
    Create a texture image consisting of regularly-spaced set of lines.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The line profile. ``'linear'`` produces a "^"-shaped line
        profile.  ``'spherical'`` produces a rounded cylindrical or
        elliptical profile.  See ``height`` for more details.

    thickness : int
        Thickness of the line over the entire profile (i.e full width at
        zero intensity).

    height : float
        The maximum height (data value) of the line.

        For a ``'spherical'`` profile, set ``height`` equal to half the
        ``thickness`` to produce a hemispherical line profile, otherwise
        the profile is elliptical.

    spacing : int
        Perpendicular spacing between adjacent line centers.

    orientation : float, optional
        The counterclockwise rotation angle in degrees.  The default
        ``orientation`` of 0 degrees corresponds to horizontal lines
        in the output image.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the line textures.
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
    Create a texture image consisting of dots at the given locations.

    If any dots overlap (e.g. the ``locations`` separations are smaller
    than the dot size), then the greater data value of the two is taken,
    not the sum.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The dot profile. ``'linear'`` produces a cone-shaped dot
        profile.  ``'spherical'`` produces a hemispherical or
        half-ellipsoid dot profile.  See ``height`` for more details.

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
    yy, xx = np.indices(dot_shape)
    radius = (diameter - 1) / 2
    r = np.sqrt((xx - radius)**2 + (yy - radius)**2)
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
        ``thickness`` to produce a hemispherical line profile, otherwise
        the profile is elliptical.

    spacing : int
        Perpendicular spacing between adjacent line centers.

    orientation : float, optional
        The counterclockwise rotation angle in degrees.  The default
        ``orientation`` of 0 degrees corresponds to horizontal lines
        in the output image.

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
        profile.  ``'spherical'`` produces a hemispherical or
        half-ellipsoid dot profile (see ``height`` for details).

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

    The star texture is a parabolic "bowl" with a circular base
    of given ``radius``, bowl ``depth``, and ``base_height`.

    Parameters
    ----------
    x_0 : float
        x position of the center of the star texture.

    y_0 : float
        y position of the center of the star texture.

    radius : float
        The circular radius of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_height : float
        The base height of the star texture.  This is the texture height
        at the "bowl" minimum.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    radius = Parameter()
    depth = Parameter()
    base_height = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, radius, depth, base_height):
        """Star model function."""

        # NOTE: min_height is added to keep the star texture values > 0 at
        #       the bowl center (r=0) when base_height is zero
        min_height = 0.0001

        xx = x - x_0
        yy = y - y_0
        r = np.sqrt(xx**2 + yy**2)
        star = depth * (r / radius)**2 + base_height + min_height
        star[r > radius] = 0.
        return star


class StarClusterTexture(Fittable2DModel):
    """
    A 2D star cluster texture model.

    The star cluster texture is comprised of three touching star
    textures (`StarTexture`) arranged in an equilateral triangle
    pattern.  Each individual star texture has the same ``radius``,
    ``depth``, and ``base_height``.

    Parameters
    ----------
    x_0 : float
        x position of the center of the star cluster.

    y_0 : float
        y position of the center of the star cluster.

    radius : float
        The circular radius of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_height : float
        The base height of the star texture.  This is the texture height
        at the "bowl" minimum.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    radius = Parameter()
    depth = Parameter()
    base_height = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, radius, depth, base_height):
        """Star cluster model function."""
        h1 = radius / np.sqrt(3.)
        h2 = 2. * radius / np.sqrt(3.)
        y1, x1 = (y_0 - h1, x_0 - radius)
        y2, x2 = (y_0 - h1, x_0 + radius)
        y3, x3 = (y_0 + h2, x_0)
        star1 = StarTexture(x1, y1, radius, depth, base_height)(x, y)
        star2 = StarTexture(x2, y2, radius, depth, base_height)(x, y)
        star3 = StarTexture(x3, y3, radius, depth, base_height)(x, y)
        return np.maximum(np.maximum(star1, star2), star3)


def starlike_model_base_height(image, model_type, x, y, radius, depth,
                               base_percentile=75, image_indices=None):
    """
    Create a star-like (star or star cluster) texture model where the
    model base height has been calculated from the image values where
    the texture is non-zero.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    model_type : {'star', 'star_cluster'}
        The type of the star-like texture.

    x, y : float
        The ``x`` and ``y`` image position of the star or star cluster.

    radius : float
        The circular radius of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_percentile : float in the range of [0, 100]
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    image_indices : tuple of 2D `~numpy.ndarray`s
        A ``(yy, xx)`` tuple where ``yy`` and ``xx`` are 2D images with
        the same shape of the input ``image`` and represent the ``y``
        and ``x`` image indices (i.e. the tuple returned from
        ``np.indices(image)``).  Use ``image_indices`` when calling this
        function in a loop (see `make_starlike_models`).  If `None`,
        then ``np.indices(image)`` will be called.

    Returns
    -------
    model : `StarTexture` or `StarClusterTexture`
        A `StarTexture` or `StarClusterTexture` model object.  `None` is
        returned if the model does not overlap with the input image.
    """

    if model_type == 'star':
        Texture = StarTexture
    elif model_type == 'star_cluster':
        Texture = StarClusterTexture
    else:
        raise ValueError('model_type must be "star" or "star_cluster"')

    if image_indices is None:
        yy, xx = np.indices(image.shape)
    else:
        yy, xx = image_indices
        if yy.shape != xx.shape:
            raise ValueError('x and y image_indices must have the same '
                             'shape.')
        if yy.shape != image.shape:
            raise ValueError('x and y image_indices must have the same '
                             'shape as the input image.')

    texture = Texture(x, y, radius, depth, 1.0)(xx, yy)
    mask = (texture != 0)
    if not np.any(mask):
        # texture contains only zeros (e.g. bad position)
        warnings.warn('Source texture at (x, y) = ({0}, {1}) does not '
                      'overlap with the image'.format(x, y),
                      AstropyUserWarning)
        return None

    if base_percentile is None:
        base_height = 0.
    else:
        base_height = np.percentile(image[mask], base_percentile)

    return Texture(x, y, radius, depth, base_height)


def make_starlike_models(image, model_type, sources, radius_a=10, radius_b=5,
                         depth=5, base_percentile=75):
    """
    Create the star-like (star or star cluster) texture models to be
    applied to an image.

    Given the position and flux amplitude of each source (``sources``),
    a list of texture models is generated.  The radius of the star (used
    in both `StarTexture` and `StarCluster` textures) for each source is
    linearly scaled by the source flux as:

        .. math:: radius = radius_a + (radius_b * flux / max_flux)

    where ``max_flux`` is the maximum ``flux`` value of all the input
    ``sources``.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    model_type : {'star', 'star_cluster'}
        The type of the star-like texture.

    sources : `~astropy.table.Table`
        A table defining the stars or star clusters.  The table must
        contain ``'xcen'``, ``'ycen'``, and ``'flux'`` columns.

    radius_a : float
        The intercept term in calculating the star radius (see above).

    radius_b : float
        The slope term in calculating the star radius (see above).

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_percentile : float in the range of [0, 100]
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    models : list
        A list of `StarTexture` or `StarClusterTexture` model objects.
    """

    if len(sources) == 0:
        return []

    columns = ['xcen', 'ycen', 'flux']
    for column in columns:
        if column not in sources.colnames:
            raise ValueError('sources must contain a {0} '
                             'column'.format(column))

    # NOTE:  probably should exclude any bad sources (e.g. bad position)
    #        before finding the maximum flux - but expensive, so skip
    #        for now
    max_flux = float(np.max(sources['flux']))

    yy, xx = np.indices(image.shape)
    models = []
    for source in sources:
        radius = radius_a + (radius_b * source['flux'] / max_flux)
        model = starlike_model_base_height(
            image, model_type, source['xcen'], source['ycen'], radius, depth,
            base_percentile=base_percentile, image_indices=(yy, xx))
        if model is not None:
            models.append(model)
    return models


def sort_starlike_models(models):
    """
    Sort star-like texture models by their ``base_height`` parameter.

    Parameters
    ----------
    models : list of `StarTexture` and/or `StarClusterTexture`
        A list of star-like texture models including stars
        (`StarTexture`) and/or star clusters (`StarClusterTexture`).
        Each model must contain an ``base_height`` parameter.

    Returns
    -------
    sorted_models : list of `StarTexture` and/or `StarClusterTexture`
        A list of `StarTexture` and/or `StarClusterTexture` models
        sorted by the ``base_height`` parameter in increasing order.
    """

    return sorted(models, key=attrgetter('base_height'))


def starlike_texture_map(shape, models):
    """
    Create a star and/or star cluster texture map by combining all the
    individual star-like texture models.

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
    The texture models will be sorted by their ``base_height``s in
    increasing order and then added to the output texture map starting
    with the smallest ``base_height``.
    """

    data = np.zeros(shape)
    yy, xx = np.indices(shape)
    for model in sort_starlike_models(models):
        data = combine_textures_max(data, model(xx, yy))
    return data


def make_starlike_textures(image, star_sources, cluster_sources, radius_a=10,
                           radius_b=5, depth=5, base_percentile=75):
    """
    Make an image containing star-like textures (stars and star clusters).

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    star_sources : `~astropy.table.Table`
        A table defining the stars.  The table must contain ``'xcen'``,
        ``'ycen'``, and ``'flux'`` columns.

    cluster_sources : `~astropy.table.Table`
        A table defining the star clusters.  The table must contain
        ``'xcen'``, ``'ycen'``, and ``'flux'`` columns.

    radius_a : float
        The intercept term in calculating the star radius (see above).

    radius_b : float
        The slope term in calculating the star radius (see above).

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_percentile : float in the range of [0, 100]
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    data : `~numpy.ndarray`
        The image containing the star and star cluster textures.
    """

    star_models = make_starlike_models(image, 'star', star_sources,
                                       radius_a=radius_a, radius_b=radius_b,
                                       depth=depth,
                                       base_percentile=base_percentile)
    cluster_models = make_starlike_models(image, 'star_cluster',
                                          cluster_sources, radius_a=radius_a,
                                          radius_b=radius_b, depth=depth,
                                          base_percentile=base_percentile)
    starlike_models = star_models + cluster_models
    starlike_textures = starlike_texture_map(image.shape, starlike_models)

    return starlike_textures


def make_cusp_texture(image, x, y, radius=25, depth=40, base_percentile=None):
    """
    Make an image containing a star-like cusp texture.

    The cusp texture is used to mark the center of a galaxy.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the texture will be applied.

    x, y : float
        The ``x`` and ``y`` image position of the cusp texture.  This
        should be the galaxy center.

    radius : float
        The circular radius of the star texture.

    depth : float
        The maximum depth of the crater-like bowl of the star texture.

    base_percentile : float in the range of [0, 100]
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    data : `~numpy.ndarray`
        The image containing the cusp texture.
    """

    yy, xx = np.indices(image.shape)
    return starlike_model_base_height(image, 'star', x, y, radius, depth,
                                      base_percentile=base_percentile)(xx, yy)


def apply_textures(image, texture_image):
    """
    Apply textures to an image.

    Pixels in the input ``image`` are replaced by the non-zero pixels in
    the input ``texture_image``.

    This function is used for the star-like textures (central galaxy
    cusp, stars, and star clusters), which could be added on top of
    other textures (lines, dots, or small dots).  The star-like textures
    replace, instead of add to, image values.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    texture_image : `~numpy.ndarray`
        The texture image, which must be the same shape as the input
        ``image``.

    Returns
    -------
    data : `~numpy.ndarray`
        The image with the applied textures.
    """

    data = np.copy(image)
    idx = (texture_image != 0)
    data[idx] = texture_image[idx]
    return data


def textures_to_jpeg():
    """Generate some textures and save them to JPEG images."""
    from .image_utils import im2file as save

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


DOTS = partial(
    dots_texture_map, profile='spherical', diameter=9.0, height=8.0,
    grid_func=hexagonal_grid, grid_spacing=9.0)

SMALL_DOTS = partial(
    dots_texture_map, profile='spherical', diameter=9.0, height=4.0,
    grid_func=hexagonal_grid, grid_spacing=5.0)

LINES = partial(lines_texture_map, profile='linear', thickness=13,
                height=7.8, spacing=20, orientation=0)
