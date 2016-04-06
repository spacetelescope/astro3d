"""
This module provides tools to define textures and apply them to an
image.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from operator import attrgetter
import warnings
import numpy as np
from astropy.modeling import Parameter, Fittable2DModel
from astropy.modeling.models import Disk2D
from astropy.utils.exceptions import AstropyUserWarning


__doctest_skip__ = ['lines_texture_image', 'dots_texture_image']


def apply_texture_mask(texture_image, mask):
    """
    Apply textures only to the masked region of an image.

    Parameters
    ----------
    texture_image : `~numpy.ndarray`
        An image completely filled with textures.

    mask : `~numpy.ndarray` (bool)
        A 2D boolean mask.  The texture will be removed where the
        ``mask`` is `False` and applied only where ``mask`` is `True`.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the masked textures.
    """

    if texture_image.shape != mask.shape:
        raise ValueError('texture_image and mask must have the same shape')

    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture_image[mask]
    return data


def combine_textures_max(texture1, texture2):
    """
    Combine two texture images.

    The non-zero values of the texture image with the largest maximum
    replace the values in the other texture image.

    When sequentially using this function to combine more than two
    texture images, one should sort the textures by their maximum values
    and start combining from the lowest maximum.  This is necessary to
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


def lines_texture_image(shape, profile, thickness, height, spacing,
                        orientation=0., mask=None):
    """
    Create a texture image consisting of a regularly-spaced set of lines.

    If a ``mask`` image is input, then texture is applied only to the
    regions where the ``mask`` is `True`.

    Parameters
    ----------
    shape : tuple
        The shape of the output image.

    profile : {'linear', 'spherical'}
        The line profile. ``'linear'`` produces a "^"-shaped line
        profile.  ``'spherical'`` produces a rounded cylindrical or
        elliptical profile.  See ``height`` for more details.

    thickness : int
        The thickness of the line spanning the entire profile (i.e full
        width at zero intensity).

    height : float
        The maximum height (data value) of the line.

        For a true ``'spherical'`` profile, set ``height`` equal to half
        the ``thickness`` to produce a hemispherical line profile,
        otherwise the profile is elliptical.

    spacing : int
        The perpendicular spacing between adjacent line centers.

    orientation : float, optional
        The counterclockwise rotation angle (in degrees) for the lines.
        The default ``orientation`` of 0 degrees corresponds to
        horizontal lines (i.e. lines along rows) in the output image.

    mask : `~numpy.ndarray` (bool)
        A 2D boolean mask.  If input, the texture will be applied where
        the ``mask`` is `True`.  ``mask`` must have the same shape as
        the input ``shape``.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the "lines" texture.

    Examples
    --------
    Texture image for the NGC 602 dust region:

    >>> dust_tx = lines_texture_image(
    ...     profile='linear', thickness=15, height=5.25, spacing=25,
    ...     orientation=0, mask=dust_mask)
    """

    # start at the image center and then offset lines in both directions
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

    if mask is None:
        return data
    else:
        return apply_texture_mask(data, mask)


def dots_texture_image(shape, profile, diameter, height, locations=None,
                       grid_func=None, grid_spacing=None, mask=None):
    """
    Create a texture image consisting of dots centered at the given
    locations.

    If two dots overlap (i.e. the ``locations`` separations are smaller
    than the dot size), then the greater data value of the two is taken,
    not the sum.  This ensures the maximum ``height`` of the dot
    textures.

    Either ``locations`` or both ``grid_func`` and ``grid_spacing`` need
    to be specified.  If all are input, then ``locations`` takes
    precedence.

    If a ``mask`` image is input, then texture is applied only to the
    regions where the ``mask`` is `True`.

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

        For a true ``'spherical'`` profile, set ``height`` equal to half
        the ``diameter`` to produce a hemispherical dot, otherwise the
        dot profile is a half ellipsoid (circular base with a stretched
        height).

    locations : `~numpy.ndarray`, optional
        A ``Nx2`` `~numpy.ndarray` where each row contains the ``x`` and
        ``y`` coordinate positions.  Either ``locations`` or both
        ``grid_func`` and ``grid_spacing`` need to be specified.  If all
        are input, then ``locations`` takes precedence.

    grid_func : callable, optional
        The function used to generate the ``(x, y)`` positions of the
        dots.  Either ``locations`` or both ``grid_func`` and
        ``grid_spacing`` need to be specified.  If all are input, then
        ``locations`` takes precedence.

    grid_spacing : float, optional
        The spacing in pixels between the grid points.  Either
        ``locations`` or both ``grid_func`` and ``grid_spacing`` need to
        be specified.  If all are input, then ``locations`` takes
        precedence.

    mask : `~numpy.ndarray` (bool)
        A 2D boolean mask.  If input, the texture will be applied where
        the ``mask`` is `True`.  ``mask`` must have the same shape as
        the input ``shape``.

    Returns
    -------
    data : `~numpy.ndarray`
        An image containing the dot texture.

    Examples
    --------
    >>> shape = (1000, 1000)
    >>> dots_texture_image(shape, 'linear', 7, 3,
    ...                    locations=hexagonal_grid(shape, 10))
    """

    if int(diameter) != diameter:
        raise ValueError('diameter must be an integer')

    if locations is None:
        if grid_func is None or grid_spacing is None:
            raise ValueError('locations or both grid_func and grid_spacing '
                             'must be input')
        locations = grid_func(shape, grid_spacing)

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
            # replace pixel values in the output texture image only
            # where the values are larger in the new dot (i.e. the new dot
            # pixels are not summed with the texture image, but are
            # assigned the greater value of the new dot and the texture
            # image)
            region = data[y-radius:y+radius+1, x-radius:x+radius+1]
            dot_mask = (dot > region)
            region[dot_mask] = dot[dot_mask]

    if mask is None:
        return data
    else:
        return apply_texture_mask(data, mask)


class StarTexture(Fittable2DModel):
    """
    A 2D star texture model.

    The star texture is a parabolic "bowl" with a circular base of given
    ``radius``, bowl ``depth``, ``base_height``, and ``slope``.

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

    slope : float
        The slope of the star texture sides.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    radius = Parameter()
    depth = Parameter()
    base_height = Parameter()
    slope = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, radius, depth, base_height, slope):
        """Star model function."""

        # NOTE: min_height is added to keep the star texture values > 0 at
        #       the bowl center (r=0) when base_height is zero
        min_height = 0.0001

        xx = x - x_0
        yy = y - y_0
        r = np.sqrt(xx**2 + yy**2)
        star = depth * (r / radius)**2 + base_height + min_height

        amplitude = depth + base_height
        bowl_region = (r <= radius)
        sides_region = np.logical_and(r > radius,
                                      r <= (radius + (amplitude / slope)))
        sides = amplitude + (slope * (radius - r))
        model = np.select([bowl_region, sides_region], [star, sides])

        # make the model zero below the base_height
        zero_region = (model < (base_height + min_height))
        model[zero_region] = 0

        return model


class StarClusterTexture(Fittable2DModel):
    """
    A 2D star cluster texture model.

    The star cluster texture is comprised of three touching star
    textures (`StarTexture`) arranged in an equilateral triangle
    pattern.  Each individual star texture has the same ``radius``,
    ``depth``, ``base_height``, and ``slope``.

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

    slope : float
        The slope of the star texture sides.
    """

    x_0 = Parameter()
    y_0 = Parameter()
    radius = Parameter()
    depth = Parameter()
    base_height = Parameter()
    slope = Parameter()

    @staticmethod
    def evaluate(x, y, x_0, y_0, radius, depth, base_height, slope):
        """Star cluster model function."""
        h1 = radius / np.sqrt(3.)
        h2 = 2. * radius / np.sqrt(3.)
        y1, x1 = (y_0 - h1, x_0 - radius)
        y2, x2 = (y_0 - h1, x_0 + radius)
        y3, x3 = (y_0 + h2, x_0)
        star1 = StarTexture(x1, y1, radius, depth, base_height, slope)(x, y)
        star2 = StarTexture(x2, y2, radius, depth, base_height, slope)(x, y)
        star3 = StarTexture(x3, y3, radius, depth, base_height, slope)(x, y)
        # Disk2D is used to fill the central "hole", which needs to be
        # nonzero to prevent a possible central spike when applied to the
        # image.  ``min_height`` is added to keep the texture values > 0
        # even if base_height is zero.
        min_height = 0.0001
        disk = Disk2D(base_height + min_height, x_0, y_0, radius)(x, y)
        return np.maximum(np.maximum(np.maximum(star1, star2), star3), disk)


def starlike_model_base_height(image, model_type, x, y, radius, depth, slope,
                               base_percentile=75, image_indices=None):
    """
    Calculate the model base height for a star-like (star or star
    cluster) texture model.

    The model base height is calculated from the image values where the
    texture is non-zero.

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

    slope : float
        The slope of the star_texture sides.

    base_percentile : float in the range of [0, 100], optional
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    image_indices : tuple of 2D `~numpy.ndarray`, optional
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

    if model_type == 'stars':
        Texture = StarTexture
    elif model_type == 'star_clusters':
        Texture = StarClusterTexture
    else:
        raise ValueError('model_type must be "stars" or "star_clusters"')

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

    texture = Texture(x, y, radius, depth, 1.0, slope)(xx, yy)
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

    return base_height


def make_starlike_models(image, model_type, sources, radius_a=10, radius_b=5,
                         depth=5, slope=0.5, base_percentile=75):
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

    radius_a : float, optional
        The intercept term in calculating the star radius (see above).

    radius_b : float, optional
        The slope term in calculating the star radius (see above).

    depth : float, optional
        The maximum depth of the crater-like bowl of the star texture.

    slope : float
        The slope of the star texture sides.

    base_percentile : float in the range of [0, 100], optional
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    models : list
        A list of `StarTexture` or `StarClusterTexture` model objects.
    """

    if model_type == 'stars':
        Texture = StarTexture
    elif model_type == 'star_clusters':
        Texture = StarClusterTexture
    else:
        raise ValueError('model_type must be "stars" or "star_clusters"')

    if len(sources) == 0:
        return []

    columns = ['xcentroid', 'ycentroid', 'flux']
    for column in columns:
        if column not in sources.colnames:
            raise ValueError('sources must contain a {0} column'
                             .format(column))

    # assumes that all sources in the source table are good
    max_flux = float(np.max(sources['flux']))

    yy, xx = np.indices(image.shape)
    models = []
    for source in sources:
        xcen = source['xcentroid']
        ycen = source['ycentroid']
        radius = radius_a + (radius_b * source['flux'] / max_flux)
        base_height = starlike_model_base_height(
            image, model_type, xcen, ycen, radius, depth, slope,
            base_percentile=base_percentile, image_indices=(yy, xx))
        model = Texture(xcen, ycen, radius, depth, base_height, slope)

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


def make_starlike_textures(image, stellar_tables, radius_a=10, radius_b=5,
                           depth=5, slope=0.5, base_percentile=75):
    """
    Make an image containing star-like textures (stars and star clusters).

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image where the textures will be applied.

    stellar_tables : dict of `~astropy.table.Table`
        A dictionary of tables defining the star-like textures.  The
        dictionary can define either 'stars' or 'star_clusters'.  The
        table must contain ``'xcentroid'``, ``'ycentroid'``, and
        ``'flux'`` columns.

    radius_a : float, optional
        The intercept term in calculating the star radius (see above).

    radius_b : float, optional
        The slope term in calculating the star radius (see above).

    depth : float, optional
        The maximum depth of the crater-like bowl of the star texture.

    slope : float
        The slope of the star texture sides.

    base_percentile : float in the range of [0, 100], optional
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    data : `~numpy.ndarray`
        The image containing the star and star cluster textures.

    Notes
    -----
    The texture models will be sorted by their ``base_height``s in
    increasing order and then added to the output texture map starting
    with the smallest ``base_height``.
    """

    starlike_models = []
    for stellar_type, table in stellar_tables.items():
        starlike_models.extend(
            make_starlike_models(image, stellar_type, table,
                                 radius_a=radius_a, radius_b=radius_b,
                                 depth=depth, slope=slope,
                                 base_percentile=base_percentile))

    # create a texture map from the list of models
    data = np.zeros(image.shape)
    yy, xx = np.indices(image.shape)
    for model in sort_starlike_models(starlike_models):
        data = combine_textures_max(data, model(xx, yy))
    return data


def make_cusp_model(image, x, y, radius=25, depth=40, slope=0.5,
                    base_percentile=None):
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

    radius : float, optional
        The circular radius of the star texture.

    depth : float, optional
        The maximum depth of the crater-like bowl of the star texture.

    slope : float
        The slope of the star texture sides.

    base_percentile : float in the range of [0, 100], optional
        The percentile of the image data values within the source
        texture (where the texture is non-zero) used to define the base
        height of the model texture.  If `None`, then the model
        base_height will be zero.

    Returns
    -------
    data : `~numpy.ndarray`
        The image containing the cusp texture.
    """

    base_height = starlike_model_base_height(
        image, 'stars', x, y, radius, depth, slope,
        base_percentile=base_percentile)
    return StarTexture(x, y, radius, depth, base_height, slope)


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
