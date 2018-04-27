"""
This module provides tools to define textures and apply them to an
image.
"""
import warnings

from astropy import log
from astropy.modeling import Parameter, Fittable2DModel
from astropy.modeling.models import Disk2D
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from scipy.ndimage import binary_dilation

__doctest_skip__ = ['LinesTexture', 'DotsTexture']

# Configure logging
log.setLevel('DEBUG')


def mask_texture_image(texture_image, mask):
    """
    Mask a texture image.

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

    if mask is None:
        return texture_image

    if texture_image.shape != mask.shape:
        raise ValueError('texture_image and mask must have the same shape')

    data = np.zeros_like(mask, dtype=np.float)
    data[mask] = texture_image[mask]
    return data


class SquareGrid(object):
    """
    Class to generate ``(x, y)`` coordinates for a regular square grid
    over a given image shape.

    Parameters
    ----------
    spacing : float
        The spacing in pixels between the centers of adjacent squares.
        This is the same as the square size.

    offset : float, optional
        An optional offset to apply in both the ``x`` and ``y``
        directions from the nominal starting position of ``(0, 0)``.
    """

    def __init__(self, spacing, offset=0):
        self.spacing = spacing
        self.offset = offset

    def __call__(self, shape):
        """
        Generate ``(x, y)`` coordinates for a regular square grid over a
        given image shape.

        Parameters
        ----------
        shape : tuple
            The shape of the image over which to create the grid.

        Returns
        -------
        coords : `~numpy.ndarray`
            A ``N x 2`` array where each row contains the ``x`` and ``y``
            coordinates of the square centers.
        """

        y, x = np.mgrid[self.offset:shape[0]:self.spacing,
                        self.offset:shape[1]:self.spacing]
        return np.transpose(np.vstack([x.ravel(), y.ravel()]))


class HexagonalGrid(object):
    """
    Class to generate ``(x, y)`` coordinates for a hexagonal grid over a
    given image shape.

    Parameters
    ----------
    spacing : float
        The spacing in pixels between the centers of adjacent hexagons.
        This is also the "size" of the hexagon, as measured perpendicular
        from a side to the opposite side.

    offset : float, optional
        An optional offset to apply in both the ``x`` and ``y``
        directions from the nominal starting position of ``(0, 0)``.
    """

    def __init__(self, spacing, offset=0):
        self.spacing = spacing
        self.offset = offset

    def __call__(self, shape):
        """
        Generate ``(x, y)`` coordinates for a hexagonal grid over a
        given image shape.

        Parameters
        ----------
        shape : tuple
            The shape of the image over which to create the grid.

        Returns
        -------
        coords : `~numpy.ndarray`
            A ``N x 2`` array where each row contains the ``x`` and ``y``
            coordinates of the hexagon centers.
        """

        x_spacing = 2. * self.spacing / np.sqrt(3.)
        y, x = np.mgrid[self.offset:shape[0]:self.spacing,
                        self.offset:shape[1]:x_spacing]
        # shift the odd rows by half of the x_spacing
        for i in range(1, len(x), 2):
            x[i] += 0.5 * x_spacing
        return np.transpose(np.vstack([x.ravel(), y.ravel()]))


class RandomPoints(object):
    """
    Class to generate ``(x, y)`` coordinates at random positions over a
    given image shape.

    Parameters
    ----------
    spacing : float
        The "average" spacing between the random positions.
        Specifically, ``spacing`` defines the number of random positions
        as ``shape[0] * shape[1] / spacing**2``.
    """

    def __init__(self, spacing):
        self.spacing = spacing

    def __call__(self, shape):
        """
        Generate ``(x, y)`` coordinates at random positions over a given
        image shape.

        Parameters
        ----------
        shape : tuple
            The shape of the image over which to create the random points.

        Returns
        -------
        coords : `~numpy.ndarray`
            A ``N x 2`` array where each row contains the ``x`` and ``y``
            coordinates of the random positions.
        """

        npts = shape[0] * shape[1] / self.spacing**2
        x = np.random.random(npts) * shape[1]
        y = np.random.random(npts) * shape[0]
        return np.transpose(np.vstack([x, y]))


class LinesTexture(object):
    """
    Class to create a texture image consisting of a regularly-spaced set
    of lines.

    Parameters
    ----------
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
    """

    def __init__(self, profile, thickness, height, spacing, orientation=0.):
        self.profile = profile
        self.thickness = thickness
        self.height = height
        self.spacing = spacing
        self.orientation = orientation

    def __call__(self, shape, mask=None):
        """
        Create a texture image of lines.

        Parameters
        ----------
        shape : tuple
            The shape of the output image.

        mask : bool `~numpy.ndarray`, optional
            A 2D boolean mask.  If input, the texture will be applied
            where the ``mask`` is `True`.  ``mask`` must have the same
            shape as the input ``shape``.

        Returns
        -------
        data : `~numpy.ndarray`
            An image containing the line texture.
        """

        # start at the image center and then offset lines in both directions
        xc = shape[1] / 2
        yc = shape[1] / 2
        x = np.arange(shape[1]) - xc
        y = np.arange(shape[0]) - yc
        xp, yp = np.meshgrid(x, y)

        angle = np.pi * self.orientation/180.
        s, c = np.sin(angle), np.cos(angle)
        # x = c*xp + s*yp    # unused
        y = -s*xp + c*yp

        # compute maximum possible offsets
        noffsets = int(np.sqrt(xc**2 + yc**2) / self.spacing)
        offsets = self.spacing * (np.arange(noffsets*2 + 1) - noffsets)

        # loop over all offsets
        data = np.zeros(shape)
        h_thick = (self.thickness - 1) / 2.
        for offset in offsets:
            y_diff = y - offset
            idx = np.where((y_diff > -h_thick) & (y_diff < h_thick))
            if idx:
                if self.profile == "spherical":
                    data[idx] = ((self.height / h_thick) *
                                 np.sqrt(h_thick**2 - y_diff[idx]**2))
                elif self.profile == "linear":
                    data[idx] = ((self.height / h_thick) *
                                 (h_thick - np.abs(y_diff[idx])))

        return mask_texture_image(data, mask)


class DotsTexture(object):
    """
    Class to create a texture image consisting of dots centered at the
    given locations.

    If two dots overlap (i.e. their separations are smaller than the dot
    size), then the greater data value of the two is taken, not the sum.
    This ensures that the maximum height of the dot textures is the
    input ``height``.

    Either ``locations`` or ``grid`` needs to be input.  If both are
    input, then ``locations`` takes precedence.

    Parameters
    ----------
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
        ``y`` coordinate positions.  Either ``locations`` or ``grid``
        needs to be specified.  If both are input, then ``locations``
        takes precedence.

    grid : callable, optional
        The function or callable object used to generate the ``(x, y)``
        positions of the dots.  Either ``locations`` or ``grid`` needs
        to be specified.  If both are input, then ``locations`` takes
        precedence.
    """

    def __init__(self, profile, diameter, height, locations=None, grid=None):
        if int(diameter) != diameter:
            raise ValueError('diameter must be an integer')
        diameter = int(diameter)

        if locations is None:
            if grid is None:
                raise ValueError('locations or grid must be input')
        self.locations = locations
        self.grid = grid

        dot_shape = (diameter, diameter)
        dot = np.zeros(dot_shape)
        yy, xx = np.indices(dot_shape)
        radius = (diameter - 1) // 2
        r = np.sqrt((xx - radius)**2 + (yy - radius)**2)
        idx = np.where(r < radius)

        if profile == 'spherical':
            dot[idx] = (height / radius) * np.sqrt(radius**2 - r[idx]**2)
        elif profile == 'linear':
            dot[idx] = (height / radius) * np.abs(radius - r[idx])
        else:
            raise ValueError('profile must be "spherical" or "linear"')

        self.dot = dot
        self.radius = radius

    def __call__(self, shape, mask=None):
        """
        Create a texture image of dots.

        Parameters
        ----------
        shape : tuple
            The shape of the output image.

        mask : bool `~numpy.ndarray`, optional
            A 2D boolean mask.  If input, the texture will be applied
            where the ``mask`` is `True`.  ``mask`` must have the same
            shape as the input ``shape``.

        Returns
        -------
        data : `~numpy.ndarray`
            An image containing the dot texture.
        """

        if self.locations is None:
            self.locations = self.grid(shape)

        data = np.zeros(shape)
        for (x, y) in self.locations:
            x = np.rint(x).astype(int)
            y = np.rint(y).astype(int)

            # exclude points too close to the edge
            if not (x < self.radius or x > (shape[1] - self.radius - 1) or
                    y < self.radius or y > (shape[0] - self.radius - 1)):
                # replace pixel values in the output texture image only
                # where the values are larger in the new dot (i.e. the new dot
                # pixels are not summed with the texture image, but are
                # assigned the greater value of the new dot and the texture
                # image)
                cutout = data[y-self.radius:y+self.radius+1,
                              x-self.radius:x+self.radius+1]
                dot_mask = (self.dot > cutout)
                cutout[dot_mask] = self.dot[dot_mask]

        return mask_texture_image(data, mask)


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

    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits in each
        dimension, ``((y_low, y_high), (x_low, x_high))``
        """

        extent = self.radius + (self.depth * self.slope)

        return ((self.y_0 - extent, self.y_0 + extent),
                (self.x_0 - extent, self.x_0 + extent))

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

    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits in each
        dimension, ``((y_low, y_high), (x_low, x_high))``
        """

        extent = (2. * self.radius) + (self.depth * self.slope) + 2

        return ((self.y_0 - extent, self.y_0 + extent),
                (self.x_0 - extent, self.x_0 + extent))

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

        return np.maximum.reduce([star1, star2, star3, disk])


def make_stellar_models(model_type, stellar_table, star_radius_a=10,
                        star_radius_b=5, cluster_radius_a=10,
                        cluster_radius_b=5, depth=5, slope=0.5):
    """
    Create the stellar (star or star cluster) texture models to be
    applied to an image.

    Given the position and flux (or magnitude) of each source
    (``stellar_table``), a list of texture models is generated.

    The radius of the star texture for each source is linearly scaled by
    the source flux as:

        .. math:: radius = radius_a + (radius_b * flux / max_flux)

    where ``max_flux`` is the maximum ``flux`` value of all the sources
    in the input ``stellar_table``.

    Parameters
    ----------
    model_type : {'star', 'star_cluster'}
        The type of the stellar texture.

    stellar_table : `~astropy.table.Table`
        A table defining the stars or star clusters.  The table must
        contain ``'xcentroid'`` and ``'ycentroid'`` columns and either a
        ``'flux'`` or ``'magnitude'`` column.

    star_radius_a : float, optional
        The intercept term in calculating the radius of the single star
        texture (see above).

    star_radius_b : float, optional
        The slope term in calculating the radius of the single star
        texture (see above).

    cluster_radius_a : float, optional
        The intercept term in calculating the radius of the star cluster
        texture (see above).

    cluster_radius_b : float, optional
        The slope term in calculating the radius of the star cluster
        texture (see above).

    depth : float, optional
        The maximum depth of the crater-like bowl of the star texture
        (for both single stars and star clusters).

    slope : float, optional
        The slope of the star texture sides (for both single stars and
        star clusters).

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

    if len(stellar_table) == 0:
        return []

    columns = ['xcentroid', 'ycentroid']
    for column in columns:
        if column not in stellar_table.colnames:
            raise ValueError('stellar_table must contain a {0} column'
                             .format(column))

    if 'flux' in stellar_table.colnames:
        fluxes = stellar_table['flux']
    else:
        if 'magnitude' not in stellar_table.colnames:
            raise ValueError('stellar_table must contain either a "flux" or '
                             '"magnitude" column'.format(column))

        fluxes = 10**(-0.4 * stellar_table['magnitude'])

    # assumes that all sources in the stellar_table are good
    max_flux = float(np.max(fluxes))
    log.debug('max_flux = "{}"'.format(max_flux))

    if model_type == 'stars':
        radius = star_radius_a + (star_radius_b * fluxes / max_flux)
    elif model_type == 'star_clusters':
        radius = cluster_radius_a + (cluster_radius_b * fluxes / max_flux)

    models = []
    base_height = 0.
    for i, source in enumerate(stellar_table):
        model = Texture(source['xcentroid'], source['ycentroid'],
                        radius[i], depth, base_height, slope)

        if model is not None:
            models.append(model)

    return models


def stellar_base_height(data, model, stellar_mask=None, selem=None):
    """
    Calculate the base height for a stellar (star or star cluster)
    texture.

    The base height is calculated from the image values just outside the
    region where the texture will be applied.

    Note that the image is clipped at the base height, it is not used
    for the actual stellar texture.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The image where the textures will be applied.

    model : `StarTexture` or `StarClusterTexture`
        A `StarTexture` or `StarClusterTexture` model object.

    stellar_mask : bool `~numpy.ndarray`, optional
        A mask image of all the stellar textures.  If input, then these
        pixels are not included in the calculation (e.g. close
        neighboring textures) for the base height.

    selem : `~numpy.ndarray`, optional
        The 2D structural element used to dilate the model mask.

    Returns
    -------
    base_height : float
        The base height of the stellar texture.  `None` is returned if
        the model does not overlap with the input image.
    """

    if stellar_mask is not None and (data.shape != stellar_mask.shape):
        raise ValueError('data and stellar_mask must have the same shape')

    if selem is None:
        selem = np.ones((3, 3))

    model_mask = np.zeros(data.shape)
    model.render(model_mask)
    model_mask = (model_mask != 0)
    if not np.any(model_mask):
        # texture contains only zeros (e.g. bad position)
        warnings.warn('stellar model does not overlap with the image.',
                      AstropyUserWarning)
        return None

    model_mask_dilated = binary_dilation(model_mask, selem)
    model_mask_xor = np.logical_xor(model_mask_dilated, model_mask)

    if stellar_mask is not None:
        border_mask = np.logical_and(model_mask_xor, ~stellar_mask)
    else:
        border_mask = model_mask_xor

    if np.any(border_mask):
        return np.max(data[border_mask])
    else:
        # no bordering pixels (e.g. texture overlaps others on all
        # sides)
        return None


def make_stellar_textures(data, stellar_tables, star_radius_a=10,
                          star_radius_b=5, cluster_radius_a=10,
                          cluster_radius_b=5, depth=5, slope=0.5,
                          exclusion_mask=None):
    """
    Make an image containing stellar textures (stars and star clusters).
    and an image containing the base heights for each texture.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The image where the textures will be applied.

    stellar_tables : dict of `~astropy.table.Table`
        A dictionary of tables defining the stellar textures, keyed by
        type ('stars' or 'star_clusters').  The dictionary can define
        both 'stars' and 'star_clusters'.  The table(s) must contain
        ``'xcentroid'`` and ``'ycentroid'`` columns and either a
        ``'flux'`` or ``'magnitude'`` column.

    star_radius_a : float, optional
        The intercept term in calculating the radius of the single star
        texture (see above).

    star_radius_b : float, optional
        The slope term in calculating the radius of the single star
        texture (see above).

    cluster_radius_a : float, optional
        The intercept term in calculating the radius of the star cluster
        texture (see above).

    cluster_radius_b : float, optional
        The slope term in calculating the radius of the star cluster
        texture (see above).

    depth : float, optional
        The maximum depth of the crater-like bowl of the star texture
        (for both single stars and star clusters).

    slope : float, optional
        The slope of the star texture sides (for both single stars and
        star clusters).

    exclusion_mask : 2D `~numpy.ndarray` (bool), optional
        A 2D boolean mask.  Textures will not be included if any portion
        of them overlaps with the exclusion mask.  For example, this is
        used to prevent overlapping textures with the central galaxy
        cusp texture.

    Returns
    -------
    textures : `~numpy.ndarray`
        An image containing the stellar (star and star cluster)
        textures.

    base_heights : `~numpy.ndarray`
        An image containing the base heights for each stellar texture.

    Notes
    -----
    To handle overlapping textures, the texture models will be sorted by
    their base heights in increasing order and then added to the output
    texture map starting with the smallest base height.
    """

    stellar_models = []
    # include both stars and star clusters
    for stellar_type, table in stellar_tables.items():
        stellar_models.extend(make_stellar_models(
            stellar_type, table, star_radius_a=star_radius_a,
            star_radius_b=star_radius_b, cluster_radius_a=cluster_radius_a,
            cluster_radius_b=cluster_radius_b, depth=depth, slope=slope))

    # create mask of all stellar textures
    stellar_mask = np.zeros(data.shape)
    for model in stellar_models:
        model.render(stellar_mask)
    stellar_mask = (stellar_mask != 0)

    # define the base heights
    base_heights = []
    good_models = []
    selem = np.ones((3, 3))
    for model in stellar_models:
        height = stellar_base_height(data, model, stellar_mask=stellar_mask,
                                     selem=selem)
        if height is not None:
            base_heights.append(height)
            good_models.append(model)

    # define the stellar textures
    stellar_textures = np.zeros(data.shape)
    base_heights = np.array(base_heights)
    idx = np.argsort(base_heights)
    base_heights = base_heights[idx]
    good_models = [good_models[i] for i in idx]

    base_heights_img = np.zeros(data.shape)
    for (model, height) in zip(good_models, base_heights):
        texture = np.zeros(data.shape)
        model.render(texture)
        mask = (texture != 0)

        if exclusion_mask is not None:
            if np.any(np.logical_and(mask, exclusion_mask)):
                continue

        stellar_textures[mask] = texture[mask]
        base_heights_img[mask] = height

    return stellar_textures, base_heights_img
