"""Functions to prepare image before conversion to 3D model."""
from __future__ import division, print_function

# STDLIB
import warnings

# Anaconda
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from matplotlib.path import Path
from scipy import ndimage

# THIRD-PARTY
import photutils

# LOCAL
from . import meshcreator
from . import imageutils as iutils
from . import texture as _texture


def make_model(image, spiralarms=None, disk=None, clusters=None, height=150.,
               stars=None, base_thickness=10, double=True, has_texture=True,
               has_intensity=True):
    """Applies a number of image transformations to enable
    the creation of a meaningful 3D model from a numpy array.

    Parameters
    ----------
    image : ndarray
        Image array to process.

    spiralarms : ndarray
        Final boolean mask for spiral arms.
        This is a combined mask from all the individual spiral arms.

    disk : ndarray
        Boolean mask for disk.

    clusters : `astropy.table.Table`
        Selected star clusters.

    height : float
        The maximum height above the base.

    stars : list of `~astro3d.gui.astroObjects.Region`
        Foreground stars that need to be patched.

    base_thickness : int
        Thickness of the base so model is stable when printed
        on its side.

    double : bool
        Double- or single-sided.

    has_texture : bool
        Apply textures.

    has_intensity : bool
        Generate intensity map.

    Returns
    -------
    out_image : ndarray
        Prepared image ready for STL.

    """
    if not has_texture and not has_intensity:
        raise ValueError('Model must have textures or intensity!')

    # Texture only
    texture_image = np.zeros_like(image)

    log.info('Input image shape: {0}'.format(image.shape))
    imsz = max(image.shape)  # GUI allows only approx. 1000

    if stars is not None:
        log.info('Removing stars')
        image = remove_stars(image, stars)

    log.info('Filtering image (first pass)')
    fil_size = int(0.01 * imsz)  # imsz / 100
    image = ndimage.filters.median_filter(image, size=fil_size)
    image = np.ma.masked_equal(image, 0.0)
    image = iutils.normalize(image, True)

    log.info('Scaling top')
    image = scale_top(image, disk, spiralarms)
    image = iutils.normalize(image, True)

    log.info('Current image shape: {0}'.format(image.shape))

    # Only works for single-disk image.
    # Do this even for smooth intensity map.
    if disk is not None:
        log.info('Replacing cusp')
        cusp_rad = 0.02 * imsz  # 20

        if not has_intensity:
            cusp_percentile = None
            cusp_height = 10
        else:
            cusp_percentile = 10
            cusp_height = 40

        cusp_texture = replace_cusp(
            image, radius=cusp_rad, height=cusp_height, percent=cusp_percentile)
        cusp_mask = cusp_texture > 0
        image[cusp_mask] = cusp_texture[cusp_mask]
        texture_image[cusp_mask] = cusp_texture[cusp_mask]

    log.info('Emphasizing regions')
    image = emphasize_regions(image, (spiralarms, disk))

    log.info('Cropping image')
    if clusters is not None:
        log.info('Clusters before cropping: {0}'.format(len(clusters['xcen'])))
    image, (spiralarms, disk), clusters, iy1, iy2, ix1, ix2 = iutils.crop_image(
        image, _max=1.0, masks=(spiralarms, disk), table=clusters)
    texture_image = texture_image[iy1:iy2, ix1:ix2]
    if clusters is not None:
        log.info('Clusters after cropping: {0}'.format(len(clusters['xcen'])))

    log.info('Current image shape: {0}'.format(image.shape))

    log.info('Filtering image (second pass, height={0})'.format(height))
    image = ndimage.filters.median_filter(image, fil_size)  # 10
    image = ndimage.filters.gaussian_filter(image, 3)  # Magic?
    image = np.ma.masked_equal(image, 0)
    image = iutils.normalize(image, True, height)

    if clusters is not None and has_texture:
        c2 = None
        n_clus_added = 0

        if has_intensity:
            h_percentile = 75
        else:
            h_percentile = None

        for cluster in clusters:
            c1 = make_star_cluster(
                image, cluster, clusters['flux'][0], h_percentile=h_percentile,
                r_fac_mul=1.0)
            if c1 is None:
                continue
            if c2 is None:
                c2 = c1
            else:
                c2 = add_clusters(c2, c1)

            n_clus_added += 1

        log.info('Displaying {0} clusters'.format(n_clus_added))

        if c2 is None:
            clustermask = None
        else:
            clustermask = c2 != 0
            image[clustermask] = c2[clustermask]
            texture_image[clustermask] = c2[clustermask]
    else:
        clustermask = None

    # At this point, unsuppressed regions that are not part of disk means
    # spiral arms

    if has_texture:
        log.info('Adding textures for spiral arms and disk')
        texture = galaxy_texture(image, lmask=disk)
        if clustermask is not None:
            texture[clustermask] = 0
        image += texture
        texture_image += texture

    if isinstance(image, np.ma.core.MaskedArray):
        image = image.data

    log.info('Making base')
    if double:
        base_dist = 60  # Magic?
        base = make_base(image, dist=base_dist, height=base_thickness,
                         snapoff=True)
    else:
        base = make_base(image, height=base_thickness, snapoff=False)

    if has_intensity:
        out_image = image
    else:
        out_image = texture_image

    return out_image + base


def replace_cusp(image, location=None, radius=20, height=40, percent=10):
    """Replaces the center of the galaxy, which would be
    a sharp point, with a crator.

    Parameters
    ----------
    image : ndarray
        Image array.

    location : tuple
        ``(y, x)`` coordinate of the crator.
        If not given, use brightest pixel on image.

    radius : int
        Radius of the crator in pixels.

    height : int
        Height of the crator.

    percent : float or `None`
        Percentile between 0 and 100, inclusive, used to
        re-adjust height of marker.
        If `None` is given, then no readjustment is done.

    Returns
    -------
    cusp_texture : ndarray
        Crator values to be added.

    """
    if location is None:
        y, x = np.where(image == image.max())
    else:
        y, x = location

    if not np.isscalar(y):
        med = len(y) / 2
        y, x = y[med], x[med]

    ymin, ymax = y - radius, y + radius
    xmin, xmax = x - radius, x + radius

    if percent is None:
        top = 0.0
    else:
        top = np.percentile(image[ymin:ymax, xmin:xmax], percent)

    star = make_star(radius, height)
    smask = star != -1

    diam = 2 * radius + 1
    ymax = ymin + diam
    xmax = xmin + diam
    cusp_texture = np.zeros(image.shape)
    cusp_texture[ymin:ymax, xmin:xmax][smask] = top + star[smask]

    return cusp_texture


def remove_stars(image, stars):
    """Patches all bright/foreground stars marked as such by the user.

    Parameters
    ----------
    image : ndimage

    stars : list of `~astro3d.gui.astroObjects.Region`
        Foreground stars that need to be patched.

    Returns
    -------
    image : ndimage

    """
    if not isinstance(stars, list):
        stars = [stars]

    starmasks = [region_mask(image, star, True) for star in stars]

    for mask in starmasks:
        ypoints, xpoints = np.where(mask)
        dist = max(ypoints.ptp(), xpoints.ptp())
        xx = [xpoints, xpoints, xpoints + dist, xpoints - dist]
        yy = [ypoints + dist, ypoints - dist, ypoints, ypoints]
        newmasks = []

        for x, y in zip(xx, yy):
            try:
                pts = image[y, x]
            except IndexError as e:
                warnings.warn('remove_star() failed: {0}\n\timage[{1},{2}]'
                              ''.format(e, y, x), AstropyUserWarning)
                continue
            else:
                newmasks.append(pts)

        if len(newmasks) == 0:
            continue

        medians = [newmask.mean() for newmask in newmasks]
        index = np.argmax(medians)
        image[mask] = newmasks[index]

    return image


def emphasize_regions(image, masks, threshold=20, niter=2):
    """Emphasize science data and suppress background.

    Parameters
    ----------
    image : ndarray

    masks : list
        List of masks that mark areas of interest.
        If no mask provided (a list of `None`), entire
        image is used for calculations.

    threshold : float
        After regions are emphasized, values less than
        this are set to zero.

    niter : int
        Number of iterations.

    Returns
    -------
    image : ndarray

    """
    has_mask = [False if m is None else True for m in masks]

    for i in range(niter):
        if not any(has_mask):
            _min = image.mean()
        else:
            _min = min(image[mask].mean() for mask in masks if mask is not None)
        _min -= image.std() * 0.5
        minmask = image < _min
        image[minmask] =  image[minmask] * (image[minmask] / _min)

    # Remove low bound
    boolarray = image < threshold
    log.debug('# background pix set to zero: {0}'.format(len(image[boolarray])))
    image[boolarray] = 0

    return image


def make_star(radius, height):
    """Creates a crator-like depression that can be used
    to represent a star.

    Similar to :func:`astro3d.utils.texture.make_star`.

    """
    a = np.arange(radius * 2 + 1)
    x, y = np.meshgrid(a, a)
    r = np.sqrt((x - radius) ** 2 + (y - radius) **2)
    star = height / radius ** 2 * r ** 2
    star[r > radius] = -1
    return star


def scale_top(image, lmask=None, dmask=None, percent=30, factor=10.0):
    """Linear scale of very high values of image.

    Parameters
    ----------
    image : ndarray
        Image array.

    lmask : ndarray
        Boolean mask of disk.

    dmask : ndarray
        Boolean mask of combined spiral arms.

    percent : float
        Percentile between 0 and 100, inclusive.
        Only used if mask(s) is given.

    factor : float
        Scaling factor.

    Returns
    -------
    image : ndarray
        Scaled image.

    """
    if lmask is not None and dmask is not None:
        mask = lmask # & ~dmask
    elif lmask is None and dmask is None:
        mask = None
    elif dmask is None:
        mask = lmask
    else:
        mask = dmask

    if mask is None:
        top = image.mean() + image.std()
    else:
        top = np.percentile(image[mask], percent)

    topmask = image > top
    image[topmask] = top + (image[topmask] - top) * factor / image.max()

    return image


def galaxy_texture(galaxy, scale=None, lmask=None, hexgrid_spacing=7,
                   dots_profile='linear', dots_width=5, dots_scale=3.2,
                   lines_profile='linear', lines_width=10, lines_spacing=20,
                   lines_scale=1.2, lines_orient=0, fil_size=25,
                   fil_invscale=1.1):
    """Applies texture to the spiral arms and disk of galaxy.

    Lines to mark disk, and dots to mark spiral arms.
    Input array must be already pre-processed accordingly.

    .. note::

        ``scale`` works well for NGC 3344 (first test galaxy)
        but poorly for NGC 1566 (second test galaxy).

    Parameters
    ----------
    galaxy : ndimage
        Input array with background already suppressed.
        Unsuppressed regions that are not disk are assumed
        to be spiral arms.

    scale : number or `None`
        If only this is given, texture masks are automatically
        generated. This is ignored if ``lmask`` is given.

    lmask : ndarray or `None`
        Boolean mask for disk.
        If given, textured areas are the masked regions.

    hexgrid_spacing : int
        Spacing for :func:`~astro3d.utils.texture.hex_grid` to populate dots.

    dots_profile : {'linear', 'spherical'}
        How to arrange the dots.

    dots_width : int
        Width of each dot.

    dots_scale : float
        Scaling for dot height.

    lines_profile : {'linear', 'spherical'}
        How to draw the lines.

    lines_width, lines_spacing : int
        Width and spacing for each line.

    lines_scale : float
        Scaling for line height.

    lines_orient : float
        Orientation of the lines in degrees.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.

    fil_invscale : float
        Filter is divided by this number.

    Returns
    -------
    textured_galaxy : ndarray

    """
    if scale is None or lmask is not None:
        scale = 1

    galmax = galaxy.max()
    maxfilt = ndimage.filters.maximum_filter(galaxy, fil_size)

    # Try to automatically find disk if not given

    if lmask is None:
        log.info('No mask given; Attempting auto-find disk and spiral arms')
        fac = galmax - scale * galaxy.std()
        #fac = galmax - fil_invscale * galaxy.std()
        lmask = galaxy > fac
        dmask = galaxy <= fac
    else:
        dmask = ~lmask

    # Mark spiral arms as dots.
    # This means remove disk and non-galaxy regions from the texture.

    dotgrid = _texture.hex_grid(galaxy.shape, hexgrid_spacing)
    dots = _texture.dots(
        dots_profile, galaxy.shape, dots_width, dots_scale, dotgrid)

    dotmask = maxfilt / fil_invscale - galaxy
    dotmask[dotmask > 0] = 0
    dotmask[dotmask < 0] = 1
    dots *= dotmask
    dots[galaxy < 1] = 0
    dots[lmask] = 0

    # Mark disk as lines.
    # This means remove spiral arms and non-galaxy regions from the texture.

    lines = _texture.lines(
        lines_profile, galaxy.shape, lines_width, lines_spacing, lines_scale,
        lines_orient)
    linemask = maxfilt + 5  # Magic?
    linemask[linemask < galmax] = 1
    linemask[linemask > galmax] = 0
    lines *= linemask
    lines[dmask] = 0


    # Mark the regions with both dots and lines

    filt = ndimage.filters.maximum_filter(lines, fil_size - 15)  # 10
    dots[filt != 0] = 0

    # Debug info

    where = np.where(lines)
    log.debug('line texture locations: {0}, '
              '{1}'.format(where[0].ptp(), where[1].ptp()))

    return dots + lines


def make_star_cluster(image, peak, max_intensity, r_fac_add=15, r_fac_mul=5,
                      height=5, h_percentile=75, fil_size=10):
    """Mark star clusters for each given position.

    Parameters
    ----------
    image : ndarray

    peak : `astropy.table.Table` row
        One star cluster entry.

    max_intensity : float
        Max intensity for all the clusters.

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

    Returns
    -------
    array : ndarray

    """
    x, y, intensity = peak['xcen'], peak['ycen'], peak['flux']
    radius = r_fac_add + r_fac_mul * intensity / float(max_intensity)
    star = make_star(radius, height)
    diam = 2 * radius
    r = star.shape[0]
    dr = r / 2
    star_mask = star != -1
    dy = 0.5 * radius * np.sqrt(3)  # Magic?
    centers = [(y + dy, x), (y - dy, x + radius), (y - dy, x - radius)]
    array = np.zeros(image.shape)
    imx1 = int(x - diam)
    imx2 = int(x + diam)
    imy1 = int(y - diam)
    imy2 = int(y + diam)

    if h_percentile is None:
        _max = 0.0
    else:
        try:
            _max = np.percentile(image[imy1:imy2, imx1:imx2], h_percentile)
        except (IndexError, ValueError) as e:
            warnings.warn('Make cluster failed: {0}\n\timage[{1}:{2},{3}:{4}]'
                          ''.format(e, imy1, imy2, imx1, imx2),
                          AstropyUserWarning)
            return None

    for (cy, cx) in centers:
        xx1 = int(cx - dr)
        xx2 = xx1 + r
        yy1 = int(cy - dr)
        yy2 = yy1 + r

        try:
            array[yy1:yy2, xx1:xx2][star_mask] = _max + star[star_mask]
        except (IndexError, ValueError) as e:
            warnings.warn('Make cluster failed: {0}\n\tarray[{1}:{2},{3}:{4}]'
                          ''.format(e, yy1, yy2, xx1, xx2), AstropyUserWarning)
            return None

    filt = ndimage.filters.maximum_filter(array, fil_size)
    mask = (filt > 0) & (image > filt) & (array == 0)
    array[mask] = filt[mask]

    return array


def region_mask(image, region, interpolate, fil_size=3):
    """Uses `matplotlib.path.Path` to generate a
    numpy boolean array, which can then be used as
    a mask for a region.

    Parameters
    ----------
    image : ndarray
        Image to apply mask to.

    region : `~astro3d.gui.astroObjects.Region`
        Region to generate mask for.

    interpolate : `True`, number, or tuple
        For filter used in mask generation.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.

    Returns
    -------
    mask : ndarray or `None`
        Boolean mask for the region.

    """
    if region is None:
        return None

    y, x = np.indices(image.shape)
    y, x = y.flatten(), x.flatten()
    points = np.vstack((x, y)).T
    polygon = Path([(p.x(), p.y()) for p in region.points()])
    mask = polygon.contains_points(points).reshape(image.shape)

    if interpolate:
        if interpolate == True:  # Magic?
            interpolate = (np.percentile(image[mask], 50),
                           np.percentile(image[mask], 75))
        elif np.isscalar(interpolate):
            interpolate = (np.percentile(image[mask], 0),
                           np.percentile(image[mask], interpolate))
        else:
            interpolate = (np.percentile(image[mask], interpolate[0]),
                           np.percentile(image[mask], interpolate[1]))

        nmin, nmax = interpolate
        filtered = np.zeros(mask.shape)
        filtered[mask] = 1
        radius = min(axis.ptp() for axis in np.where(mask))
        filtered = ndimage.filters.maximum_filter(
            filtered, min(radius, image.shape[0] / 33))  # Magic?
        filtered = image * filtered
        mask = mask | ((filtered > nmin) & (filtered < nmax))
        maxfilt = ndimage.filters.maximum_filter(mask.astype(int), fil_size)
        mask = maxfilt != 0

    return mask


def combine_masks(masks):
    """Combine boolean masks into a single mask."""
    if not masks:
        return None

    return reduce(lambda m1, m2: m1 | m2, masks)


def add_clusters(cluster1, cluster2):
    """Add two star clusters together.

    Parameters
    ----------
    cluster1, cluster2 : ndarray
        See :func:`make_star_cluster`.

    Returns
    -------
    cluster1 : ndarray

    """
    mask = cluster2 != 0

    if cluster1[mask].min() < cluster2[mask].min():
        m = mask
    else:
        m = cluster1 == 0

    cluster1[m] = cluster2[m]
    return cluster1


def find_peaks(image, remove=0, num=None, threshold=8, npix=10, minpeaks=35):
    """Identifies the brightest point sources in an image.

    Parameters
    ----------
    image : ndarray
        Image to find.

    remove : int
        Number of brightest point sources to remove.

    num : int
        Number of unrejected brightest point sources to return.

    threshold, npix : int
        Parameters for ``photutils.detect_sources()``.

    minpeaks : int
        This is the minimum number of peaks that has to be found,
        if possible.

    Returns
    -------
    peaks : list
        Point sources.

    """
    while threshold >= 4:
        segm_img = photutils.detect_sources(
            image, snr_threshold=threshold, npixels=npix, mask_val=0.0)
        isophot = photutils.segment_photometry(image, segm_img)
        if len(isophot['xcen']) >= minpeaks:
            break
        else:
            threshold -= 1

    isophot.sort('flux')
    isophot.reverse()

    if remove > 0:
        isophot.remove_rows(range(remove))

    if num is not None:
        peaks = isophot[:num]
    else:
        peaks = isophot

    return peaks


def make_base(image, dist=60, height=10, snapoff=True):
    """Used to create a stronger base for printing.
    Prevents model from shaking back and forth due to printer vibration.

    .. note::

        Raft can be added using Makerware during printing process.

    Parameters
    ----------
    image : ndarray

    dist : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.
        Only used if ``snapoff=True``.

    height : int
        Height of the base.

    snapoff : bool
        If `True`, base is thin around object border so it
        can be snapped off. Set this to `False` for flat
        texture map or one sided prints.

    Returns
    -------
    max_filt : ndarray
        Array containing base values.

    """
    if snapoff:
        max_filt = ndimage.filters.maximum_filter(image, dist)
        max_filt[max_filt < 1] = -5  # Magic?
        max_filt[max_filt > 1] = 0
        max_filt[max_filt < 0] = height
    else:
        max_filt = np.zeros_like(image) + height

    return max_filt


######################################
# OLD FUNCTIONS - FOR REFERENCE ONLY #
######################################


def _scale_top_old(image):
    """Not used."""
    top = image.mean() + image.std()
    image[image > top] = top + (image[image > top] - top) * 10. / image.max()
    return image


def _replace_cusp_old(image):
    """Not used."""
    scale = 3
    jump = 1
    radius = None
    ratio = None
    while True:
        top = image.mean() + scale * image.std()
        to_replace = np.where(image > top)
        ymin, ymax = to_replace[0].min(), to_replace[0].max()
        xmin, xmax = to_replace[1].min(), to_replace[1].max()
        radius = max(xmax - xmin, ymax - ymin) / 2.
        log.info('radius = {0}'.format(radius))
        if ratio is None:
            ratio = image.shape[0] / radius
        if radius < 20:
            if jump > 0: jump *= -0.5
            scale += jump
        elif radius > 30:
            if jump < 0: jump *= -0.5
            scale += jump
        else:
            ratio = (image.shape[0] / float(radius)) / float(ratio)
            break
    star = make_star(radius, 40)
    image[ymin:ymin + 2*radius + 1, xmin:xmin + 2*radius + 1][star != -1] = top + star[star != -1]
    return image, ratio


def _prepFits(filename=None, array=None, height=150.0, spiralarms=None,
              disk=None, stars=None, clusters=None, rotation=0.0,
              filter_radius=2, replace_stars=True, texture=True, num=15,
              remove=0):
    """Prepares a fits file to be printed with a 3D printer.

    This is the original method used by Roshan to turn a numpy
    array to an STL file.

    .. note::

        Do not used. For reference only.

    Parameters
    ----------
    filename : str
        Image file to process.
        This is only used if ``array`` is not given.

    array : ndarray
        Image array to process.

    height : float
        The maximum height above the base.

    spiralarms, disk : Region
        Squish values below the average height of input regions.

    stars : list
        A list of very bright objects (usually stars) that need
        to be removed in order for proper scaling.

    clusters
        Add star clusters.

    rotation : float
        Number of degrees to rotate the image, usually to undo a
        previously rotated image.

    filter_radius : int
        The amount of smoothing to apply to the image.
        Keep between 2 and 5.

    replace_stars : bool
        Replaces high values with an artificial star that is
        better for texture.

    texture : bool
        Automatically applies a certain texture to galaxies.

    num : int
        Number of peaks to find.

    remove : list
        List of peaks to remove.

    Returns
    -------
    img : ndarray
        Prepared image ready for STL.

    """
    # TODO: ratio is not a good indicator
    ratio = None

    img = array
    if not img:
        if not filename:
            raise ValueError("Must provide either filename or array")
        # Get file
        log.info("Getting file")
        img = fits.getdata(filename)
        img = np.flipud(img)
    h, w = img.shape

    if stars:
        log.info("Removing stars")
        if isinstance(stars, dict):
            stars = [stars]
        starmasks = [region_mask(img, star, True) for star in stars]
        for mask in starmasks:
            ypoints, xpoints = np.where(mask)
            dist = max(ypoints.ptp(), xpoints.ptp())
            newmasks = [img[ypoints+dist, xpoints], img[ypoints-dist, xpoints],
                        img[ypoints, xpoints+dist], img[ypoints, xpoints-dist]]
            medians = [newmask.mean() for newmask in newmasks]
            index = np.argmax(medians)
            img[mask] = newmasks[index]

    spiralarms = [region_mask(img, arm, True) for arm in spiralarms]
    disk = region_mask(img, disk, True)
    masks = spiralarms + [disk]

    if rotation:
        log.info("Rotating Image")

        if masks:
            masks = [ndimage.interpolation.rotate(
                    mask.astype(int), rotation).astype(bool) for mask in masks]
            spiralarms = masks[:-1]
            disk = masks[-1]

        img = ndimage.interpolation.rotate(img, rotation)

        log.info("Cropping image")

        if masks:
            img, masks = iutils.crop_image(img, 1.0, masks)[:2]
            spiralarms = masks[:-1]
            disk = masks[-1]
        else:
            img = iutils.crop_image(img, 1.0)[0]

    peaks = find_peaks(img, remove, num)

    # Filter values (often gets rid of stars), normalize
    log.info("Filtering image")
    img = ndimage.filters.median_filter(img, max(h, w) / 100)
    img = np.ma.masked_equal(img, 0.0)
    img = iutils.normalize(img, True)

    # Rescale very high values (cusp of galaxy, etc.)
    log.info("Scaling top")
    img = scale_top(img, disk, combine_masks(spiralarms))
    #img = scale_top_old(img)
    img = iutils.normalize(img, True)

    if replace_stars:
        log.info("Replacing stars")
        scale = 3
        jump = 1
        radius = None
        while True:
            top = img.mean() + scale * img.std()
            to_replace = np.where(img > top)
            ymin, ymax = to_replace[0].min(), to_replace[0].max()
            xmin, xmax = to_replace[1].min(), to_replace[1].max()
            radius = max(xmax-xmin, ymax-ymin) / 2.
            if ratio == None: ratio = h / radius
            log.info(radius)
            if radius < 20:
                if jump > 0: jump *= -0.5
                scale += jump
            elif radius > 30:
                if jump < 0: jump *= -0.5
                scale += jump
            else:
                ratio = (h / float(radius)) / float(ratio)
                break
        star = make_star(radius, 40)
        img[ymin:ymin+2*radius+1, xmin:xmin+2*radius+1][star != -1] = (
            top + star[star != -1])

    # Squish lower bound
    if spiralarms or disk:
        log.info("Squishing lower bound")
        img = emphasize_regions(img, masks)

    # Get rid of 'padding'
    log.info("Cropping image")
    if masks and clusters:
        img, masks, peaks = iutils.crop_image(img, 1.0, masks, peaks)
        spiralarms = masks[:-1]
        disk = masks[-1]
    elif masks:
        img, masks = iutils.crop_image(img, 1.0, masks)[:2]
        spiralarms = masks[:-1]
        disk = masks[-1]
    elif clusters:
        img, dummy, peaks = iutils.crop_image(img, 1.0, table=peaks)
    else:
        img = iutils.crop_image(img, 1.0)[0]

    # Filter, smooth, normalize again
    log.info("Filtering image")
    img = ndimage.filters.median_filter(img, 10) # Needs to be adjustable for image size
    img = ndimage.filters.gaussian_filter(img, filter_radius)
    img = np.ma.masked_equal(img, 0)
    img = iutils.normalize(img, True, height)

    clustermask = None
    if clusters:
        log.info("Adding clusters")
        clusters = reduce(
            add_clusters,
            [make_star_cluster(img, peak, peaks['flux'][0]) for peak in peaks])
        clustermask = clusters != 0
        img[clustermask] = clusters[clustermask]

    if texture:
        log.info("Adding texture")
        #texture = galaxy_texture(img, lmask=disk, dmask=combine_masks(spiralarms))
        texture = galaxy_texture(img, 1.1)
        #texture = a_texture(img, masks)
        if clusters is not None:
            texture[clustermask] = 0
        img = img + texture

    if isinstance(img, np.ma.core.MaskedArray):
        img = img.data

    return img


def _prepareImg(filename, height=30, filter_radius=None, crop=False,
               invert=False, compress=True):
    """An old method, used for testing img2stl.to_mesh on random images.

    .. note::

        Do not use. For reference only.

    """
    img = None
    if filename[-5:] == '.fits':
        f = fits.open(filename)
        for hdu in f:
            if isinstance(hdu.data, np.ndarray):
                img = hdu.data
                break
        f.close()
    else:
        img = iutils.img2array(filename)
    if crop != False:
        if np.isscalar(crop):
            img = iutils.crop_image(img, crop)[0]
        else:
            iutils.crop_image(img, 1.0)[0]

        if np.isscalar(crop):
            img = remove_background(img, crop)
        else:
            img = remove_background(img, 1.0)

    if compress and img.shape[0] > 500:
        img = iutils.compressImage(img, 500)
    if filter_radius:
        img = ndimage.filters.gaussian_filter(img, filter_radius)
    img = img - img.min()
    if invert:
        img = img.max() - img
    img = iutils.normalize(img, True, height)
    return np.fliplr(img)
