"""Functions to prepare image before conversion to 3D model."""
from __future__ import division, print_function

# STDLIB
import warnings
from collections import defaultdict

# Anaconda
import numpy as np
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
from scipy import ndimage

# THIRD-PARTY
import photutils

# LOCAL
from . import imageutils as iutils
from . import texture as _texture


def make_model(image, region_masks=defaultdict(list), peaks={}, height=150.0,
               base_thickness=10, clus_r_fac_add=15, clus_r_fac_mul=1,
               star_r_fac_add=15, star_r_fac_mul=1,
               layer_order=['lines', 'dots'], double=True, has_texture=True,
               has_intensity=True, is_spiralgal=False):
    """Apply a number of image transformations to enable
    the creation of a meaningful 3D model for an astronomical
    image from a Numpy array.

    Boundaries are set by :func:`emphasize_regions` and
    :func:`~astro3d.utils.imageutils.crop_image`.

    Parameters
    ----------
    image : ndarray
        Image array to process.

    region_masks : dict
        A dictionary that maps each texture type to a list of
        corresponding boolean masks.

    peaks : dict
        A dictionary that maps each texture type to a `astropy.table.Table`.

    height : float
        The maximum height above the base.

    base_thickness : int
        Thickness of the base so model is stable when printed
        on its side.

    clus_r_fac_add, clus_r_fac_mul, star_r_fac_add, star_r_fac_mul : float
        Crater radius scaling factors for star clusters and stars,
        respectively. See :func:`make_star_cluster`.

    layer_order : list
        Order of texture layers (dots, lines) to apply.
        Top/foreground layer overwrites the bottom/background.
        This is only used if ``is_spiralgal=False`` and ``has_texture=True``.

    double : bool
        Double- or single-sided.

    has_texture : bool
        Apply textures.

    has_intensity : bool
        Generate intensity map.

    is_spiralgal : bool
        Special processing for a single spiral galaxy.

    Returns
    -------
    out_image : ndarray
        Prepared image ready for STL.

    """
    if not has_texture and not has_intensity:
        raise ValueError('Model must have textures or intensity!')

    smooth_key = 'smooth'
    dots_key = 'dots'
    lines_key = 'lines'
    disk = None
    spiralarms = None

    # Old logic specific to single spiral galaxy
    if is_spiralgal:
        smooth_key = 'stars'
        lines_key = 'disk'
        dots_key = 'spiral'

        if len(region_masks[lines_key]) > 0:
            disk = region_masks[lines_key][0]

        if len(region_masks[dots_key]) > 0:
            spiralarms = region_masks[dots_key][0]

    log.info('Input image shape: {0}'.format(image.shape))
    imsz = max(image.shape)  # GUI allows only approx. 1000

    log.info('Smoothing {0} region(s)'.format(len(region_masks[smooth_key])))
    image = remove_stars(image, region_masks[smooth_key])

    log.info('Filtering image (first pass)')
    fil_size = int(0.01 * imsz)  # imsz / 100
    image = ndimage.filters.median_filter(image, size=fil_size)
    image = np.ma.masked_equal(image, 0.0)
    image = iutils.normalize(image, True)

    if is_spiralgal:
        log.info('Scaling top')
        image = scale_top(image, mask=disk)
        image = iutils.normalize(image, True)

    log.info('Current image shape: {0}'.format(image.shape))

    # Only works for single-disk image.
    # Do this even for smooth intensity map to avoid sharp peak in model.
    cusp_mask = None
    cusp_texture_flat = None
    if disk is not None:
        log.info('Replacing cusp')
        cusp_rad = 0.02 * imsz  # 20
        cusp_texture = replace_cusp(
            image, mask=disk, radius=cusp_rad, height=40, percent=10)
        cusp_mask = cusp_texture > 0

        if not has_intensity:
            cusp_texture_flat = replace_cusp(
                image, mask=disk, radius=cusp_rad, height=10, percent=None)

        image[cusp_mask] = cusp_texture[cusp_mask]

    log.info('Emphasizing regions')
    image = emphasize_regions(
        image, region_masks[dots_key] + region_masks[lines_key])

    log.info('Cropping image')
    image, iy1, iy2, ix1, ix2 = iutils.crop_image(image, _max=1.0)
    log.info('Current image shape: {0}'.format(image.shape))

    log.info('Cropping region masks')
    croppedmasks = defaultdict(list)
    for key, mlist in region_masks.iteritems():
        if key == smooth_key:  # Smoothing already done
            continue
        for mask in mlist:
            croppedmasks[key].append(mask[iy1:iy2, ix1:ix2])
    region_masks = croppedmasks
    if is_spiralgal:
        if len(region_masks[lines_key]) > 0:
            disk = region_masks[lines_key][0]
        if len(region_masks[dots_key]) > 0:
            spiralarms = region_masks[dots_key][0]
    if cusp_mask is not None:
        cusp_mask = cusp_mask[iy1:iy2, ix1:ix2]
    if cusp_texture_flat is not None:
        cusp_texture_flat = cusp_texture_flat[iy1:iy2, ix1:ix2]

    if 'clusters' in peaks:
        clusters = peaks['clusters']
        log.info('Clusters before cropping: {0}'.format(len(clusters)))
        clusters = clusters[(clusters['xcen'] > ix1) &
                            (clusters['xcen'] < ix2 - 1) &
                            (clusters['ycen'] > iy1) &
                            (clusters['ycen'] < iy2 - 1)]
        clusters['xcen'] -= ix1
        clusters['ycen'] -= iy1
        log.info('Clusters after cropping: {0}'.format(len(clusters)))
    else:
        clusters = []

    if 'stars' in peaks:
        markstars = peaks['stars']
        log.info('Stars before cropping: {0}'.format(len(markstars)))
        markstars = markstars[(markstars['xcen'] > ix1) &
                              (markstars['xcen'] < ix2 - 1) &
                              (markstars['ycen'] > iy1) &
                              (markstars['ycen'] < iy2 - 1)]
        markstars['xcen'] -= ix1
        markstars['ycen'] -= iy1
        log.info('Stars after cropping: {0}'.format(len(markstars)))
    else:
        markstars = []

    log.info('Filtering image (second pass, height={0})'.format(height))
    image = ndimage.filters.median_filter(image, fil_size)  # 10
    image = ndimage.filters.gaussian_filter(image, 3)  # Magic?
    image = np.ma.masked_equal(image, 0)
    image = iutils.normalize(image, True, height)

    # Texture layer that is added later overwrites previous layers if overlap
    if has_texture:

        # Dots and lines

        if is_spiralgal:
            # At this point, unsuppressed regions that are not part of disk
            # means spiral arms
            log.info('Adding textures for spiral arms and disk')
            texture_layer = galaxy_texture(image, lmask=disk, cmask=cusp_mask)

        else:
            texture_layer = np.zeros(image.shape)

            # Apply layers from bottom up
            for layer_key in layer_order[::-1]:
                if layer_key == dots_key:
                    texture_func = dots_from_mask
                elif layer_key == lines_key:
                    texture_func = lines_from_mask
                else:
                    log.warning('{0} is not a valid texture, skipping...'
                                ''.format(layer_key), AstropyUserWarning)
                    continue

                log.info('Adding {0}'.format(layer_key))
                for mask in region_masks[layer_key]:
                    cur_texture = texture_func(image, mask=mask)
                    texture_layer[mask] = cur_texture[mask]

        image += texture_layer

        # Stars and star clusters

        clustexarr = None

        if has_intensity:
            h_percentile = 75
        else:
            h_percentile = None

        # Add star clusters

        n_clus_added = 0

        if len(clusters) > 0:
            maxclusflux = max(clusters['flux'])

        for cluster in clusters:
            c1 = make_star_cluster(
                image, cluster,  maxclusflux, h_percentile=h_percentile,
                r_fac_add=clus_r_fac_add, r_fac_mul=clus_r_fac_mul, n_craters=3)
            if not np.any(c1):
                continue
            if clustexarr is None:
                clustexarr = c1
            else:
                clustexarr = add_clusters(clustexarr, c1)
            n_clus_added += 1

        log.info('Displaying {0} clusters'.format(n_clus_added))

        # Add individual stars

        n_star_added = 0

        if len(markstars) > 0:
            maxstarflux = max(markstars['flux'])

        for mstar in markstars:
            s1 = make_star_cluster(
                image, mstar, maxstarflux, h_percentile=h_percentile,
                r_fac_add=star_r_fac_add, r_fac_mul=star_r_fac_mul, n_craters=1)
            if not np.any(s1):
                continue
            if clustexarr is None:
                clustexarr = s1
            else:
                clustexarr = add_clusters(clustexarr, s1)
            n_star_added += 1

        log.info('Displaying {0} stars'.format(n_star_added))

        # Both stars and star clusters share the same mask

        if clustexarr is not None:
            clustermask = clustexarr != 0
            if has_intensity:
                image[clustermask] = clustexarr[clustermask]
            else:
                texture_layer[clustermask] = clustexarr[clustermask]

        # For texture-only model, need to add cusp to texture layer
        if not has_intensity and cusp_mask is not None:
            texture_layer[cusp_mask] = cusp_texture_flat[cusp_mask]

    # endif has_texture

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
        out_image = image + base
    else:
        out_image = texture_layer + base

    return out_image


def remove_stars(image, starmasks):
    """Patches all bright/foreground stars marked as such by the user.

    Parameters
    ----------
    image : ndimage

    starmasks : list
        List of boolean masks of foreground stars that need to be patched.

    Returns
    -------
    image : ndimage

    """
    for mask in starmasks:
        ypoints, xpoints = np.where(mask)
        dist = max(ypoints.ptp(), xpoints.ptp())
        xx = [xpoints, xpoints, xpoints + dist, xpoints - dist]
        yy = [ypoints + dist, ypoints - dist, ypoints, ypoints]
        newmasks = []
        warn_msg = []

        for x, y in zip(xx, yy):
            try:
                pts = image[y, x]
            except IndexError as e:
                warn_msg.append('\t{0}'.format(e))
            else:
                newmasks.append(pts)

        if len(newmasks) == 0:
            warnings.warn('remove_stars() failed:\n{0}'.format(
                '\n'.join(warn_msg)), AstropyUserWarning)
            continue

        medians = [newmask.mean() for newmask in newmasks]
        index = np.argmax(medians)
        image[mask] = newmasks[index]

    return image


def scale_top(image, mask=None, percent=30, factor=10.0):
    """Linear scale of very high values of image.

    Parameters
    ----------
    image : ndarray
        Image array.

    mask : ndarray
        Mask of region with very high values. E.g., disk.

    percent : float
        Percentile between 0 and 100, inclusive.
        Only used if ``mask`` is given.

    factor : float
        Scaling factor.

    Returns
    -------
    image : ndarray
        Scaled image.

    """
    if mask is None:
        top = image.mean() + image.std()
    else:
        top = np.percentile(image[mask], percent)

    topmask = image > top
    image[topmask] = top + (image[topmask] - top) * factor / image.max()

    return image


def replace_cusp(image, mask=None, radius=20, height=40, percent=10):
    """Replaces the center of the galaxy, which would be
    a sharp point, with a crater.

    Parameters
    ----------
    image : ndarray
        Image array.

    mask : ndarray
        Mask of the disk.

    radius : int
        Radius of the crater in pixels.

    height : int
        Height of the crater.

    percent : float or `None`
        Percentile between 0 and 100, inclusive, used to
        re-adjust height of marker.
        If `None` is given, then no readjustment is done.

    Returns
    -------
    cusp_texture : ndarray
        Crater values to be added.

    """
    cusp_texture = np.zeros(image.shape)

    if mask is None:
        y, x = np.where(image == image.max())
    else:
        a = np.ma.array(image.data, mask=~mask)
        y, x = np.where(a == a.max())

    if not np.isscalar(y):
        med = len(y) // 2
        y, x = y[med], x[med]

    log.info('\tCenter of galaxy at X={0} Y={1}'.format(x, y))

    ymin = max(y - radius, 0)
    ymax = min(y + radius, image.shape[0])
    xmin = max(x - radius, 0)
    xmax = min(x + radius, image.shape[1])

    if percent is None:
        top = 0.0
    else:
        top = np.percentile(image[ymin:ymax, xmin:xmax], percent)

    star = make_star(radius, height)
    smask = star != -1

    diam = 2 * radius + 1
    ymax = min(ymin + diam, image.shape[0])
    xmax = min(xmin + diam, image.shape[1])
    cusp_texture[ymin:ymax, xmin:xmax][smask] = top + star[smask]

    return cusp_texture


def emphasize_regions(image, masks, threshold=20, niter=2):
    """Emphasize science data and suppress background.

    Parameters
    ----------
    image : ndarray

    masks : list
        List of masks that mark areas of interest.
        If no mask provided (empty list), entire
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
    n_masks = len(masks)

    for i in range(niter):
        if n_masks < 1:
            _min = image.mean()
        else:
            _min = min([image[mask].mean() for mask in masks])
        _min -= image.std() * 0.5
        minmask = image < _min
        image[minmask] =  image[minmask] * (image[minmask] / _min)

    # Remove low bound
    boolarray = image < threshold
    log.debug('# background pix set to zero: {0}'.format(len(image[boolarray])))
    image[boolarray] = 0

    return image


def make_star(radius, height):
    """Creates a crater-like depression that can be used
    to represent a star.

    Similar to :func:`astro3d.utils.texture.make_star`.

    """
    a = np.arange(radius * 2 + 1)
    x, y = np.meshgrid(a, a)
    r = np.sqrt((x - radius) ** 2 + (y - radius) **2)
    star = height / radius ** 2 * r ** 2
    star[r > radius] = -1
    return star


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
    #log.info('\tcluster radius = {0}'.format(radius, r_fac_add, r_fac_mul))
    star = make_star(radius, height)
    diam = 2 * radius
    r = star.shape[0]
    dr = r / 2
    star_mask = star != -1
    imx1 = int(x - diam)
    imx2 = int(x + diam)
    imy1 = int(y - diam)
    imy2 = int(y + diam)

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
        except (IndexError, ValueError) as e:
            warnings.warn('Make star/cluster failed: {0}\n\timage[{1}:{2},'
                          '{3}:{4}]'.format(e, imy1, imy2, imx1, imx2),
                          AstropyUserWarning)
            return array

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
            return np.zeros(image.shape)

    filt = ndimage.filters.maximum_filter(array, fil_size)
    mask = (filt > 0) & (image > filt) & (array == 0)
    array[mask] = filt[mask]

    return array


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


def dots_from_mask(image, mask=None, hexgrid_spacing=7, dots_width=5,
                   dots_scale=3.2, fil_size=25, fil_invscale=1.1):
    """Apply dots texture to region marked by given mask.

    Parameters
    ----------
    image : ndarray
        Input image with background already suppressed.

    mask : ndarray
        Boolean mask of the region to be marked.
        If not given, it is guessed from image values.

    hexgrid_spacing : int
        Spacing for :func:`~astro3d.utils.texture.hex_grid` to
        populate dots.

    dots_width : int
        Width of each dot.

    dots_scale : float
        Scaling for dot height.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.
        Only used if ``mask`` is not given.

    fil_invscale : float
        Filter is divided by this number.
        Only used if ``mask`` is not given.

    Returns
    -------
    dots : ndarray
        Output array with texture values.

    """
    dots = _texture.dots('linear', image.shape, dots_width, dots_scale,
                         _texture.hex_grid(image.shape, hexgrid_spacing))

    if mask is None:
        maxfilt = ndimage.filters.maximum_filter(image, fil_size)
        dotmask = maxfilt / fil_invscale - image
        dotmask[dotmask > 0] = 0
        dotmask[dotmask < 0] = 1
    else:
        dotmask = np.zeros_like(dots)
        dotmask[mask] = 1

    # Exclude background
    dotmask[image < 1] = 0

    dots *= dotmask
    return dots


def lines_from_mask(image, mask=None, lines_width=10, lines_spacing=20,
                    lines_scale=1.2, lines_orient=0, fil_size=25):
    """Apply lines texture to region marked by given mask.

    Parameters
    ----------
    image : ndarray
        Input image with background already suppressed.

    mask : ndarray
        Boolean mask of the region to be marked.
        If not given, it is guessed from image values.

    lines_width, lines_spacing : int
        Width and spacing for each line.

    lines_scale : float
        Scaling for line height.

    lines_orient : float
        Orientation of the lines in degrees.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.
        Only used if ``mask`` is not given.

    Returns
    -------
    lines : ndarray
        Output array with texture values.

    """
    lines = _texture.lines('linear', image.shape, lines_width, lines_spacing,
                           lines_scale, lines_orient)

    if mask is None:
        imgmax = image.max()
        maxfilt = ndimage.filters.maximum_filter(image, fil_size)
        linemask = maxfilt + 5  # Magic?
        linemask[linemask < imgmax] = 1
        linemask[linemask > imgmax] = 0
    else:
        linemask = np.zeros_like(lines)
        linemask[mask] = 1

    # Exclude background
    linemask[image < 1] = 0

    lines *= linemask
    return lines


def galaxy_texture(galaxy, lmask=None, cmask=None, scale=1.0, hexgrid_spacing=7,
                   dots_width=5, dots_scale=3.2, lines_width=10,
                   lines_spacing=20, lines_scale=1.2, lines_orient=0,
                   fil_size=25, fil_invscale=1.1):
    """Apply texture to the spiral arms and disk of galaxy.

    Lines to mark disk, and dots to mark spiral arms.
    Input array must be already pre-processed accordingly.

    .. note::

        ``scale`` works well for NGC 3344 (first test galaxy)
        but poorly for NGC 1566 (second test galaxy).

    Parameters
    ----------
    galaxy : ndarray
        Input array with background already suppressed.
        Unsuppressed regions that are not disk are assumed
        to be spiral arms.

    lmask, cmask : ndarray or `None`
        Boolean masks for disk and cusp.
        If not given, it is guessed from image values.

    scale : float
        Scaling for auto texture generation without mask.
        This is only used if ``lmask`` is not given.

    hexgrid_spacing, dots_width, dots_scale
        See :func:`dots_from_mask`.

    lines_width, lines_spacing, lines_scale, lines_orient
        See :func:`lines_from_mask`.

    fil_size : int
        Filter size for :func:`~scipy.ndimage.filters.maximum_filter`.

    fil_invscale : float
        Filter is divided by this number.

    Returns
    -------
    textured_galaxy : ndarray
        Output array with texture values.

    """
    # Try to automatically find disk if not given
    if lmask is None:
        log.info('No mask given; Attempting auto-find disk and spiral arms')
        galmax = galaxy.max()
        fac = galmax - scale * galaxy.std()
        #fac = galmax - fil_invscale * galaxy.std()
        lmask = galaxy > fac
        dmask = galaxy <= fac
    else:
        dmask = ~lmask

    # Mark spiral arms as dots.
    dots = dots_from_mask(
        galaxy, hexgrid_spacing=hexgrid_spacing, dots_width=dots_width,
        dots_scale=dots_scale, fil_size=fil_size, fil_invscale=fil_invscale)

    # Mark disk as lines.
    lines = lines_from_mask(
        galaxy, lines_width=lines_width, lines_spacing=lines_spacing,
        lines_scale=lines_scale, lines_orient=lines_orient, fil_size=fil_size)

    # Remove disk from spiral arms texture, and vice versa.
    dots[lmask] = 0
    lines[dmask] = 0
    if cmask is not None:
        lines[cmask] = 0
    filt = ndimage.filters.maximum_filter(lines, fil_size - 15)  # 10
    dots[filt != 0] = 0

    # Debug info
    where = np.where(lines)
    log.debug('line texture locations: {0}, '
              '{1}'.format(where[0].ptp(), where[1].ptp()))

    return dots + lines


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
        max_filt = np.zeros(image.shape) + height

    return max_filt


def combine_masks(masks):
    """Combine boolean masks into a single mask."""
    if len(masks) == 0:
        return masks

    return reduce(lambda m1, m2: m1 | m2, masks)


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
    from astropy.io import fits

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
        starmasks = [star.to_mask(img) for star in stars]
        for mask in starmasks:
            ypoints, xpoints = np.where(mask)
            dist = max(ypoints.ptp(), xpoints.ptp())
            newmasks = [img[ypoints+dist, xpoints], img[ypoints-dist, xpoints],
                        img[ypoints, xpoints+dist], img[ypoints, xpoints-dist]]
            medians = [newmask.mean() for newmask in newmasks]
            index = np.argmax(medians)
            img[mask] = newmasks[index]

    spiralarms = [arm.to_mask(img) for arm in spiralarms]
    disk = disk.to_mask(img)
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
