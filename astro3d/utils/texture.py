"""This module adds textures (repeating or random in some aspect) to
an image. Separations and widths generally are specified in pixels.
In general, the intensity profile used is circular (scaled) for 1D
structures like lines, and spherical (scaled) for points, or the end
of 1D structures (e.g., end of a dash).

"""
from __future__ import division, print_function

# Anaconda
import numpy as np
from astropy import log
from numpy import random as rand


def lines(profile, shape, width, spacing, scale, orientation):
    """Create a regularly spaced set of lines.

    Parameters
    ----------
    profile : {'linear', 'spherical'}

    shape : tuple
        Image shape.

    width : int
        Thickness of the line (perpendicular).

    spacing : int
        Perpendicular spacing between lines.

    scale : float
        Scaling factor.

    orientation : float
        Orientation is in degrees.

    Returns
    -------
    im : ndarray

    """
    # start in center of image and offset both ways
    im = np.zeros(shape)
    xp, yp = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    y0, x0 = shape[0]/2, shape[1]/2
    xp = xp - x0
    yp = yp - y0
    angle = np.pi * orientation/180.
    s, c = np.sin(angle), np.cos(angle)
    x = c*xp + s*yp
    y = -s*xp + c*yp
    # compute maximum possible offsets
    noffsets = int(np.sqrt(x0**2+y0**2)/spacing)
    offsets = spacing * (np.arange(noffsets*2+1)-noffsets)
    # loop over all offsets
    for offset in offsets:
        ind = np.where(((y-offset) < width) & ((y-offset) > 0))
        if ind:
            if profile == "spherical":
                im[ind] = scale*np.sqrt((y[ind]-offset)*(width-y[ind]+offset))
            elif profile == "linear":
                im[ind] = scale*(width/2-np.abs(y[ind]-offset-width/2))
    return im


def dots(profile, shape, width, scale, locations):
    """Create dots at the given locations.

    If dots overlap, the greater value of the two is taken,
    not the sum.

    Parameters
    ----------
    profile : {'linear', 'spherical'}

    shape : tuple
        Size of the image.

    width : int
        Width of the dot.

    scale : float
        Scale applied to the dot.
        Normally, height is half-width if scale is 1.

    locations : list
        List of ``x, y`` pairs.

    Returns
    -------
    im : ndarray

    Examples
    --------
    >>> shape = (1000, 1031)
    >>> dots('linear', shape, 7, 3, locations=hex_grid(shape, 10))

    """
    im = np.zeros(shape)
    x, y = np.meshgrid(np.arange(width+1), np.arange(width+1))
    subim = 0.*x
    radius = np.sqrt((x-width/2)**2 + (y-width/2)**2)
    ind = np.where(radius < width/2)
    if profile == "spherical":
        log.debug('bozo2')
        # subim[ind] = scale * np.sqrt(width**2/4 - (x[ind]-width/2)**2 - (y[ind]-width/2)**2)
        subim[ind] = scale * np.sqrt(width**2/4 - radius[ind]**2)
    elif profile == "linear":
        subim[ind] = scale * np.abs(width/2 - radius[ind])
    else:
        raise ValueError("unknown profile option, must be spherical or linear")
    for point in locations:
        # exclude points too close to edge
        x0, y0 = point
        if not (x0 < width/2 or x0 > (shape[1]-width/2) or
            y0 < width/2 or y0 > (shape[0]-width/2)):
            xim = im[y0-width/2:y0+width/2+1,x0-width/2:x0+width/2+1]
            xim[subim>xim] = subim[subim>xim]
    return im


def point_grid(shape, spacing):
    """Generate a list of point coordinates in a regular grid.

    Parameters
    ----------
    shape : tuple
        Size of the image.

    spacing : int

    Returns
    -------
    im : ndarray

    """
    nx = int(shape[1]/spacing)
    ny = int(shape[0]/spacing)
    x, y = np.meshgrid(np.arange(nx),np.arange(ny))
    x = x*spacing
    y = y*spacing
    return zip(list(x.flat),list(y.flat))


def hex_grid(shape, spacing):
    """Generate a hex grid.

    .. note::

        ``spacing=10`` for ``dustgashexgrid``.

    Parameters
    ----------
    shape : tuple
        Size of the image.

    spacing : int
        The distance between hex side and center.

    Returns
    -------
    im : ndarray

    """
    point_spacing = 2.*spacing/np.sqrt(3.)
    nx = int(shape[1]/point_spacing)
    ny = int(shape[0]/spacing)
    # must treat even and odd rows differently
    xset = (np.arange(nx-1).astype(np.float)+.5)*point_spacing
    yset = np.arange(ny-1)*spacing
    # produce appropriate combinations
    xlist, ylist = [],[]
    for i,y in enumerate(yset):
        offset = (i % 2)*0.5*point_spacing
        for x in xset:
            xlist.append(x+offset)
            ylist.append(y)
    return zip(xlist, ylist)


def random_points(shape, spacing):
    """Generate random points.

    Parameters
    ----------
    shape : tuple
        Size of the image.

    spacing : int

    Returns
    -------
    im : ndarray

    """
    npts = shape[0]*shape[1]/spacing**2
    x = rand.random(npts)*shape[1]
    y = rand.random(npts)*shape[0]
    return zip(list(x),list(y))


def ladder(profile, shape, spacing, width, scale, length, offset):
    """Produce an image of a set of vertical line segments arranged
    in a horizonal row.

    Parameters
    ----------
    profile, shape, spacing, width, scale
        See :func:`lines`.

    length : int

    offset : int
        Number of pixels from the bottom of the image.

    Returns
    -------
    im : ndarray

    """
    # Use lines to generate initial image and then mask
    im = lines(profile, shape, width, spacing, scale, 90.)
    im[:offset] = 0
    im[offset+length:] = 0
    return im


def dashed_line(profile, shape, width, scale, pattern_length, duty_fraction,
                offset):
    """Create one horizontal dashed line with the specified
    pattern length and duty fraction.

    Parameters
    ----------
    profile, shape, width, scale
        See :func:`lines`.

    pattern_length : int

    duty_fraction : float
        For example, 0.2 means 20% is line and 80% is blank.

    offset : int
        Offset from bottom of image.

    Returns
    -------
    im : ndarray

    """
    im = lines(profile, shape, width, shape[0], scale, 0.)
    # find out where line is
    woff = int(width/2)
    ind = np.where(im[:,1] == im[:,1].max())[0][0]
    log.info('{0} {1} {2} {3}'.format(ind, width, woff, offset))
    im2 = im * 0.
    im2[offset-woff:offset-woff+width] = im[ind-woff:ind-woff+width]
    # mask out blank parts
    ramp = np.arange(shape[1])
    mask = (ramp % pattern_length) < duty_fraction*pattern_length
    return mask * im2


def make_segment_texture():
    """Generate pre-defined segment texture.

    Returns
    -------
    im : ndarray
        Square image of 200 pixels on each side.

    """
    profile = "spherical"
    width = 10
    shape = (200,200)
    lim = ladder(profile, shape, 15, width, 1, 30, 150)
    dim1 = dashed_line(profile, shape, width, 1, 30, .5, 120 )
    dim2 = dashed_line(profile, shape, width, 1, 40, .8, 80 )
    dim3 = dashed_line(profile, shape, width*2, .5, 60, .67, 40 )
    return lim+dim1+dim2+dim3


def make_star(radius, height):
    """Generate star texture.

    Parameters
    ----------
    radius : int

    height : int

    Returns
    -------
    star : ndimage

    """
    x, y = np.meshgrid(np.arange(radius*2+1), np.arange(radius*2+1))
    r = np.sqrt((x-radius)**2 + (y-radius)**2)
    star = height/radius**2 * r**2
    star[r > radius] = 0
    return star


def dobatch():
    """Batch script to generate textures and save them to JPEG images."""
    from .imutils import im2file as save

    shape = (200,200)
    sep = [25,15,10,5]
    wid = [15,10,6,3]

    for s,w in zip(sep, wid):
        log.info('{0} {1}'.format(s, w))
        for prof in ['spherical', 'linear']:
            log.info('\t{0}'.format(prof))

            lim = lines(prof,shape,w,s,1.,0.)
            save(lim, 'lines_{0}_spacing{1}_width_{2}.jpg'.format(prof,s,w))

            dim = dots(prof,shape,w,1., point_grid(shape,s))
            save(dim, 'dots_{0}_spacing{1}_width_{2}.jpg'.format(prof,s,w))

            rlim = lim.transpose()
            lim[rlim > lim] = rlim[rlim > lim]
            save(lim, 'hatch_{0}_spacing{1}_width_{2}.jpg'.format(prof,s,w))

            hdim = dots(prof,shape,w,1, hex_grid(shape,s))
            save(hdim, 'hexdots_{0}_spacing{1}_width_{2}.jpg'.format(prof,s,w))
