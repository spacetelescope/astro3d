"""
This module provides functions to convert an image array to STL files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from copy import deepcopy
import numpy as np
from astropy import log


def make_triangles(image):
    """
    Tessellate an image into a 3D model using triangles.

    Each pixel is split into two triangles (upper left and lower right).
    Invalid triangles (e.g. with one vertex off the edge) are excluded.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image from which to create the triangular mesh.

    Returns
    -------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.
    """

    ny, nx = image.shape
    y, x = np.indices((ny, nx))
    npts = (ny - 1) * (nx - 1)
    vertices = np.dstack((x, y, image))     # x, y, z vertices

    # upper-left triangles
    ul_tri = np.zeros(npts, 4, 3)
    ul_tri[:, 1, :] = vertices[:-1, 1:].reshape(npts, 3)   # top-left vertices
    ul_tri[:, 2, :] = vertices[:-1, :-1].reshape(npts, 3)  # bottom-left
    ul_tri[:, 3, :] = vertices[1:, 1:].reshape(npts, 3)    # top-right

    # lower-right triangles
    lr_tri = np.zeros(npts, 4, 3)
    lr_tri[:, 1, :] = vertices[1:, 1:].reshape(npts, 3)    # top-right vert.
    lr_tri[:, 2, :] = vertices[:-1, :-1].reshape(npts, 3)  # bottom-left
    lr_tri[:, 3, :] = vertices[1:, :-1].reshape(npts, 3)   # bottom-right

    #sides = make_sides(ults, lrts)
    bottom = make_model_bottom((ny-1, nx-1))
    #triangles = np.concatenate((ul_tri, lr_tri, sides, bottom))
    triangles = np.concatenate((ul_tri, lr_tri, bottom))
    triangles[:, 0, :] = calculate_normals(triangles)

    return normalize_triangles(triangles)


def make_sides(ults, lrts):
    """
    Make the model sides.

    Parameters
    ----------
    ults, lrts : NxMx3x4 `~numpy.ndarray`
        The upper-left and lower-right triangles.

    Returns
    -------
    result : Lx3x4 `~numpy.ndarray`
        The triangles for the model sides.
    """

    a = ults[0].copy()
    a[:, :, 3] = a[:, :, 1]
    a[:, 2, 3] = 0
    b = ults[:, 0].copy()
    b[:, :, 2] = b[:, :, 1]
    b[:, :, 2][:, 2] = 0
    c = ults[-1].copy()
    c[:, 1, 1:3] = c[:, 1, 1:3] + 1
    c[:, 2, 1:3] = 0
    d = ults[:, -1].copy()
    d[:, 0, 1] = d[:, 0, 1] + 1
    d[:, 0, 3] = d[:, 0, 3] + 1
    d[:, 2, 1] = 0
    d[:, 2, 3] = 0

    e = lrts[0].copy()
    e[:, 1, 1:3] = e[:, 1, 1:3] - 1
    e[:, 2, 1:3] = 0
    f = lrts[:, 0].copy()
    f[:, 0, 1] = f[:, 0, 1] - 1
    f[:, 0, 3] = f[:, 0, 3] - 1
    f[:, 2, 1] = 0
    f[:, 2, 3] = 0
    g = lrts[-1].copy()
    g[:, 1, 3] = g[:, 1, 3] + 1
    g[:, 2, 3] = 0
    h = lrts[:, -1].copy()
    h[:, 0, 2] = h[:, 0, 2] + 1
    h[:, 2, 2] = 0

    a[:, :, [1, 2]] = a[:, :, [2, 1]]
    b[:, :, [1, 2]] = b[:, :, [2, 1]]
    c[:, :, [1, 2]] = c[:, :, [2, 1]]
    d[:, :, [1, 2]] = d[:, :, [2, 1]]
    e[:, :, [1, 2]] = e[:, :, [2, 1]]
    f[:, :, [1, 2]] = f[:, :, [2, 1]]
    g[:, :, [1, 2]] = g[:, :, [2, 1]]
    h[:, :, [1, 2]] = h[:, :, [2, 1]]

    return np.concatenate((a, b, c, d, e, f, g, h))


def make_model_bottom(shape, calculate_normals=False):
    """
    Create the bottom of the model.

    The bottom is a rectangle of given ``shape`` at z=0 that is split
    into two triangles.  The triangle normals are in the -z direction.

    Parameters
    ----------
    shape : 2 tuple
        The image shape.

    calculate_normals : bool
        Set to `True` to calculate the normals.

    Returns
    -------
    result : 2x4x3 `~numpy.ndarray`
        The two triangles for the model bottom.
    """

    ny, nx = shape
    triangles = np.zeros((2, 4, 3))
    # lower-right triangle as viewed from the bottom
    triangles[0, 1:, :] = [[nx, 0, 0], [0, 0, 0], [0, ny, 0]]
    # upper-left triangle as viewed from the bottom
    triangles[1, 1:, :] = [[nx, ny, 0], [nx, 0, 0], [0, ny, 0]]

    if calculate_normals:
        # vertices were ordered such that normals are in the -z direction
        triangles[:, 0, :] = calculate_normals(triangles)

    return triangles


def calculate_normals(triangles):
    """
    Calculate the normal vectors for a set of triangles.

    The normal vector is calculated using the cross product of two
    triangle sides.  The normal direction follows the right-hand rule
    applied to the order of the triangle vertices.

    Parameters
    ----------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.

    Returns
    -------
    result : Nx1x3 `~numpy.ndarray`
        An array of normal vectors.
    """

    vertex1 = triangles[:, 1, :]
    vertex2 = triangles[:, 2, :]
    vertex3 = triangles[:, 3, :]
    vec1 = vertex2 - vertex1     # vector of first triangle side
    vec2 = vertex3 - vertex1     # vector of second triangle side
    return np.cross(vec1, vec2)


# TODO
def normalize_triangles(triangles):
    """
    Ensure model can fit on MakerBot plate.

    .. note::

        The sizing may be off when using different software
        or a different printer.

        All sizes are in mm, not inches.
    """

    xsize = triangles[:, 1:, 0].ptp()
    if xsize > 140:
        triangles = triangles * 140 / float(xsize)

    ysize = triangles[:, 1:, 1].ptp()
    if ysize > 140:
        triangles = triangles * 140 / float(ysize)

    zsize = triangles[:, 1:, 2].ptp()
    if zsize > 100:
        triangles = triangles * 100 / float(zsize)

    return triangles


def reflect_triangles(triangles):
    """
    Reflect a triangle mesh about the z axis.

    The triangle vertices are reflected about the z axis and then
    reordered such that the triangle normal is consistent with the
    right-hand rule.  The triangle normal is also reflected about the z
    axis.  All of these steps are required to properly reflect the mesh.

    Parameters
    ----------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.

    Returns
    -------
    result : Nx4x3 `~numpy.ndarray`
        The refected triangles.
    """

    triangles2 = np.copy(triangles)
    triangles2[:, 0, 2] = -triangles2[:, 0, 2]   # reflect normal about z axis
    triangles2[:, 1:, 2] = -triangles2[:, 1:, 2]      # reflect z vertices
    triangles2[:, 1:, :] = triangles2[:, 1:, :][:, ::-1]   # reorder vertices
    return triangles2


def write_binary_stl(triangles, filename):
    """
    Write a binary STL file.

    Parameters
    ----------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.

    filename : str
        The output filename.
    """

    triangles = triangles.astype('<f4')
    triangles = triangles.reshape((triangles.shape[0], 12))
    buff = np.zeros((triangles.shape[0],), dtype=('f4,'*12+'i2'))

    for n in range(12):    # fill in array by columns
        col = 'f' + str(n)
        buff[col] = triangles[:, n]

    strhdr = "binary STL format"
    strhdr += (80-len(strhdr))*" "
    ntri = len(buff)
    larray = np.zeros((1,), dtype='<u4')
    larray[0] = ntri

    with open(filename, 'wb') as f:
        f.write(strhdr)
        f.write(larray.tostring())
        buff.tofile(f)


def write_ascii_stl(triangles, filename):
    """
    Write an ASCII STL file.

    Parameters
    ----------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.

    filename : str
        The output filename.
    """

    with open(filename, 'w') as f:
        f.write("solid model\n")
        for t in triangles:
            f.write("facet normal %e %e %e\n" % tuple(t[0]))
            f.write("\touter loop\n")
            f.write("\t\tvertex %e %e %e\n" % tuple(t[1]))
            f.write("\t\tvertex %e %e %e\n" % tuple(t[2]))
            f.write("\t\tvertex %e %e %e\n" % tuple(t[3]))
            f.write("\tendloop\n")
            f.write("endfacet\n")
        f.write("endsolid model")


def write_mesh(image, filename_prefix, depth=1, double_sided=False,
               stl_format='binary', clobber=False):
    """
    Write an image to a STL file by splitting each pixel into two
    triangles.

    Parameters
    ----------
    image : ndarray
        The image to convert.

    filename_prefix : str
        The prefix of output file. ``'.stl'`` is automatically appended.

    depth : int
        The depth of the back plate. Should probably be between
        10 and 30. A thicker plate gives greater stability, but
        uses more material and has a longer build time.
        For writing JPG or PNG images, a depth of 10 probably
        suffices.

    double_sided : bool
        Set to `True` for a double-sided model, which will be a simple
        reflection.

    stl_format : {'binary', 'ascii'}
        Format for the output STL file.  The default is 'binary'.  The
        binary STL file is harder to debug, but takes up less storage
        space.

    clobber : bool, optional
        Set to `True` to overwrite any existing file.
    """

    if isinstance(image, np.ma.core.MaskedArray):
        image = deepcopy(image.data)

    triangles = make_triangles(image, depth)

    if double_sided:
        triangles = np.concatenate((triangles, reflect_triangles(triangles)))

    if stl_format == 'binary':
        write_func = write_binary_stl
    elif stl_format == 'ascii':
        write_func = write_ascii_stl
    else:
        raise ValueError('stl_format must be "binary" or "ascii"')

    filename = filename_prefix + '.stl'
    if os.path.exists(filename) and not clobber:
        raise IOError('File "{0}" already exists. Use clobber=True to '
                      'overwrite'.format(filename))
    else:
        write_func(triangles, filename)
        log.info('Saved "{0}"'.format(filename))
