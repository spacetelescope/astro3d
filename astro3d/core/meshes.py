"""
This module provides functions to convert an image array to STL files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from copy import deepcopy
import numpy as np
from astropy import log


def make_triangles(image, mm_per_pixel=0.242, center_model=True):
    """
    Create a 3D model of a 2D image using triangular tessellation.

    Each pixel is split into two triangles (upper left and lower right).

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image from which to create the triangular mesh.

    mm_per_pixel : float, optional
        The physical scale of the model.

    center_model : bool, optional
        Set to `True` to center the model at ``(x, y) = (0, 0)``.  This
        will center the model on the printer plate.

    Returns
    -------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.
    """

    ny, nx = image.shape
    yy, xx = np.indices((ny, nx))
    npts = (ny - 1) * (nx - 1)
    vertices = np.dstack((xx, yy, image))     # x, y, z vertices

    # upper-left triangles
    ul_tri = np.zeros((npts, 4, 3))
    ul_tri[:, 1, :] = vertices[1:, :-1].reshape(npts, 3)   # top-left vertices
    ul_tri[:, 2, :] = vertices[:-1, :-1].reshape(npts, 3)  # bottom-left
    ul_tri[:, 3, :] = vertices[1:, 1:].reshape(npts, 3)    # top-right

    # lower-right triangles
    lr_tri = np.zeros((npts, 4, 3))
    lr_tri[:, 1, :] = vertices[1:, 1:].reshape(npts, 3)    # top-right
    lr_tri[:, 2, :] = vertices[:-1, :-1].reshape(npts, 3)  # bottom-left
    lr_tri[:, 3, :] = vertices[:-1, 1:].reshape(npts, 3)   # bottom-right

    sides = make_sides(vertices)
    bottom = make_model_bottom((ny-1, nx-1))
    triangles = np.concatenate((ul_tri, lr_tri, sides, bottom))
    triangles[:, 0, :] = calculate_normals(triangles)

    if center_model:
        triangles[:, 1:, 0] -= (nx - 1) / 2.
        triangles[:, 1:, 1] -= (ny - 1) / 2.

    return scale_triangles(triangles, mm_per_pixel=mm_per_pixel)


def make_side_triangles(side_vertices, flip_order=False):
    """
    Make the triangles for a single side.

    Parameters
    ----------
    side_vertices : NxMx3 `~numpy.ndarray`
        The (x, y, z) vertices along one side of the image.  ``N`` is
        length of the y size of the image and ``M`` is the x size of the
        image.

    flip_order : bool, optional
        Set to flip the ordering of the triangle vertices to keep the
        normals pointed "outward".

    Returns
    -------
    triangles : Nx4x3 `~numpy.ndarray`
        The triangles for a single side.
    """

    npts = len(side_vertices) - 1
    side_bottom = np.copy(side_vertices)
    side_bottom[:, 2] = 0
    ul_tri = np.zeros((npts, 4, 3))
    lr_tri = np.zeros((npts, 4, 3))

    if not flip_order:
        ul_tri[:, 1, :] = side_vertices[1:]     # top-left
        ul_tri[:, 2, :] = side_bottom[1:]       # bottom-left
        ul_tri[:, 3, :] = side_vertices[:-1]    # top-right
        lr_tri[:, 1, :] = side_vertices[:-1]    # top-right
        lr_tri[:, 2, :] = side_bottom[1:]       # bottom-left
        lr_tri[:, 3, :] = side_bottom[:-1]      # bottom-right
    else:
        ul_tri[:, 1, :] = side_vertices[:-1]    # top-left
        ul_tri[:, 2, :] = side_bottom[:-1]      # bottom-left
        ul_tri[:, 3, :] = side_vertices[1:]     # top-right
        lr_tri[:, 1, :] = side_vertices[1:]     # top-right
        lr_tri[:, 2, :] = side_bottom[:-1]      # bottom-left
        lr_tri[:, 3, :] = side_bottom[1:]       # bottom-right
    return np.concatenate((ul_tri, lr_tri))


def make_sides(vertices):
    """
    Make the model sides.

    Parameters
    ----------
    vertices : NxMx3 `~numpy.ndarray`
        The (x, y, z) vertices of the entire mesh.  ``N`` is length of
        the y size of the image and ``M`` is the x size of the image.

    Returns
    -------
    triangles : Nx4x3 `~numpy.ndarray`
        The triangles comprised the model sides.
    """

    side1 = make_side_triangles(vertices[:, 0])    # x=0
    side2 = make_side_triangles(vertices[0, :], flip_order=True)    # y=0
    side3 = make_side_triangles(vertices[:, -1], flip_order=True)   # x=-1
    side4 = make_side_triangles(vertices[-1, :])   # y=-1
    return np.concatenate((side1, side2, side3, side4))


def make_model_bottom(shape, calculate_normals=False):
    """
    Create the bottom of the model.

    The bottom is a rectangle of given ``shape`` at z=0 that is split
    into two triangles.  The triangle normals are in the -z direction.

    Parameters
    ----------
    shape : 2-tuple
        The image shape.

    calculate_normals : bool, optional
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
    result : Nx3 `~numpy.ndarray`
        An array of normal vectors.
    """

    vertex1 = triangles[:, 1, :]
    vertex2 = triangles[:, 2, :]
    vertex3 = triangles[:, 3, :]
    vec1 = vertex2 - vertex1     # vector of first triangle side
    vec2 = vertex3 - vertex1     # vector of second triangle side
    return np.cross(vec1, vec2)


def scale_triangles(triangles, mm_per_pixel=0.242):
    """
    Uniformly scale triangles given the input physical scale.

    Note that the default physical scale was derived assuming a x=1000
    pixel image, which can be printed with a maximum size of 242 mm on
    the MakerBot 5 printer.

    The maximum model sizes for the MakerBot 2 printer are:
        ``x``: 275 mm
        ``y``: 143 mm
        ``z``: 150 mm

    The maximum model sizes for the MakerBot 5 printer are:
        ``x``: 242 mm
        ``y``: 189 mm
        ``z``: 143 mm

    Parameters
    ----------
    triangles : Nx4x3 `~numpy.ndarray`
        An array of normal vectors and vertices for a set of triangles.

    mm_per_pixel : float, optional
        The physical scale of the model.

    Returns
    -------
    triangles : Nx4x3 `~numpy.ndarray`
        The scaled triangles.
    """

    triangles[:, 1:, :] *= mm_per_pixel
    model_xsize = triangles[:, 1:, 0].ptp()
    model_ysize = triangles[:, 1:, 1].ptp()
    model_zsize = triangles[:, 1:, 2].ptp()
    log.info('Model size: x={0} mm, y={1} mm, z={2} mm'.format(
        model_xsize, model_ysize, model_zsize))

    return triangles


def reflect_triangles(triangles):
    """
    Reflect a triangle mesh about the ``z`` axis.

    The triangle vertices are reflected about the ``z`` axis and then
    reordered such that the triangle normal is consistent with the
    right-hand rule.  The triangle normal is also reflected about the
    ``z`` axis.  All of these steps are required to properly reflect the
    mesh.

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
        f.write(strhdr.encode())
        f.write(larray.tobytes())
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


def write_mesh(image, filename_prefix, mm_per_pixel=0.242, double_sided=False,
               stl_format='binary', clobber=False):
    """
    Write an image to a STL file by splitting each pixel into two
    triangles.

    Parameters
    ----------
    image : `~numpy.ndarray`
        The image to convert.

    filename_prefix : str
        The prefix of output file. ``'.stl'`` is automatically appended.

    mm_per_pixel : float, optional
        The physical scale of the model.

    double_sided : bool, optional
        Set to `True` for a double-sided model, which will be a simple
        reflection.

    stl_format : {'binary', 'ascii'}, optional
        Format for the output STL file.  The default is 'binary'.  The
        binary STL file is harder to debug, but takes up less storage
        space.

    clobber : bool, optional
        Set to `True` to overwrite any existing file.
    """

    if isinstance(image, np.ma.core.MaskedArray):
        image = deepcopy(image.data)

    triangles = make_triangles(image, mm_per_pixel=mm_per_pixel)

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
