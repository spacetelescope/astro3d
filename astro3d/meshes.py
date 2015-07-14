"""
This module provides functions to convert an image array to STL files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import deepcopy
from struct import unpack
import numpy as np
from astropy import log


def get_triangles(npimage, depth=10):
    """
    Make upper left and lower right triangles.

    Automatically exclude invalid triangles (triangles with
    one vertex off the edge).

    Parameters
    ----------
    npimage : ndarray

    depth : int

    Returns
    -------
    normalized_triset : ndarray
        An array of triangles.
    """

    npimage = npimage + depth
    h, w = npimage.shape
    y, x = np.indices((h, w))
    cube = np.dstack((x, y, npimage))
    ults = np.zeros((h-1, w-1, 3, 4))
    lrts = np.zeros((h-1, w-1, 3, 4))
    ults[:, :, :, 1] = cube[:-1, :-1]
    ults[:, :, :, 2] = cube[:-1, 1:]
    ults[:, :, :, 3] = cube[1:, :-1]
    lrts[:, :, :, 1] = cube[1:, 1:]
    lrts[:, :, :, 2] = cube[1:, :-1]
    lrts[:, :, :, 3] = cube[:-1, 1:]
    sides = make_sides(ults, lrts)
    ults = ults.reshape(((ults.shape[0])*(ults.shape[1]), 3, 4))
    lrts = lrts.reshape(((lrts.shape[0])*(lrts.shape[1]), 3, 4))
    triset = get_cross(np.concatenate((ults, lrts, sides)))
    triset = np.swapaxes(triset, 1, 2).copy()
    triset = np.concatenate((triset, make_bottom(h-1, w-1)))

    return normalize_triangles(triset)


def make_sides(ults, lrts):
    """Creates the sides of the base."""

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


def make_bottom(height, width):
    """Create the bottom of the base."""

    bottom = np.array([[[width, 0, 0], [0, 0, 0], [0, height, 0]],
                       [[width, height, 0], [width, 0, 0], [0, height, 0]]])
    triset = np.zeros((2, 4, 3))
    triset[0, 1:] = bottom[0]
    triset[1, 1:] = bottom[1]

    for tri in triset:
        v1 = tri[2] - tri[1]
        v2 = tri[3] - tri[1]
        tri[0] = np.cross(v1, v2)

    return triset


def normalize_triangles(triset):
    """
    Ensure model can fit on MakerBot plate.

    .. note::

        The sizing may be off when using different software
        or a different printer.

        All sizes are in mm, not inches.
    """

    xsize = triset[:, 1:, 0].ptp()
    if xsize > 140:
        triset = triset * 140 / float(xsize)

    ysize = triset[:, 1:, 1].ptp()
    if ysize > 140:
        triset = triset * 140 / float(ysize)

    zsize = triset[:, 1:, 2].ptp()
    if zsize > 100:
        triset = triset * 100 / float(zsize)

    return triset


def get_cross(triset):
    """
    Set the normal vector for each triangle.

    This is necessary for some 3D printing software, including MakerWare.
    """

    t1 = triset[:, :, 1]
    t2 = triset[:, :, 2]
    t3 = triset[:, :, 3]
    v1 = t2 - t1
    v2 = t3 - t1
    triset[:, :, 0] = np.cross(v1, v2)

    return triset


def to_mesh(image, filename, depth=1, double_sided=False, _ascii=False):
    """
    Write an image to STL file by splitting each pixel into two
    triangles.

    Parameters
    ----------
    image : ndarray
        The image to convert.

    filename : str
        Prefix of output file. ``.stl`` is automatically appended.

    depth : int
        The depth of the back plate. Should probably be between
        10 and 30. A thicker plate gives greater stability, but
        uses more material and has a longer build time.
        For writing JPG or PNG images, a depth of 10 probably
        suffices.

    double_sided : bool
        Set to `True` for a double-sided model.

    _ascii : bool
        Write in binary or ASCII format.
        Binary STL file is harder to debug, but takes up less
        storage space.
    """

    if isinstance(image, np.ma.core.MaskedArray):
        npimage = deepcopy(image.data)
    else:
        npimage = deepcopy(image)

    triset = get_triangles(npimage, depth)

    if double_sided:
        triset2 = triset.copy()
        triset2[:, 0] = -triset2[:, 0]
        triset2[:, 1:, 2] = -triset2[:, 1:, 2]
        triset = np.concatenate((triset, triset2))

    if _ascii:
        write_func = write_ascii
    else:
        write_func = write_binary

    fname = filename + '.stl'
    write_func(triset, fname)
    log.info('{0} saved'.format(fname))


def write_binary(triset, filename):
    """
    Write a binary STL file.

    Parameters
    ----------
    triset : ndarray
        A set of triangles and normal vectors.

    filename : str
        Output filename.
    """

    triset = triset.astype('<f4')
    triset = triset.reshape((triset.shape[0], 12))
    buff = np.zeros((triset.shape[0],), dtype=('f4,'*12+'i2'))

    for n in range(12):    # Fills in array by column
        col = 'f' + str(n)
        buff[col] = triset[:, n]

    # Took the header straight from stl.py
    strhdr = "binary STL format"
    strhdr += (80-len(strhdr))*" "
    ntri = len(buff)
    larray = np.zeros((1,), dtype='<u4')
    larray[0] = ntri

    with open(filename, 'wb') as f:
        f.write(strhdr)
        f.write(larray.tostring())
        buff.tofile(f)


def write_ascii(triset, filename):
    """
    Like :func:`write_binary` but in ASCII format.

    .. note::

        Recommended for debugging only.
    """

    with open(filename, 'w') as f:
        f.write("solid bozo\n")
        for t in triset:
            f.write("facet normal %e %e %e\n" % tuple(t[0]))
            f.write("\touter loop\n")
            f.write("\t\tvertex %e %e %e\n" % tuple(t[1]))
            f.write("\t\tvertex %e %e %e\n" % tuple(t[2]))
            f.write("\t\tvertex %e %e %e\n" % tuple(t[3]))
            f.write("\tendloop\n")
            f.write("endfacet\n")
        f.write("endsolid bozo")


def read_binary(filename):
    """
    Read binary STL file.

    http://sukhbinder.wordpress.com/2013/11/28/binary-stl-file-reader-in-python-powered-by-numpy/

    Parameters
    ----------
    filename : str

    Returns
    -------
    header, normals, points, v1, v2, v3
    """

    with open(filename, 'rb') as fp:
        header = fp.read(80)
        nn = fp.read(4)
        numtri = unpack('i', nn)[0]
        record_dtype = np.dtype(
            [('normals', np.float32, (3, )),
             ('Vertex1', np.float32, (3, )),
             ('Vertex2', np.float32, (3, )),
             ('Vertex3', np.float32, (3, )),
             ('atttr', '<i2', (1, ))])
        data = np.fromfile(fp, dtype=record_dtype, count=numtri)

    normals = data['normals']
    v1 = data['Vertex1']
    v2 = data['Vertex2']
    v3 = data['Vertex3']

    p = np.append(v1, v2, axis=0)
    p = np.append(p, v3, axis=0)
    points = np.array(list(set(tuple(p1) for p1 in p)))

    return header, normals, points, v1, v2, v3
