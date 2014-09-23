"""Data objects for GUI."""
from __future__ import division, print_function

# STDLIB
import sys
from copy import deepcopy

# Anaconda
import numpy as np
from astropy import log
from astropy.io import fits
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# LOCAL
from img2stl import imageprep, meshcreator
from img2stl.imageutils import split_image


class File(object):
    """Data holder.

    A class that stores all the information necessary to
    construct the STL file from the original numpy array.

    .. note::

        This class should be renamed.

    Parameters
    ----------
    data : ndarray
        Original data from FITS or JPEG file.

    image : QPixmap
        Scaled image for display purposes.

    height : float
        Height of the model.

    Attributes
    ----------
    data, image, height
        Same as inputs.

    spiralarms : list of ``Region``
        Spiral arms of the galaxy.

    disk : ``Region``
        Disk of the galaxy.

    stars : list of ``Region``
        Foreground stars that need to be patched.

    clusters : ``astropy.Table``
        Selected star clusters.

    """
    def __init__(self, data, image, height=150.0):
        super(File, self).__init__()
        self.data = data
        self.image = image
        self.height = height
        self.spiralarms = []
        self.disk = None
        self.stars = []
        self.clusters = None

    def scale(self):
        """Return the ratio by which the image has been scaled."""
        return self.image.height() / float(self.data.shape[0])

    def scaleRegions(self):
        """Scale the coordinates of all Regions from the display
        image to the corresponding locations on the actual data.

        """
        self.spiralarms = [reg.scaledRegion(self) for reg in self.spiralarms
                           if reg is not None]
        self.stars = [reg.scaledRegion(self) for reg in self.stars
                      if reg is not None]

        if self.disk is not None:
            self.disk = self.disk.scaledRegion(self)

    def make_3d(self, fname, depth=1., double=False, _ascii=False,
                has_texture=True, has_intensity=True, split_halves=True):
        """Generate STL file.

        #. Scale regions.
        #. Create boolean masks for the regions.
        #. Use ``imageprep.make_model()`` to perform transformations
           on the array and obtains a new array ready for STL creator.
        #. Split array into two halves (optional).
        #. Use ``meshcreator.to_mesh()`` to make STL file(s).

        Parameters
        ----------
        fname : str or QString
            Filename prefix. The ``.stl`` extension is automatically
            added. Also see ``split_halves``.

        depth : float
            Depth of back plate.

        double : bool
            Double- or single-sided.

        _ascii : bool
            ASCII or binary format.

        has_texture : bool
            Apply textures.

        has_intensity : bool
            Generate intensity map.

        split_halves : bool
            If `True`, image is split into two halves and the files
            will have ``1.stl`` or ``2.stl`` added to ``fname``,
            respectively.

        """
        self.scaleRegions()
        self.data = np.flipud(self.data)

        spiralmask = imageprep.combine_masks(
            [imageprep.region_mask(self.data, reg, True)
             for reg in self.spiralarms if reg is not None])
        disk = imageprep.region_mask(self.data, self.disk, True)

        # make_model() modifies input in-place, so pass in a copy instead
        model = imageprep.make_model(
            deepcopy(self.data), spiralarms=spiralmask, disk=disk,
            clusters=self.clusters, height=self.height, stars=self.stars,
            double=double, has_texture=has_texture, has_intensity=has_intensity)

        # Input filename might be QString.
        # Remove any .stl suffix because it is added by to_mesh()
        fname = str(fname)
        if fname.lower().endswith('.stl'):
            fname = fname[:-4]

        if split_halves:
            model1, model2 = split_image(model)
            meshcreator.to_mesh(model1, fname + '_1', depth, double, _ascii)
            meshcreator.to_mesh(model2, fname + '_2', depth, double, _ascii)
        else:
            meshcreator.to_mesh(model, fname, depth, double, _ascii)


class Region(object):
    """
    Stores the name, QPolygonF, and visibility associated with each region. Also contains
    various useful methods each region can use.
    """

    def __init__(self, name, region):
        """
        Inputs:
            name - a string giving the name of the Region.
            region - a QPolygonF giving the shape of the region.
        Variables:
            self.name - same as input name.
            self.region - same as input region.
            self.visibility - a boolean giving the visibility (shown/hidden) of the region.
        """
        super(Region, self).__init__()
        self.name = name
        self.region = region
        self.visible = False

    @classmethod
    def fromfile(cls, filename, _file=None):
        """
        Input: String filename, File _file
        Output: Region
        Purpose: If a Region has been saved using the save() method, this class method can read
                    and return a new Region object from the file.
        """
        region = QPolygonF()
        if _file is not None:
            scale = _file.scale()
        else:
            scale = 1.0
        name = ''
        with open(filename) as f:
            name = f.readline().split()[0]
            for line in f:
                coords = line.split(" ")
                region << QPointF(float(coords[0]) * scale, float(coords[1]) * scale)
        return cls(name, region)

    def contains(self, x, y, scaled=1):
        """
        Input: float x, float y, float scaled
        Output: boolean
        Purpose: Returns whether a certain point is inside the region. This capability has since been
                    replaced by imageprep.region_mask(), and I believe it is no longer used.
        """
        x *= scaled
        y *= scaled
        p = QPointF(x, y)
        return QGraphicsPolygonItem(self.region).contains(p)

    def points(self, scale=1):
        """
        Input: float scale
        Output: a list of QPointFs
        Purpose: Returns a list of all points that are part of the polygon.
        """
        i = 0
        p = None
        points = []
        if self.region is not None:
            while p != self.region.last():
                p = self.region.at(i)
                points.append(p)
                i += 1
        return points

    def get_bounding_box(self):
        """
        Returns the min/max x and y coordinates of all points in polygon. I believe this method is
        no longer used.
        """
        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')
        for p in self.points():
            if p.x() < xmin:
                xmin = p.x()
            elif p.x() > xmax:
                xmax = p.x()
            if p.y() < ymin:
                ymin = p.y()
            elif p.y() > ymax:
                ymax = p.y()
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def save(self, filename, _file=None):
        """
        Saves the Region to a text file. Useful for debugging purposes.
        """
        if _file is not None:
            scale = 1.0 / _file.scale()
        else:
            scale = 1.0

        with open(filename, 'w') as f:
            f.write(unicode('{0}\n'.format(self.name)))
            for p in self.points():
                f.write(unicode('{0} {1}\n'.format(int(p.x() * scale),
                                                   int(p.y() * scale))))

    def scaledRegion(self, _file):
        """
        Input: File _file
        Output: Region
        Purpose: Returns a new region object scaled to match the data of the input _file object.
        """
        region = QPolygonF()
        for point in self.points():
            scale = 1 / _file.scale()
            p = QPointF(int(point.x() * scale), int(point.y() * scale))
            region << p
        return Region(self.name, region)


class _MergedRegion(Region):
    """
    Originally supposed to represent a merging of two or more regions. The class does not currently
    work, and only about half the relevant methods throughout the GUI can support MergedRegions.
    """
    def __init__(self, list_of_regions):
        super(MergedRegion, self).__init__() # Will raise error, no arguments
        self.name = list_of_regions[0].name
        self.region = [region.region for region in list_of_regions]
        self.originals = list_of_regions
        self.visible = False

    def contains(self, x, y, scaled=False):
        return any(map(lambda reg: reg.contains(x, y, scaled), self.region))

    def split(self):
        return self.originals
