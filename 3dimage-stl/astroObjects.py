import sys

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from astropy.io import fits
import numpy as np

from img2stl import imageprep, meshcreator

class File(object):

	def __init__(self, name, data, image):
		super(File, self).__init__()
		self.name = name
		self.data = data
		self.image = image
		self.height = 150.0
		self.spiralarms = []
		self.disk = None
		self.stars = []
		self.clusters = None

	def save(self, fname):
		prihdr = fits.Header()
		prihdr['C_SCALE'] = self.data.shape[0] / float(self['Original'].shape[0])
		prihdr['DIMS'] = self.data.shape
		prihdr['DIMS_O'] = self['Original'].shape
		hdu = fits.PrimaryHDU(self.data, header=prihdr)
		hdu.writeto(fname)

	def scale(self):
		return self.image.height() / float(self.data.shape[0])

	def scaleRegions(self):
		self.spiralarms = [reg.scaledRegion(self) for reg in self.spiralarms]
		self.disk = self.disk.scaledRegion(self)
		self.stars = [reg.scaledRegion(self) for reg in self.stars]

	def make_3d(self, fname, depth=1, double=False, _ascii=False):
		self.scaleRegions()
		self.data = np.flipud(self.data)

		spiralmask = imageprep.combine_masks([imageprep.region_mask(self.data, reg, True) for reg
			in self.spiralarms])
		disk = imageprep.region_mask(self.data, self.disk, True)

		model = imageprep.make_model(self.data, spiralmask, disk, self.clusters,
							self.height, self.stars)

                # Split bottom/top here

		print 'Creating mesh'
		meshcreator.to_mesh(model, fname, depth, double, _ascii)
		print 'Done!'


class Region(dict):
	"""
	Stores values associated with each region constructed
	TODO: Add pen color
	TODO: Set pen color
	"""

	def __init__(self, name, region):
		super(Region, self).__init__()
		self.name = name
		self.region = region
		self.visible = False

	@classmethod
	def fromfile(cls, filename, _file=None):
		region = QPolygonF()
		scale = (_file.image.height() / float(_file.data.shape[0])) if _file != None else 1
		name = ''
		with open(filename) as f:
			name = f.readline()
			for line in f:
				coords = line.split(" ")
				region << QPointF(float(coords[0]) * scale, float(coords[1]) * scale)
		return cls(name, region)

	def contains(self, x, y, scaled=False):
		if scaled:
			x *= scaled
			y *= scaled
		p = QPointF(x, y)
		return QGraphicsPolygonItem(self.region).contains(p)

	def points(self, scale=1):
		i = 0
		p = None
		points = []
		while p != self.region.last():
			p = self.region.at(i)
			points.append(p)
			i += 1
		return points

	def get_bounding_box(self):
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

	def setVisible(self, flag):
		self.visible = flag

	def isVisible(self):
		return self.visible

	def save(self, filename, _file=None):
		with open(filename, 'w') as f:
			f.write(unicode(self.name + "\n"))
			for p in self.points():
				string = ''
				scale = (1 / _file.scale()) if _file is not None else 1
				string = str(int((p.x()) * scale)) + " " + str(int(p.y() * scale)) + "\n"
				f.write(unicode(string))

	def scaledRegion(self, _file):
		region = QPolygonF()
		for point in self.points():
			scale = 1 / _file.scale()
			p = QPointF(int(point.x() * scale), int(point.y() * scale))
			region << p
		return Region(self.name, region)

class MergedRegion(Region):

	def __init__(self, list_of_regions):
		super(MergedRegion, self).__init__()
		self.name = list_of_regions[0].name
		self.region = [region.region for region in list_of_regions]
		self.originals = list_of_regions
		self.visible = False

	def contains(self, x, y, scaled=False):
		return any(map(lambda reg: reg.contains(x, y, scaled), self.region))

	def split(self):
		return self.originals
