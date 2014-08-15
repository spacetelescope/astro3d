from PyQt4.QtGui import *
from PyQt4.QtCore import *

class StarScene(QGraphicsScene):

	def __init__(self, parent, width, height):
		super(StarScene, self).__init__(parent)
		self.item = None
		self.size = QSize(width, height)
		self.regions = {}

	def addImg(self, pixmap):
		self.clear()
		pixmap = pixmap.scaled(self.size, Qt.KeepAspectRatio)
		self.item = QGraphicsPixmapItem(pixmap)
		self.addItem(self.item)
		for name in self.regions:
			self.addItem(self.regions[name])
		return pixmap

	def addReg(self, region):
		self.regions[region.name] = self.reg_add(region.region)

	def reg_add(self, sub_region):
		if type(sub_region) == list or type(sub_region) == tuple:
			return map(self.reg_add, sub_region)
		else:
			r = QGraphicsPolygonItem(sub_region)
			r.setPen(QColor(0, 100, 200))
			self.addItem(r)
			return r

	def delReg(self, region):
		self.reg_remove(self.regions[region.name)
		del self.regions[region.name]

	def reg_remove(self, sub_region):
		if type(sub_region) == list or type(sub_region) == tuple:
			map(self.reg_remove, sub_region)
		else:
			self.removeItem(sub_region)

	def clear(self):
		for i in self.items():
			self.removeItem(i)
		self.item = None

class RegionStarScene(QGraphicsScene):

	def __init__(self, parent, pixmap, name):
		super(RegionStarScene, self).__init__(parent)
		self.name = name
		self.item = QGraphicsPixmapItem(pixmap)
		self.addItem(self.item)
		self.points = []
		self.shape = None
		self.display_shape = None

	def mousePressEvent(self, event):
		p = event.scenePos()
		self.points.append(p)
		e = QGraphicsEllipseItem(p.x(), p.y(), 10, 10)
		e.setPen(QPen(QColor(0, 255, 0)))
		self.addItem(e)
		if len(self.points) >= 3:
			self.shape = QPolygonF(self.points)
			if self.display_shape is not None:
				self.removeItem(self.display_shape)
			self.display_shape = QGraphicsPolygonItem(self.shape)
			self.display_shape.setPen(QPen(QColor(0, 100, 200)))
			self.addItem(self.display_shape)

	def getRegion(self):
		return self.name, self.shape

	def clear(self):
		for i in self.items():
			self.removeItem(i)
		self.addItem(self.item)
		self.points = []
		self.shape = None
		self.display_shape = None

class ClusterStarScene(QGraphicsScene):

	def __init__(self, parent, pixmap, points):
		# Point coordinates incorrect
		super(ClusterStarScene, self).__init__(parent)
		self.points = points
		self.image = QGraphicsPixmapItem(pixmap)
		self.addItem(self.image)
		self.graphicspoints = []
		for point in points[:15]:
			self.graphicspoints.append(QGraphicsEllipseItem(point[0] - 5, point[1] - 5, 10, 10))
			self.graphicspoints[-1].setPen(QPen(QColor(200, 50, 50)))
			self.addItem(self.graphicspoints[-1])
		self.added = 15
		self.toremove = []

	def mousePressEvent(self, event):
		p = event.scenePos()
		for i, gp in enumerate(self.graphicspoints):
			if gp.contains(p):
				self.removeItem(gp)
				self.graphicspoints.remove(gp)
				self.toremove.append(i)
				self.graphicspoints.append(QGraphicsEllipseItem(self.points[self.added][0] - 5, 
															self.points[self.added][1] - 5, 10, 10))
				# TODO: cannot remove all points
				self.graphicspoints[-1].setPen(QPen(QColor(200, 50, 50)))
				self.addItem(self.graphicspoints[-1])
				self.added += 1
				break

	def get_points(self):
		toreturn = []
		for i, point in enumerate(self.points):
			if len(toreturn) == 15:
				return toreturn
			elif i not in self.toremove:
				toreturn.append(point)