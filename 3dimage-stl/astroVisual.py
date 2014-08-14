import sys, os

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy as np
from scipy import ndimage
from astropy.io import fits
import qimage2ndarray as q2a
import photutils.utils as putils
from PIL import Image

from astroWidgets import *
from astroObjects import *
from star_scenes import *
from wizard import ThreeDModelWizard
from img2stl import imageprep

class AstroProject(QMainWindow):
	"""
	This program allows the user to upload any .fits image file and subsequently convert the image 
	into an stl file, which can then be printed with a 3D printer.

	TODO: Regions - 
		TODO: Region Panel
			-change region color
	TODO: Stars -
		TODO: Automatically find stars
		TODO: Manually touch up star autofind
		TODO: Allow loading of star locations
	*TODO: File I/O - 
		- Load Stars
		- Save Stars (ASCII Table)
		*-Save all as zip file
	"""

	def __init__(self, argv=None):
		super(AstroProject, self).__init__()
		self.filename = None # Unnecessary?
		self.curr = None # Do we need to handle multiple pics?
		self.transformation = lambda img: putils.scale_linear(img, percent=99)
		self.files = [] # Do we need to handle multiple pics?
		self.regions = []
		self.setGeometry(300, 150, 800, 800)
		self.createWidgets()
		self.resize(840, 860)
		self.setWindowTitle('3D Star Field Creator')
		if argv and (argv[0] in ('debug', '-debug', '--debug')):
			print 'running script'
			self.run_auto_login_script(argv[1:]) # FOR DEBUG PURPOSES ONLY
			wizard = ThreeDModelWizard(self, True)
			self.show()
		else:
			wizard = ThreeDModelWizard(self)
			self.show()
			

	# GUI Creation
	def addActions(self, target, actions):
		"""Adds an arbitary number of actions to a menu. A value of None will create a separator"""
		for action in actions:
			if action == None:
				target.addSeparator()
			elif isinstance(action, QMenu):
				target.addMenu(action)
			else:
				target.addAction(action)
	def createWidgets(self):
		self.widget = MainPanel(self)
		self.setCentralWidget(self.widget)
		self.regionDlg = None

		fileLoadAction = self.createAction("&Load", self.fileLoad, None, "Load another image file")
		fileSaveAction = self.createAction("&Save", self.fileSave, None, "Save all data in new directory")
		fileQuitAction = self.createAction("&Exit", QCoreApplication.instance().quit, None, "Exit the session")
		fileMenu = self.menuBar().addMenu("&File")
		self.addActions(fileMenu, (fileLoadAction, fileSaveAction, None, fileQuitAction))

		imageLinearTransform = self.createAction("Linear", self.setTransformation, "Linear")
		imageLogTransform = self.createAction("Log", self.setTransformation, "Logarithmic")
		imageSqrtTransform = self.createAction("Sqrt", self.setTransformation, "Sqrt")
		self.imageMenu = self.menuBar().addMenu("&Image")
		self.addActions(self.imageMenu, (imageLinearTransform, imageLogTransform, imageSqrtTransform))
		
		regionDrawAction = self.createAction("Draw Region", self.drawRegion, None, "Indicate a region on the image")
		regionSaveRegionAction = self.createAction("Save Region", self.saveRegion, None, "Save the currently displayed region")
		regionClearAction = self.createAction("Clear Region", self.clearRegion, None, "Clear the current region")
		self.regionMenu = self.menuBar().addMenu("Region")
		self.addActions(self.regionMenu, (regionDrawAction, regionSaveRegionAction, regionClearAction))

		for action in self.imageMenu.actions():
			action.setEnabled(False)
		for action in self.regionMenu.actions():
			action.setEnabled(False)
	def createAction(self, text, slot=None, data=None, tip=None, checkable=False):
		"""Creates an action that can be assigned to a button or menu"""
		action = QAction(text, self)
		if tip != None:
			action.setToolTip(tip)
			action.setStatusTip(tip)
		if slot != None:
			action.triggered.connect(slot)
		if data != None:
			action.setData(data)
		if checkable:
			action.setCheckable(True)
		return action

	# Basic Actions
	def fileLoad(self):
		"""Loads and displays a .fits image file. Saves it as a File object"""
		path = QFileInfo(self.filename).path() \
			if self.filename != None else "."
		fname = QFileDialog.getOpenFileName(self, "3D Model Creator - Load Image", \
			path, "FITS Files (*.fits)")
		if fname.isEmpty():
			return
		data = fits.getdata(str(fname))
		if data == None:
			QMessageBox.warning(self, "File Error", "This file does not contain image data")
			return
		name = os.path.basename(str(fname))
		image = QPixmap()
		image = image.fromImage(makeqimage(data, self.transformation, self.widget.size))
		pic = self.widget.addImage(image)
		self.files.append(File(name, data, pic))
		self.curr = self.files[len(self.files) - 1]
		action = self.createAction(name, self.changeImage, len(self.files) - 1, None)
		self.imageMenu.addAction(action)
		self.regionMenu.actions()[0].setEnabled(True)
		for action in self.imageMenu.actions():
			action.setEnabled(True)
	def fileSave(self):
		# get path
		path = QFileInfo(self.filename).path() \
			if self.filename != None else "."
		path = QFileDialog.getSaveFileName(self, "3D Model Creator - Save Files", \
			path)
		#path = '/Users/rrao/Documents/Internship/PyQt/testdir'
		path = str(path)
		os.mkdir(path)
		"""for _file in self.files:
			fname = path + "/3d_" + _file.name
			_file.save(fname)"""
		for reg in self.regions:
			fname = path + "/" + reg.name + ".reg"
			reg.save(fname, self.curr)
		self.curr.clusters.write(path + "/clusterpath", format='ascii')

	# Region Editing
	def clearRegion(self):
		"""Clears a region and deletes it if it has already been saved"""
		"""TODO: delete warning"""
		self.widget.clear_region()
	def deleteRegion(self, region):
		if not isinstance(region, Region):
			map(self.deleteRegion, region)
		else:
			self.hideRegion(region)
			self.regions.remove(region)
	def showRegion(self, region):
		if not isinstance(region, Region):
			region = filter(lambda reg: reg.visible == False, region)
			map(self.showRegion, region)
		elif not region.visible:
			self.widget.display_region(region)
			region.visible = True
	def hideRegion(self, region):
		if not isinstance(region, Region):
			region = filter(lambda reg: reg.visible, region)
			map(self.hideRegion, region)
		elif region.visible:
			self.widget.hide_region(region)
			region.visible = False
	def drawRegion(self, name):
		"""Switches to the interactive region drawing scene and obtains the name of the region to draw"""
		"""TODO: Handle duplicate regions"""
		self.widget.region_drawer(name)
		self.regionMenu.actions()[1].setEnabled(True)
		self.regionMenu.actions()[2].setEnabled(True)
	def get_region(self, name):
		for reg in self.regions:
			if reg.name == name:
				return reg
	def mergeRegions(self, name, list_of_regions): # No support for merged regions
		self.hideRegion(list_of_regions)
		map(self.regions.remove, list_of_regions)
		region = MergedRegion(list_of_regions)
		region.name = name
		self.regions.append(region)
		self.showRegion(region)
	def splitRegion(self, region): # No support for merged regions
		self.hideRegion(region)
		self.regions.remove(region)
		originals = region.originals
		self.regions += originals
		self.showRegion(originals)
	def saveRegion(self):
		"""Saves a region as a Region object (not to file). Exits the region drawing view."""
		name, region = self.widget.save_region()
		reg = Region(name, region)
		self.regions.append(reg)
		if len(self.regions) < 3:
			self.curr.spiralarms.append(reg)
			print 'Spiral'
		elif len(self.regions) == 3:
			self.curr.disk = reg
			print 'Disk'
		else:
			self.curr.stars.append(reg)
			print 'Star'
		self.regionMenu.actions()[1].setEnabled(False)
		self.regionMenu.actions()[2].setEnabled(False)
		self.showRegion(reg)
	
	# Image Transformations
	def changeImage(self):
		"""Allows the user to select which image to view"""
		action = self.sender()
		index = action.data().toInt()[0]
		self.curr = self.files[index]
		self.widget.setImage()
	def setTransformation(self, trans=None):
		"""Sets the current image transformation and updates all images with it."""
		if trans is None:
			action = self.sender()
			trans = action.data().toString()
		if trans == "Linear":
			self.transformation = lambda img: putils.scale_linear(img, percent=99)
		elif trans == "Logarithmic":
			self.transformation = lambda img: putils.scale_log(img, percent=99)
		elif trans == "Sqrt":
			self.transformation = lambda img: putils.scale_sqrt(img, percent=99)
		self.update_all()
	def update_all(self): # Necessary to support multiple images
		"""
		A simple helper method. Applies the remake_image method to all images when a 
		transformation has been applied to all of them.
		"""
		for f in self.files:
			self.remake_image(f)
		self.widget.setImage()
	def remake_image(self, f):
		"""A helper method. Recreates the stored pixmap when a transformation has been applied to the data."""
		pic = QPixmap()
		pic = pic.fromImage(makeqimage(f.data, self.transformation, self.widget.size))
		f.image = pic
	
	def resizeImage(self, _file, width, height):
		image = Image.fromarray(_file.data)
		image = image.resize((width, height))
		_file.data = np.array(image, dtype=np.float64)
		self.remake_image(_file)
		self.widget.setImage()
	def find_clusters(self):
		image = self.curr.data
		scale = self.curr.scale()
		peaks = imageprep.find_peaks(np.flipud(image))
		self.curr.clusters = peaks
		xcen = peaks['xcen'] * scale
		ycen = peaks['ycen'] * scale
		coords = np.dstack((xcen, ycen)).reshape((-1, 2))
		self.widget.cluster_find(coords)
	def save_clusters(self):
		toremove = self.widget.save_clusters()
		self.curr.clusters.remove_rows([i for i in toremove])
		self.curr.clusters = self.curr.clusters[:15]

	# For DEBUG PURPOSES ONLY
	def run_auto_login_script(self, argv):
		fname = "/Users/rrao/Documents/Internship/fits_files/ngc3344_uvis_f555w_sci.fits"
		data = fits.getdata(fname)
		name = os.path.basename(str(fname))
		image = QPixmap()
		image = image.fromImage(makeqimage(data, self.transformation, self.widget.size))
		pic = self.widget.addImage(image)
		self.files.append(File(name, data, pic))
		self.curr = self.files[len(self.files) - 1]
		action = self.createAction(name, self.changeImage, len(self.files) - 1, None)
		self.imageMenu.addAction(action)
		self.regionMenu.actions()[0].setEnabled(True)
		for action in self.imageMenu.actions():
			action.setEnabled(True)
		self.resizeImage(self.curr, 2000, 2000)
		self.setTransformation('Logarithmic')
		path = "/Users/rrao/Documents/Internship/test/testsave2/"
		disk = Region.fromfile(path + "Disk.reg")
		spiral1 = Region.fromfile(path + "Spiral1.reg")
		spiral2 = Region.fromfile(path + "Spiral2.reg")
		spiral3 = Region.fromfile(path + "Spiral3.reg")
		star1 = Region.fromfile(path + "Star1.reg")
		star2 = Region.fromfile(path + "Star2.reg")
		regions = [spiral1, spiral2, spiral3, disk, star1, star2]
		regions = [reg.scaledRegion(1/float(self.curr.scale())) for reg in regions]
		for reg in regions:
			self.regions.append(reg)
			if len(self.regions) < 4:
				self.curr.spiralarms.append(reg)
			elif len(self.regions) == 4:
				self.curr.disk = reg
			else:
				self.curr.stars.append(reg)
		from astropy.table import Table
		t = Table.read(path + 'clusterpath', format='ascii')
		self.curr.clusters = t

class MainPanel(QWidget):
	"""
	The central widget for AstroProject. It contains all other (visual) widgets and will display
	images, lists, etc. interchangeably. It contains a single main scene that can display the current 
	picture, and little else. Other more complex visual scenes get the current picture, etc. from the main
	scene.
	"""
	def __init__(self, parent):
		super(MainPanel, self).__init__(parent)

		self.parent = parent
		self.view = QGraphicsView(self)
		self.view.setAlignment(Qt.AlignCenter)

		layout = QGridLayout(self)
		layout.addWidget(self.view, 0, 0)
		self.setLayout(layout)
		self.size = QSize(parent.width(), parent.height())
		self.view.setMinimumSize(self.size)
		self.resize(self.size)

		self.main_scene = StarScene(self, self.view.width(), self.view.height())
		self.current_scene = self.main_scene
		self.view.setScene(self.current_scene)

		self.show()

	def addImage(self, pixmap):
		"""Takes in a QPixmap and adds the picture to the main scene as a QPixmapGraphicsItem"""
		return self.main_scene.addImg(pixmap)

	def setImage(self):
		"""Sets the current image to a previously loaded image"""
		self.main_scene.addImg(self.parent.curr.image)
		self.update_scene(self.main_scene)

	def region_drawer(self, name, points=None, region=None):
		"""Sets the scene to an interactive drawing scene, allowing the user to draw regions"""
		draw_scene = RegionStarScene(self, self.parent.curr.image, name, points, region)
		self.update_scene(draw_scene)

	def cluster_find(self, points):
		cluster_scene = ClusterStarScene(self, self.parent.curr.image, points)
		self.update_scene(cluster_scene)

	def save_region(self):
		"""Sets the scene to the non-interactive main scene. Passes region information to parent to save"""
		name, region = self.current_scene.getRegion()
		self.update_scene(self.main_scene)
		return name, region

	def save_clusters(self):
		toremove = self.current_scene.toremove
		self.update_scene(self.main_scene)
		return toremove

	def clear_region(self):
		"""Clears the currently displayed region"""
		self.current_scene.clear()

	def display_region(self, region):
		"""Shows a region on top of the non-interactive main scene"""
		self.main_scene.addReg(region)
		self.update_scene(self.main_scene)

	def hide_region(self, region):
		"""Hides a region from the non-interactive main scene"""
		self.main_scene.delReg(region)
		self.update_scene(self.main_scene)

	def update_scene(self, scene):
		"""A simple helper method. Sets the current scene to the input and changes the view."""
		self.current_scene = scene
		self.view.setScene(self.current_scene)

def makeqimage(nparray, transformation=None, size=None):
	"""
	Takes in an nparray and returns a qimage.
	Performs various transformations (linear, log, sqrt, etc.) on the image.
	Clips and scales pixel values so only a certain range is shown.
	Scales and inverts the image.
	All transformations are nondestructive (performed on a copy of the input array).
	"""
	npimage = nparray.copy()
	npimage[npimage < 0] = 0
	npimage = transformation(npimage)
	npimage = q2a._normalize255(npimage, True)
	qimage = q2a.gray2qimage(npimage, (0, 255), size)
	return qimage

def main(argv):
	app = QApplication(argv)
	window = AstroProject(argv[1:])
	sys.exit(app.exec_())

if __name__ == '__main__':
	main(sys.argv)

		

