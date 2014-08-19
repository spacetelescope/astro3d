import sys, os

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy as np
from scipy import ndimage
from astropy.io import fits
import qimage2ndarray as q2a
import photutils.utils as putils
from PIL import Image

from astroObjects import File, Region, MergedRegion
from star_scenes import StarScene, RegionStarScene, ClusterStarScene
from wizard import ThreeDModelWizard
from img2stl import imageprep

class AstroGUI(QMainWindow):
	"""
	This program allows the user to upload any .fits image file and subsequently convert the image 
	into an stl file, which can then be printed with a 3D printer.

	This class is the Main Window. It can have a menu bar and it displays the image. All other code
	is initialized from this class. The methods contained in this class primarily exist to interface
	between different objects, specifically by storing changes made by the wizard and the MainPanel 
	(below) in File and Region objects. Furthermore it applies changes made from the wizard to the
	Image, thus changing the image display.
	"""

	def __init__(self, argv=None):
		"""
		Inputs:
			argv - passed in when the program is started from the command line. Used to run the debug 
					script. See run_auto_login_script method for details.
		Variables:
			self.transformation - provides the transformation (linear, log, or sqrt) applied to the 
									image before it is displayed. Applied for visualization and display
									purposes only.
			self.regions - a list of all regions the user has created
			self.files & self.curr - initially it seemed necessary to allow the user to upload multiple
										images and use them in the creation of a model. To this end,
										self.files is the list of images uploaded, while self.curr is
										a pointer to the currently selected (displayed) image. However,
										the engine does not require multiple images to work, so this 
										functionality may be unnecessary.
			self.widget - initialized in self.createWidgets, this is the MainPanel, which displays the
							image.

		"""
		super(AstroGUI, self).__init__()
		self.filename = None # Unnecessary?
		self.curr = None # Do we need to handle multiple pics?
		self.transformation = lambda img: putils.scale_linear(img, percent=99)
		self.files = [] # Do we need to handle multiple pics?
		self.regions = []
		self.setGeometry(300, 150, 800, 800)
		self.createWidgets()
		self.resize(840, 860)
		self.setWindowTitle('3D Star Field Creator')
		if argv and argv[0] == 'debug':
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
		"""
		Creates the File, Image, and Region menus. With the creation of the wizard, these menus serve no purpose.
		"""
		self.widget = MainPanel(self)
		self.setCentralWidget(self.widget)

		fileLoadAction = self.createAction("&Load", self.fileLoad, "Load another image file")
		fileSaveAction = self.createAction("&Save", self.fileSave, "Save all data in new directory")
		fileQuitAction = self.createAction("&Exit", QCoreApplication.instance().quit, "Exit the session")
		fileMenu = self.menuBar().addMenu("&File")
		self.addActions(fileMenu, (fileLoadAction, fileSaveAction, None, fileQuitAction))

		imageLinearTransform = self.createAction("Linear", lambda: self.setTransformation("Linear"))
		imageLogTransform = self.createAction("Log", lambda: self.setTransformation("Logarithmic"))
		imageSqrtTransform = self.createAction("Sqrt", lambda: self.setTransformation("Sqrt"))
		self.imageMenu = self.menuBar().addMenu("&Image")
		self.addActions(self.imageMenu, (imageLinearTransform, imageLogTransform, imageSqrtTransform))
		
		regionDrawAction = self.createAction("Draw Region", self.drawRegion, "Indicate a region on the image")
		regionSaveRegionAction = self.createAction("Save Region", self.saveRegion, "Save the currently displayed region")
		regionClearAction = self.createAction("Clear Region", self.clearRegion, "Clear the current region")
		self.regionMenu = self.menuBar().addMenu("Region")
		self.addActions(self.regionMenu, (regionDrawAction, regionSaveRegionAction, regionClearAction))

		for action in self.imageMenu.actions():
			action.setEnabled(False)
		for action in self.regionMenu.actions():
			action.setEnabled(False)
	def createAction(self, text, slot=None, tip=None, checkable=False):
		"""Creates an action with given text, slot (method), tooltip, and checkable."""
		action = QAction(text, self)
		if tip != None:
			action.setToolTip(tip)
			action.setStatusTip(tip)
		if slot != None:
			action.triggered.connect(slot)
		if checkable:
			action.setCheckable(True)
		return action

	# Basic Actions
	def fileLoad(self):
		"""
		Launches the QFileDialog to obtain a filename to load. Uses fits.getdata to obtain the numpy
		array. Uses makeqimage to create a QImage from the numpy array. With QPixmap.fromImage, it
		then creates a QPixmap, which is passed to the MainPanel (self.widget). A File object is
		also created to store information. An action that allows the user to change the current image
		(in the case where multiple images are uploaded) is created, while several other actions are
		enabled.
		"""
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
		action = self.createAction(name, lambda: self.changeImage(len(self.files) - 1), None)
		self.imageMenu.addAction(action)
		self.regionMenu.actions()[0].setEnabled(True)
		for action in self.imageMenu.actions():
			action.setEnabled(True)
	def fileSave(self):
		"""Used for debug purposes"""
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
		"""Interfaces between wizard and MainPanel. Clears a region that is currently being drawn."""
		self.widget.clear_region()
	def deleteRegion(self, region):
		"""
		Input: Region region
		Purpose: Deletes a previously drawn region, removing it from self.regions and from the screen.
		"""
		if not isinstance(region, Region):
			map(self.deleteRegion, region)
		else:
			self.hideRegion(region)
			self.regions.remove(region)
	def showRegion(self, region):
		"""
		Input: Region region
		Purpose: Displays the hidden region(s) passed in as the region parameter.
		"""
		if not isinstance(region, Region):
			region = filter(lambda reg: reg.visible == False, region)
			map(self.showRegion, region)
		elif not region.visible:
			self.widget.display_region(region)
			region.visible = True
	def hideRegion(self, region):
		"""
		Input: Region region
		Output: Hides the displayed region(s) passed in as the region parameter.
		"""
		if not isinstance(region, Region):
			region = filter(lambda reg: reg.visible, region)
			map(self.hideRegion, region)
		elif region.visible:
			self.widget.hide_region(region)
			region.visible = False
	def drawRegion(self, name):
		"""
		Input: String name
		Purpose: Tells the MainPanel to switch to the interactive region drawing QGraphicsScene. Takes 
					the parameter name, which denotes the name of the region to be drawn. Also enables 
					some actionsin the regionMenu, which can be deleted. Regions with the same name, 
					however, will cause errors, but this would probably be better handled in the wizard, 
					which is where the name is obtained.
		"""
		self.widget.region_drawer(name)
		self.regionMenu.actions()[1].setEnabled(True)
		self.regionMenu.actions()[2].setEnabled(True)
	def get_region(self, name):
		"""
		Input: String name
		Output: Region reg
		Purpose: Since I used a list to store the regions, it was necessary to create this method. A 
					better solution would be to make self.regions a dictionary.
		"""
		for reg in self.regions:
			if reg.name == name:
				return reg
	def mergeRegions(self, name, list_of_regions): # No support for merged regions
		"""
		Merges several regions together into a MergedRegion object. As of now, this ability does not 
		work. I am not sure that the capability to merge regions is necessary, but have left the 
		relevant methods in place. See astroObjects.MergedRegion for more information.
		"""
		self.hideRegion(list_of_regions)
		map(self.regions.remove, list_of_regions)
		region = MergedRegion(list_of_regions)
		region.name = name
		self.regions.append(region)
		self.showRegion(region)
	def splitRegion(self, region): # No support for merged regions
		"""Splits a MergedRegion into component regions. See above."""
		self.hideRegion(region)
		self.regions.remove(region)
		originals = region.originals
		self.regions += originals
		self.showRegion(originals)
	def saveRegion(self):
		"""
		Obtains the name (string) and region (QPolygonF) from the MainPanel, and in turn creates a 
		Region object, which is added to self.regions. The GUI does not have a way to assign different 
		regions as the spiralarms, disk, and stars to be eliminated, so I have been manually editing 
		this method to assign the regions. NGC3344 has 3 spiralarms while NGC1566 has 2, so some 
		changes need to be ensure the correct regions are assigned.
		"""
		name, region = self.widget.save_region()
		reg = Region(name, region)
		self.regions.append(reg)
		if len(self.regions) < 3: # 3 for NGC1566, 4 for NGC3344
			self.curr.spiralarms.append(reg)
			print 'Spiral'
		elif len(self.regions) == 3: # 3 for NGC1566, 4 for NGC3344
			self.curr.disk = reg
			print 'Disk'
		else:
			self.curr.stars.append(reg)
			print 'Star'
		self.regionMenu.actions()[1].setEnabled(False)
		self.regionMenu.actions()[2].setEnabled(False)
		self.showRegion(reg)
	
	# Image Transformations
	def changeImage(self, index):
		"""
		Input: int index
		Purpose: Every time an image is loaded, an action is created in the imageMenu connected to this 
					method, with index value increasing from 0 in the order images are loaded. This 
					allows the user to selecta different image using the imageMenu. If multiple image 
					capability were to be retained, it may make sense to maintain this capability 
					separate from the wizard, thereby enabling the user to switch between images at any 
					time.
		"""
		self.curr = self.files[index]
		self.widget.setImage()
	def setTransformation(self, trans=None):
		"""
		Input: String trans
		Purpose: Uses methods from photutils.utils to scale image intensity values, which allows better 
					viusalization of the images. As of now, users can select between linear, 
					logarithmic, and square root transforms. It is important to note that the scaling 
					is for display purposes only and has no effect on the engine.
		"""
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
	def remake_image(self, _file):
		"""
		Input: File _file
		Purpose: Any time a change is made to the image being displayed, remake_image can be called to 
					recreatethe relevant pixmap and change the image display.
		"""
		pic = QPixmap()
		pic = pic.fromImage(makeqimage(_file.data, self.transformation, self.widget.size))
		_file.image = pic
	
	def resizeImage(self, _file, width, height):
		"""
		Input: File _file, int width, int height
		Purpose: Uses PIL (or Pillow) to resize an array to the given dimensions. The width and height 
					are given by the user in the wizard's ImageResizePage. See 
					wizard.ThreeDModelWizard.ImageResizePage for more information.
		"""
		image = Image.fromarray(_file.data)
		image = image.resize((width, height))
		_file.data = np.array(image, dtype=np.float64)
		self.remake_image(_file)
		self.widget.setImage()
	def find_clusters(self):
		"""
		Retrieves locations of star clusters using imageprep.find_peaks, then displays them on the 
		screen for the user to see using the ClusterStarScene. This action is similar to 
		matplotlib.pyplot.scatter.
		"""
		image = self.curr.data
		scale = self.curr.scale()
		peaks = imageprep.find_peaks(np.flipud(image))
		self.curr.clusters = peaks
		xcen = peaks['xcen'] * scale
		ycen = peaks['ycen'] * scale
		coords = np.dstack((xcen, ycen)).reshape((-1, 2))
		self.widget.cluster_find(coords)
	def save_clusters(self):
		"""
		Removes clusters identified by the user as other objects (foreground stars, center of galaxy
		etc.), then saves the 15 brightest star clusters to the File object.
		"""
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
		action = self.createAction(name, lambda: self.changeImage(len(self.files) - 1), None)
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
	The central widget for AstroGUI. Contains a QGraphicsView which can show several QGraphicsScenes.
	The primary purpose of this widget is to interface between the AstroGUI and the QGraphicsScenes in 
	order to enable viewing images, drawing regions, and finding star clusters. In addition to its 
	QGraphicsView, it also contains a non-interactive main_scene, along with a pointer to the current 
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
		"""
		Input: QPixmap pixmap
		Output: QPixmap scaledPixmap
		Purpose: Adds the given pixmap to the display. Returns a scaled version for storage in the 
					appropriate File object.
		"""
		return self.main_scene.addImg(pixmap)

	def setImage(self):
		"""Sets the current image to AstroGUI's currently selected image."""
		self.main_scene.addImg(self.parent.curr.image)
		self.update_scene(self.main_scene)

	def region_drawer(self, name):
		"""
		Input: String name
		Purpose: Sets the scene to an interactive drawing scene, allowing the user to draw regions.
		"""
		draw_scene = RegionStarScene(self, self.parent.curr.image, name)
		self.update_scene(draw_scene)

	def cluster_find(self, points):
		"""
		Input: np.ndarray points
		Purpose: Highlights the locations of clusters given by points to allow the user to remove 
					'invalid' clusters, such as foreground stars, etc. using the interactive 
					ClusterStarScene.
		"""
		cluster_scene = ClusterStarScene(self, self.parent.curr.image, points)
		self.update_scene(cluster_scene)

	def save_region(self):
		"""
		Output: String name, QPolygonF region.
		Purpose: Sets the scene to the non-interactive main scene. Passes region information to 
					AstroGUI to save.
		"""
		name, region = self.current_scene.getRegion()
		self.update_scene(self.main_scene)
		return name, region

	def save_clusters(self):
		"""
		Output: list toremove
		Purpose: Sets the scene to the non-interactive main scene. Passes a list of indices to remove 
					from the astropy Table that contains the cluster information.
		"""
		toremove = self.current_scene.toremove
		self.update_scene(self.main_scene)
		return toremove

	def clear_region(self):
		"""Clears the currently displayed region."""
		self.current_scene.clear()

	def display_region(self, region):
		"""
		Input: Region region
		Purpose: Shows a region on top of the non-interactive main scene.
		"""
		self.main_scene.addReg(region)
		self.update_scene(self.main_scene)

	def hide_region(self, region):
		"""
		Input: Region region
		Purpose: Hides a region from the non-interactive main scene.
		"""
		self.main_scene.delReg(region)
		self.update_scene(self.main_scene)

	def update_scene(self, scene):
		"""
		Input: QGraphicsScene scene
		Purpose: A simple helper method. Sets the current scene to the input and changes the view.
		"""
		self.current_scene = scene
		self.view.setScene(self.current_scene)

def makeqimage(nparray, transformation=None, size=None):
	"""
	Input: np.ndarray nparray, function transformation, QSize size
	Output: QImage qimage
	Purpose: Performs various transformations (linear, log, sqrt, etc.) on the image.
				Clips and scales pixel values between 0 and 255.
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
	window = AstroGUI(argv[1:])
	sys.exit(app.exec_())

if __name__ == '__main__':
	main(sys.argv)

		

