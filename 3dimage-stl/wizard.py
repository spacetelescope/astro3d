from PyQt4.QtGui import *
from PyQt4.QtCore import *
from astroObjects import *

class ThreeDModelWizard(QWizard):

	def __init__(self, parent, debug=False):
		super(ThreeDModelWizard, self).__init__(parent)
		self.parent = parent
		if not debug:
			self.addPage(self.createIntroPage())
			self.addPage(self.createImageLoadPage())
			self.addPage(self.ImageResizePage(self.parent))
			self.addPage(self.IntensityScalePage(self.parent))
			self.addPage(self.RegionPage(self.parent))
			self.addPage(self.IdentifyPeakPage(self.parent))
		self.addPage(self.MakeModelPage(self.parent, debug))
		self.setWindowTitle("Create a 3D Model of a Galaxy")
		self.setVisible(True)

	def createIntroPage(self):
		page = QWizardPage()
		page.setTitle("Introduction")

		label = QLabel("This wizard will help you create a 3D model of "
				"a galaxy.")
		label.setWordWrap(True)
		vbox = QVBoxLayout()
		vbox.addWidget(label)
		page.setLayout(vbox)
		return page

	def createImageLoadPage(self):
		page = QWizardPage()
		page.setTitle("Load an Image")
		label = QLabel("First, click the button to load an image of a galaxy:")
		button = QPushButton("Load Image")
		button.clicked.connect(self.parent.fileLoad)
		vbox = QVBoxLayout()
		vbox.addWidget(label)
		vbox.addWidget(button)
		page.setLayout(vbox)
		return page

	class IntensityScalePage(QWizardPage):

		def __init__(self, grandparent):
			super(ThreeDModelWizard.IntensityScalePage, self).__init__()
			self.grandparent = grandparent

			self.setTitle("Scale Image Intensities")
			self.setSubTitle("Use a linear, logarithmic, or square root"
				" scale to better view the image")
			linbutton = QCheckBox("Linear Scale")
			logbutton = QCheckBox("Log Scale")
			sqrtbutton = QCheckBox("Sqrt Scale")

			self.bgroup = QButtonGroup()
			self.bgroup.setExclusive(True)
			self.bgroup.addButton(linbutton)
			self.bgroup.addButton(logbutton)
			self.bgroup.addButton(sqrtbutton)
			self.bgroup.setId(linbutton, 0)
			self.bgroup.setId(logbutton, 1)
			self.bgroup.setId(sqrtbutton, 2)
			linbutton.setChecked(True)

			applybutton = QPushButton("Apply")
			applybutton.clicked.connect(self.apply)

			button_grid = QGridLayout()
			button_grid.addWidget(linbutton, 0, 0)
			button_grid.addWidget(logbutton, 0, 1)
			button_grid.addWidget(sqrtbutton, 0, 2)
			button_grid.addWidget(applybutton, 1, 2)

			self.setLayout(button_grid)

		def apply(self):
			_id = self.bgroup.checkedId()
			if _id == 0:
				self.grandparent.setTransformation('Linear')
			elif _id == 1:
				self.grandparent.setTransformation('Logarithmic')
			elif _id == 2:
				self.grandparent.setTransformation('Sqrt')

	class ImageResizePage(QWizardPage):

		def __init__(self, grandparent):
			super(ThreeDModelWizard.ImageResizePage, self).__init__()
			self.grandparent = grandparent
			self.setTitle("Adjust Image Resolution")
			self.height = 0
			self.width = 0
			self.messageLabel = QLabel("")
			self.messageLabel.setWordWrap(True)
			dLabel = QLabel("Dimensions (pixels):")
			self.xtext = QLineEdit(str(self.width))
			self.ylabel = QLabel(" x " + str(self.height))
			self.xtext.textChanged.connect(self.setYText)
			button = QPushButton("Resize")
			button.clicked.connect(self.changeSize)
			grid = QGridLayout()
			grid.addWidget(dLabel, 0, 0)
			grid.addWidget(self.xtext, 0, 1)
			grid.addWidget(self.ylabel, 0, 2)
			grid.addWidget(button, 1, 1)
			vbox = QVBoxLayout()
			vbox.addWidget(self.messageLabel)
			vbox.addLayout(grid)
			self.setLayout(vbox)

		def resetUI(self, height, width):
			self.height = height
			self.width = width
			numpixels = self.height * self.width

			message = ""
			if numpixels < 900*900:
				message = "The current image resolution is too low. \
							Please increase the resolution until it is between 900x900 and \
							1300x1300 pixels."
			elif numpixels > 1300*1300:
				message = "The current image resolution is too high. \
							Please decrease the resolution until it is between 900x900 and \
							1300x1300 pixels." 
			else:
				message = "The current image resolution is perfect! \
							You can move on to the next page."
			self.messageLabel.setText(message)
			self.xtext.setText(str(width))

		def setYText(self):
			# Y size always equals x size...
			width = self.xtext.text()
			if not width.isEmpty():
				width = int(width)
				scale = int(width) / float(self.width)
				self.width = width
				self.height = self.height * scale
				self.ylabel.setText(" x " + str(int(self.height)))

		def changeSize(self):
			self.grandparent.resizeImage(self.grandparent.curr, int(self.width), int(self.height))

		def initializePage(self):
			super(ThreeDModelWizard.ImageResizePage, self).initializePage()
			h, w = self.grandparent.curr.data.shape
			self.resetUI(h, w)

	class RegionPage(QWizardPage):

		def __init__(self, grandparent):
			super(ThreeDModelWizard.RegionPage, self).__init__()
			self.grandparent = grandparent
			self.setTitle("Region Draw/Edit Page")
			self.initUI()

		def initUI(self):
			self.draw = QPushButton("Draw Region")
			self.save = QPushButton("Save Region")
			self.clear = QPushButton("Clear Region")

			self.draw.clicked.connect(self.drawRegion)
			self.save.clicked.connect(self.saveRegion)
			self.clear.clicked.connect(self.clearRegion)

			self.save.setEnabled(False)
			self.clear.setEnabled(False)

			buttongrid = QGridLayout()
			buttongrid.addWidget(self.draw, 0, 0)
			buttongrid.addWidget(self.save, 0, 1)
			buttongrid.addWidget(self.clear, 0, 2)

			hbox = self.createRegionList()

			vbox = QVBoxLayout()
			vbox.addLayout(buttongrid)
			vbox.addLayout(hbox)

			self.setLayout(vbox)
		
		def drawRegion(self):
			name, ok = QInputDialog.getText(self, 'Region Creator', 'Enter the name of this region:')
			if ok:
				self.grandparent.drawRegion(name)
				self.draw.setEnabled(False)
				self.save.setEnabled(True)
				self.clear.setEnabled(True)

		def saveRegion(self):
			self.grandparent.saveRegion()
			self.draw.setEnabled(True)
			self.save.setEnabled(False)
			self.clear.setEnabled(False)
			self.add_items()

		def clearRegion(self):
			self.grandparent.clearRegion()
			# Exit from drawscript?

		def createRegionList(self):
			self.reg_list = QListWidget()
			self.add_items()
			self.reg_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
			box = QDialogButtonBox(Qt.Vertical)
			self.show_ = box.addButton("Show", QDialogButtonBox.ActionRole)
			self.hide = box.addButton("Hide", QDialogButtonBox.ActionRole)
			self.merge = box.addButton("Merge", QDialogButtonBox.ActionRole)
			self.split = box.addButton("Split", QDialogButtonBox.ActionRole)
			self.delete = box.addButton("Delete", QDialogButtonBox.ActionRole)
			self.show_.clicked.connect(self.show_region)
			self.hide.clicked.connect(self.hide_region)
			self.merge.clicked.connect(self.merge_region)
			self.split.clicked.connect(self.split_region)
			self.delete.clicked.connect(self.delete_region)
			self.enableButtons()
			self.reg_list.itemSelectionChanged.connect(self.enableButtons)
			hbox = QHBoxLayout()
			hbox.addWidget(self.reg_list)
			hbox.addWidget(box)
			return hbox

		def add_items(self):
			self.reg_list.clear()
			self.reg_list.addItems([reg.name for reg in self.grandparent.regions])

		def enableButtons(self):
			selected = self.getSelected()
			if selected:
				self.delete.setEnabled(True)
				self.hide.setEnabled(True) if any([reg.visible for reg in selected]) \
												else self.hide.setEnabled(False)
				self.show_.setEnabled(True) if not all([reg.visible for reg in selected]) \
												else self.show_.setEnabled(False)
				self.merge.setEnabled(True) if len(selected) > 1 else self.merge.setEnabled(False)
				self.split.setEnabled(True) if len(selected) == 1 and \
												isinstance(selected[0], MergedRegion) \
												else self.split.setEnabled(False)
			else:
				self.show_.setEnabled(False)
				self.hide.setEnabled(False)
				self.merge.setEnabled(False)
				self.split.setEnabled(False)
				self.delete.setEnabled(False)

		def getSelected(self):
			return [self.grandparent.get_region(item.text()) for item in self.reg_list.selectedItems()]

		def show_region(self):
			self.grandparent.showRegion(self.getSelected())
			self.enableButtons()

		def hide_region(self):
			self.grandparent.hideRegion(self.getSelected())
			self.enableButtons()

		def merge_region(self):
			name, ok = QInputDialog.getText(self, "Merge Regions", "Enter the name of the merged region:")
			if ok:
				self.grandparent.mergeRegions(name, self.getSelected())
				self.add_items()
				self.enableButtons()

		def split_region(self):
			self.grandparent.splitRegion(self.getSelected()[0])
			self.add_items()
			self.enableButtons()

		def delete_region(self):
			self.grandparent.deleteRegion(self.getSelected())
			self.add_items()
			self.enableButtons()

	class IdentifyPeakPage(QWizardPage):

		def __init__(self, grandparent):
			super(ThreeDModelWizard.IdentifyPeakPage, self).__init__()
			self.grandparent = grandparent
			self.setTitle("Identify 15 Brightest Star Clusters")
			self.setSubTitle("""
				The 15 brightest objects will be highlighted on the image 
				for you. Some of these objects may not actually represent 
				star clusters, but may instead be single stars, or other 
				objects. Clicking on the identifying circles will remove 
				that point from the list. Once you have 15 clusters identified,
				click on the 'Save Points' button.
				""")
			savebutton = QPushButton("Save Points")
			savebutton.clicked.connect(self.grandparent.save_clusters)
			vbox = QVBoxLayout()
			vbox.addWidget(savebutton)
			self.setLayout(vbox)

		def initializePage(self):
			super(ThreeDModelWizard.IdentifyPeakPage, self).initializePage()
			self.grandparent.find_clusters()

	class MakeModelPage(QWizardPage):

		def __init__(self, grandparent, debug=False):
			super(ThreeDModelWizard.MakeModelPage, self).__init__()
			self.grandparent = grandparent
			self.debug = debug
			self.setTitle("Create an stl file")
			self.setSubTitle("If there are no more changes you would like to make, "
				"press the 'Make Model' button to create your stl file!")
			modelbutton = QPushButton("Make Model")
			modelbutton.clicked.connect(self.save_file)
			vbox = QVBoxLayout()
			vbox.addWidget(modelbutton)
			self.setLayout(vbox)

		def save_file(self, depth=1, double=False, _ascii=False):
			if not self.debug:
				path = QFileInfo(self.filename.path()) \
					if self.grandparent.filename != None else "."
				path = QFileDialog.getSaveFileName(self, "3D Model Creator - Save STL", path)
				path = str(path)
			else:
				path = "/Users/rrao/Documents/Internship/test/testsave/ngc3344_gui_test"
			self.grandparent.curr.make_3d(path, depth, double, _ascii)

	def createRotatePage(self):
		page = QWizardPage()

	def createOptionsPage(self):
		page = QWizardPage()
