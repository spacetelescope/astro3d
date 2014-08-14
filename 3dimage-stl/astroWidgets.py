import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from astroObjects import *
import numpy as np
import scipy

class PixelDlg(QDialog):

	changed = pyqtSignal()

	def __init__(self, parent, modal=False):
		super(PixelDlg, self).__init__(parent)
		self.setAttribute(Qt.WA_DeleteOnClose)
		self.setModal(modal)
		self.parent = parent
		self.initUI(parent.pmin, parent.pmax)
		self.resize(self.minimumSizeHint())
		self.resize(self.width() + 30, self.height())
		self.setWindowTitle("Active Pixel Range")
		self.setVisible(True)
		self.setFixedSize(self.width(), self.height())
		self.setSizeGripEnabled(False)
		

	def initUI(self, pmin, pmax):
		minlabel = QLabel("Set min pixel value:")
		maxlabel = QLabel("Set max pixel value:")
		self.minValue = QLineEdit(str(pmin))
		self.maxValue = QLineEdit(str(pmax))
		self.minValue.setValidator(QIntValidator(0, 9999, self.minValue))
		self.maxValue.setValidator(QIntValidator(0, 9999, self.maxValue))
		box = QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close)
		box.setOrientation(Qt.Horizontal)
		box.rejected.connect(self.reject)
		box.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
		vbox = QVBoxLayout()
		layout = QGridLayout()
		layout.addWidget(minlabel, 0, 0)
		layout.addWidget(self.minValue, 0, 1)
		layout.addWidget(maxlabel, 0, 2)
		layout.addWidget(self.maxValue, 0, 3)
		vbox.addLayout(layout)
		vbox.addWidget(box, 0, Qt.AlignCenter)
		self.setLayout(vbox)

	def apply(self):
		pmin = self.minValue.text()
		pmax = self.maxValue.text()
		try:
			if len(pmin) == 0:
				raise ValueError, ("There must be a minimum pixel value")
			if len(pmax) == 0:
				raise ValueError, ("There must be a maximum pixel value")
			if int(pmax) - int(pmin) <= 0:
				raise ValueError, ("The minimum pixel value must be less than the maximum")
		except ValueError, e:
			QMessageBox.warning(self, "Input Error", unicode(e))
			if len(pmax) == 0:
				self.maxValue.selectAll()
				self.maxValue.setFocus()
				return
			else:
				self.minValue.selectAll()
				self.minValue.setFocus()
				return
		self.parent.pmin = int(pmin)
		self.parent.pmax = int(pmax)
		self.changed.emit()

class FilterDlg(QDialog):

	changed = pyqtSignal()
	reverted = pyqtSignal()

	def __init__(self, parent, modal=False):
		super(FilterDlg, self).__init__(parent)
		self.setAttribute(Qt.WA_DeleteOnClose)
		self.setModal(modal)
		self.parent = parent
		self.initUI(parent.filter_radius, parent.filter)
		self.resize(self.sizeHint())
		self.setWindowTitle("Apply Image Filter")
		self.setVisible(True)
		self.setFixedSize(self.width(), self.height())
		self.setSizeGripEnabled(False)

	def initUI(self, radius, _filter):
		gaussian = QCheckBox("Gaussian")
		tophat = QCheckBox("Tophat")
		boxcar = QCheckBox("Boxcar")

		self.group = QButtonGroup()
		self.group.setExclusive(True)
		self.group.addButton(gaussian)
		self.group.addButton(tophat)
		self.group.addButton(boxcar)
		self.group.setId(gaussian, 0)
		self.group.setId(tophat, 1)
		self.group.setId(boxcar, 2)

		button_grid = QGridLayout()
		button_grid.addWidget(gaussian, 0, 0)
		button_grid.addWidget(tophat, 0, 1)
		button_grid.addWidget(boxcar, 0, 2)

		if _filter == 'gaussian':
			gaussian.setChecked(True)
		elif _filter == 'tophat':
			tophat.setChecked(True)
		elif _filter == 'boxcar':
			boxcar.setChecked(True)

		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(1)
		self.slider.setMaximum(20)
		self.slider.setValue(radius)
		self.slider.setTickInterval(5)
		self.slider.setTickPosition(QSlider.TicksBelow)

		label = QLabel(str(radius))
		self.slider.valueChanged.connect(lambda: label.setText(str(self.slider.value())))

		box = QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close)
		self.rvrt = box.addButton("Revert", QDialogButtonBox.ResetRole)
		box.setOrientation(Qt.Vertical)
		box.rejected.connect(self.reject)
		box.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
		self.rvrt.clicked.connect(self.revert)
		if not self.parent.curr.filtered:
			self.rvrt.setEnabled(False)

		sliderbox = QHBoxLayout()
		sliderbox.addWidget(self.slider)
		sliderbox.addWidget(label)
		vbox = QVBoxLayout()
		vbox.addLayout(button_grid)
		vbox.addLayout(sliderbox)
		hbox = QHBoxLayout()
		hbox.addLayout(vbox)
		hbox.addWidget(box)
		self.setLayout(hbox)

	def apply(self):
		_id = self.group.checkedId()
		if _id == 0:
			self.parent.filter = 'gaussian'
		elif _id == 1:
			self.parent.filter = 'tophat'
		elif _id == 2:
			self.parent.filter = 'boxcar'
		self.parent.filter_radius = self.slider.value()
		self.rvrt.setEnabled(True)
		self.changed.emit()

	def revert(self):
		self.rvrt.setEnabled(False)
		self.reverted.emit()

class ImgCompressDlg(QDialog):

	def __init__(self, parent, _file):
		super(ImgCompressDlg, self).__init__(parent)
		self.setAttribute(Qt.WA_DeleteOnClose)
		self.setModal(True)
		self._file = _file
		self.parent = parent
		self.initUI()
		self.resize(self.sizeHint())
		self.setWindowTitle("Rescale the image")
		self.setVisible(True)
		self.setFixedSize(self.size())
		self.setSizeGripEnabled(False)

	def initUI(self):
		dLabel = QLabel("Dimensions (pixels):")
		h, w = self._file['Data'].shape
		self.xtext = QLineEdit(str(w))
		self.ylabel = QLabel(" x " + str(h))
		self.xtext.setValidator(QIntValidator(1, w, self.xtext))
		self.ycheckbox = QCheckBox("Keep Aspect Ratio Constant")
		self.xtext.textChanged.connect(self.setYText)
		box = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
		box.accepted.connect(self.accept)
		box.rejected.connect(self.reject)
		box.setOrientation(Qt.Horizontal)
		self.scaleSpinBox = QSpinBox()
		self.scaleSpinBox.setRange(1, 100)
		self.scaleSpinBox.setValue(100)
		self.scaleSpinBox.valueChanged.connect(self.scaleChanged)
		scaleBox = QHBoxLayout()
		scaleBox.addWidget(self.scaleSpinBox)
		scaleBox.addWidget(QLabel("%"))
		grid = QGridLayout()
		grid.addLayout(scaleBox, 0, 0)
		grid.addWidget(dLabel, 0, 1)
		grid.addWidget(self.xtext, 0, 2)
		grid.addWidget(self.ylabel, 0, 3)
		vbox = QVBoxLayout()
		vbox.addLayout(grid)
		vbox.addWidget(box)
		self.setLayout(vbox)

	def setYText(self):
		width = self.xtext.text()
		if not width.isEmpty():
			h, w = self._file['Data'].shape
			scale = int(width) / float(w)
			height = int(h * scale)
			self.ylabel.setText(" x " + str(height))
			if self.xtext.hasFocus():
				self.scaleSpinBox.setValue(int(scale * 100))

	def scaleChanged(self):
		if self.scaleSpinBox.hasFocus():
			scale = self.scaleSpinBox.value() / 100.0
			self.xtext.setText(str(int(scale * self._file['Data'].shape[1])))

	def accept(self):
		self.parent.resizeImage(self._file, int(self.xtext.text()))
		QDialog.accept(self)

class RegionEditDlg(QDialog):
	
	def __init__(self, parent):
		super(RegionEditDlg, self).__init__(parent)
		self.setModal(True)
		self.parent = parent
		self.initUI()
		self.resize(self.sizeHint())
		self.setWindowTitle("Region Edit Tool")
		self.setVisible(True)
		self.setFixedSize(self.width(), self.height())
		self.setSizeGripEnabled(False)

	def initUI(self):
		self.reg_list = QListWidget()
		self.add_items()
		self.reg_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
		box1 = QDialogButtonBox(QDialogButtonBox.Close)
		box1.setOrientation(Qt.Vertical)
		box2 = QDialogButtonBox(Qt.Horizontal)
		self.show_ = box2.addButton("Show", QDialogButtonBox.ActionRole)
		self.hide = box2.addButton("Hide", QDialogButtonBox.ActionRole)
		self.merge = box1.addButton("Merge", QDialogButtonBox.ActionRole)
		self.split = box1.addButton("Split", QDialogButtonBox.ActionRole)
		self.delete = box2.addButton("Delete", QDialogButtonBox.ActionRole)
		box1.rejected.connect(self.reject)
		self.show_.clicked.connect(self.show_region)
		self.hide.clicked.connect(self.hide_region)
		self.merge.clicked.connect(self.merge_region)
		self.split.clicked.connect(self.split_region)
		self.delete.clicked.connect(self.delete_region)
		self.enableButtons()
		self.reg_list.itemSelectionChanged.connect(self.enableButtons)
		vbox = QVBoxLayout()
		vbox.addWidget(box2)
		vbox.addWidget(self.reg_list)
		hbox = QHBoxLayout()
		hbox.addLayout(vbox)
		hbox.addWidget(box1)
		self.setLayout(hbox)

	def add_items(self):
		self.reg_list.clear()
		self.reg_list.addItems([reg['Name'] for reg in self.parent.regions])

	def enableButtons(self):
		selected = map(self.parent.get_region, map(lambda item: item.text(), self.reg_list.selectedItems()))
		if selected:
			self.delete.setEnabled(True)
			self.hide.setEnabled(True) if any(map(lambda reg: reg.isVisible(), selected)) else self.hide.setEnabled(False)
			self.show_.setEnabled(True) if not all(map(lambda reg: reg.isVisible(), selected)) else self.show_.setEnabled(False)
			self.merge.setEnabled(True) if len(selected) > 1 else self.merge.setEnabled(False)
			self.split.setEnabled(True) if len(selected) == 1 and isinstance(selected[0], MergedRegion) else self.split.setEnabled(False)
		else:
			self.show_.setEnabled(False)
			self.hide.setEnabled(False)
			self.merge.setEnabled(False)
			self.split.setEnabled(False)
			self.delete.setEnabled(False)

	def getSelected(self):
		return map(self.parent.get_region, map(lambda item: item.text(), self.reg_list.selectedItems()))

	def show_region(self):
		self.parent.showRegion(self.getSelected())
		self.enableButtons()

	def hide_region(self):
		self.parent.hideRegion(self.getSelected())
		self.enableButtons()

	def merge_region(self):
		name, ok = QInputDialog.getText(self, "Merge Regions", "Enter the name of the merged region:")
		if ok:
			self.parent.mergeRegions(name, self.getSelected())
			self.add_items()
			self.enableButtons()

	def split_region(self):
		self.parent.splitRegion(self.getSelected()[0])
		self.add_items()
		self.enableButtons()

	def delete_region(self):
		self.parent.deleteRegion(self.getSelected())
		self.add_items()
		self.enableButtons()

class ThreeDModelWizard(QWizard):

	def __init__(self, parent):
		super(ThreeDModelWizard, self).__init__(parent)
		self.parent = parent
		self.addPage(self.createIntroPage())
		self.addPage(self.createImageLoadPage())
		self.addPage(self.ImageResizePage(self.parent))
		self.addPage(self.IntensityScalePage(self.parent))
		self.addPage(self.RegionPage(self.parent))
		self.addPage(self.IdentifyPeakPage(self.parent))
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
				print scale
				print self.height
				print width
				print self.height * scale
				print self.width * scale
				self.width = width
				self.height = self.height * scale
				self.ylabel.setText(" x " + str(int(self.height)))

		def changeSize(self):
			self.grandparent.resizeImage(self.grandparent.curr, int(self.width), int(self.height))

		def initializePage(self):
			super(ThreeDModelWizard.ImageResizePage, self).initializePage()
			h, w = self.grandparent.curr['Data'].shape
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
			self.reg_list.addItems([reg['Name'] for reg in self.grandparent.regions])

		def enableButtons(self):
			selected = self.getSelected()
			if selected:
				self.delete.setEnabled(True)
				self.hide.setEnabled(True) if any(map(lambda reg: reg.isVisible(), selected)) else self.hide.setEnabled(False)
				self.show_.setEnabled(True) if not all(map(lambda reg: reg.isVisible(), selected)) else self.show_.setEnabled(False)
				self.merge.setEnabled(True) if len(selected) > 1 else self.merge.setEnabled(False)
				self.split.setEnabled(True) if len(selected) == 1 and isinstance(selected[0], MergedRegion) else self.split.setEnabled(False)
			else:
				self.show_.setEnabled(False)
				self.hide.setEnabled(False)
				self.merge.setEnabled(False)
				self.split.setEnabled(False)
				self.delete.setEnabled(False)

		def getSelected(self):
			#return map(self.grandparent.get_region, map(lambda item: item.text(), self.reg_list.selectedItems()))
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

	def createRotatePage(self):
		page = QWizardPage()

	def createIdentifyPeakPage(self):
		page = QWizardPage()
		page.setTitle("Identify 15 Brightest Star Clusters")

	def createOptionsPage(self):
		page = QWizardPage()

"""class ImageSampleWizard(QWizard):

	def __init__(self, parent):
		super(ImageSampleWizard, self).__init__(parent)
		self.parent = parent
		self.setWindowTitle("Sample an image")
		self.setVisible(True)
		#self.scene = parent.widget.current_scene

	def createImageSelectPage(self):
		page = QWizardPage()
		page.setTitle("Image Selection")
		page.setSubTitle("Select an image to sample")
		image_list = QListWidget()
		image_list.addItems([file_['Name'] for file_ in self.parent.files])
		image_list.setSelectionMode(QAbstractItemView.SingleSelection)



	def createRegionSelectPage(self):
		pass

	class AttributesPage(QWizardPage):

		def __init__(self, parent):
			super(AttributesPage, self).__init__()
			self.setTitle("Sample Characteristics")
			self.setSubTitle("Set the aperture and step size to be used"
				" when sampling this page")
			radiusLabel = QLabel("Aperture Radius:")
			stepLabel = QLabel("Step size")
			rSpinBox = QSpinBox()
			sSpinBox = QSpinBox()
			rSpinBox.setRange(1, 20)
			sSpinBox.setRange(30, 300)
			sSpinBox.setSingleStep(10)
			rSpinBox.setValue(self.parent.aperture_radius)
			sSpinBox.setValue(self.parent.step)
			rSpinBox.valueChanged.connect(lambda: self.set_radius(rSpinBox.value()))
			sSpinBox.valueChanged.connect(lambda: self.set_step(sSpinBox.value()))
			grid = QGridLayout()
			grid.addWidget(radiusLabel, 0, 0)
			grid.addWidget(stepLabel, 1, 0)
			grid.addWidget(rSpinBox, 0, 1)
			grid.addWidget(sSpinBox, 1, 1)
			self.setLayout(grid)

		def set_radius(self, radius):
			self.parent.parent.aperture_radius = radius

		def set_step(self, step):
			self.parent.parent.step = step
			self.scene.step = step
			self.scene.drawGrid()

	def createFinishPage(self):
		pass

class ImageSampleDlg(QDialog):

	def __init__(self, parent):
		super(ImageSampleDlg, self).__init__(parent)
		self.setModal(True)
		self.setAttribute(Qt.WA_DeleteOnClose)
		self.parent = parent
		self.initUI()
		self.resize(self.minimumSizeHint())
		self.resize(self.width() + 30, self.height())
		self.setWindowTitle("Sample characteristics")
		self.setVisible(True)
		self.setFixedSize(self.width(), self.height())
		self.setSizeGripEnabled(False)

		self.scene = parent.widget.current_scene
		self.scene.drawGrid()

	def initUI(self):
		radiusLabel = QLabel("Aperture Radius:")
		stepLabel = QLabel("Step size")
		rSpinBox = QSpinBox()
		sSpinBox = QSpinBox()
		rSpinBox.setRange(1, 20)
		sSpinBox.setRange(30, 300)
		sSpinBox.setSingleStep(10)
		rSpinBox.setValue(self.parent.aperture_radius)
		sSpinBox.setValue(self.parent.step)
		box = QDialogButtonBox(QDialogButtonBox.Cancel)
		imstat = box.addButton("Get imstat", QDialogButtonBox.YesRole)
		box.setOrientation(Qt.Vertical)
		box.rejected.connect(self.reject)
		imstat.clicked.connect(self.accept)
		rSpinBox.valueChanged.connect(lambda: self.set_radius(rSpinBox.value()))
		sSpinBox.valueChanged.connect(lambda: self.set_step(sSpinBox.value()))
		grid = QGridLayout()
		grid.addWidget(radiusLabel, 0, 0)
		grid.addWidget(stepLabel, 1, 0)
		grid.addWidget(rSpinBox, 0, 1)
		grid.addWidget(sSpinBox, 1, 1)
		hbox = QHBoxLayout()
		hbox.addLayout(grid)
		hbox.addWidget(box)
		self.setLayout(hbox)

	def set_radius(self, radius):
		self.parent.aperture_radius = radius

	def set_step(self, step):
		self.parent.step = step
		self.scene.step = step
		self.scene.drawGrid()"""
