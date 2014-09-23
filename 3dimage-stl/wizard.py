"""Wizard to create 3D models."""
from __future__ import division, print_function

# Anaconda
from PyQt4.QtGui import *
from PyQt4.QtCore import *


class ThreeDModelWizard(QWizard):
    """
    This wizard is meant to guide users through the creation
    of a 3D STL file from a 2D image. It contains:

    #. Load Image - Allows the user to load an image.
    #. Resize Image - Allows the user to resize the image to
       the appropriate dimensions.
    #. Scale Intensity - Allows the user to view the image after
       linear, logarithmic, or square root filters are applied.
    #. Model Type Selection - Allows the user to select which type
       of model to make.
    #. Region Selection - Allows the user to draw and save regions.
    #. Star or Clusters Selection - Allows the user to save the
       desired amount of brightest star clusters to be marked in
       the model.
       **(Not shown for intensity map without textures.)**
    #. Make Model - Allows the user to construct and save the model
       as STL file.

    Parameters
    ----------
    parent : ``astroVisual.AstroGUI``
        The instantiating widget.

    debug : bool
        If `True`, wizard jumps straight to the final page.
        All the prior pages will be auto-populated by
        ``AstroGUI.run_auto_login_script()``.

    Attributes
    ----------
    parent
        Same as input.

    """
    NUM_PAGES = 7

    (PG_LOAD, PG_RESIZE, PG_SCALE, PG_TYPE, PG_REG, PG_CLUS,
     PG_MAKE) = range(NUM_PAGES)

    def __init__(self, parent=None, debug=False):
        super(ThreeDModelWizard, self).__init__(parent)
        self.parent = parent

        if debug:
            self.addPage(MakeModelPage(parent))
        else:
            self.setPage(self.PG_LOAD, ImageLoadPage(parent))
            self.setPage(self.PG_RESIZE, ImageResizePage(parent))
            self.setPage(self.PG_SCALE, IntensityScalePage(parent))
            self.setPage(self.PG_TYPE, ModelTypePage(parent))
            self.setPage(self.PG_REG, RegionPage(parent))
            self.setPage(self.PG_CLUS, IdentifyPeakPage(parent))
            self.setPage(self.PG_MAKE, MakeModelPage(parent))

        self.setWindowTitle('Create a 3D Model of a Galaxy')
        self.setVisible(True)

        # Quit entire GUI when done
        self.button(QWizard.FinishButton).clicked.connect(
            QCoreApplication.instance().quit)


class ImageLoadPage(QWizardPage):
    """Allows the user to load an image.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None):
        super(ImageLoadPage, self).__init__()
        self.parent = parent
        self.setTitle('Load an Image')
        default_height = 150

        label = QLabel("""
This wizard will help you create a 3D model from a 2D image.

First, enter the max height above base for scaled intensity.

Then, click the button below to load an image of a galaxy.
Currently, only FITS and JPEG are supported.


""")
        label.setWordWrap(True)

        heightlabel = QLabel('Max height:')
        self.heightbox = QLineEdit(str(default_height))
        self.heightbox.setMaxLength(4)
        hgrid = QGridLayout()
        hgrid.addWidget(heightlabel, 0, 0)
        hgrid.addWidget(self.heightbox, 0, 1)

        button = QPushButton('Load Image')
        button.clicked.connect(self.do_load)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hgrid)
        vbox.addWidget(button)

        self.setLayout(vbox)

    def do_load(self):
        self.parent.fileLoad(height=float(self.heightbox.text()))
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        return self.parent.file is not None

    def nextId(self):
        """Proceed to Resize Image page."""
        return ThreeDModelWizard.PG_RESIZE


class ImageResizePage(QWizardPage):
    """Allows the user to change the image dimensions.

    Contains an input box for image width, while automatically
    generating the correct height for the image aspect ratio.

    This page is important because the image must be of a
    specific size. Aside from the slowdown of the engine as it
    processes a very high-res image, the actual MakerBot software
    cannot handle more than approximately 2 million triangles.
    The full size model is printed front and back and split into
    two halves, meaning that a 1000x1000 image will create a
    4 million triangle model, which will subsequently produce two
    2 million triangle halves. Since part of the image is cropped
    during the model creation process, the average image can
    probably be most 1300x1300.

    **Intensity Map without Textures***

    The restriction against smaller image sizes stems from the
    need to provide texture. If there are too few pixels, then
    the dots and lines making up the texture will be spaced too
    far apart, and will therefore be very course.

    If texture is unnecessary, then a much lower resolution may
    be used (down to 500x500 at least). It is important to note
    that when printing smaller models (models that are not split
    into two halves), that a lower resolution is required (the
    smaller model will still be about 4 million triangles), and
    therefore texture cannot be added. However, the engine is
    currently optimized for a 1000x1000 pixel image, and will
    therefore require some adjustment before a 500x500 pixel
    image can be run through.

    **Image Rotation***

    If the image is rotated, then the crop function of the engine
    will remove a far greater amount of the image. It has been
    Roshan's experience that a rotated image with dimensions
    2000x2000 will be cropped to an image just slightly over
    1000x1000, which is perfect for this project. If we do not
    want users to have to deal with this issue, it may be possible
    to resize all images to 2000x2000 here, then resize the image
    again after the crop.

    Parameters
    ----------
    parent
        The instantiating widget.

    Attributes
    ----------
    parent
        Same as input.

    height, width : int
        The current height/width of the loaded image array.

    messageLabel : QLabel
        Informs the user whether the image needs to be increased
        or decreased in size.

    xtext : QLineEdit
        Allows the user to input the new desired width for the image.

    ylabel : QLabel
        Displays the appropriate height of the image given the user
        input for width in order to maintain the aspect ratio.

    """
    def __init__(self, parent=None):
        super(ImageResizePage, self).__init__()
        self.parent = parent

        self.setTitle('Adjust Image Resolution')

        # These are instantiated as 0, but are set to the
        # actual height/width when the page is initialized.
        self.height = 0
        self.width = 0
        self.size_okay = False

        self.messageLabel = QLabel('')
        self.messageLabel.setWordWrap(True)
        dLabel = QLabel('Dimensions (pixels):')
        self.xtext = QLineEdit(str(self.width))
        self.xtext.setMaxLength(4)
        self.ylabel = QLabel(' x {0}'.format(str(self.height)))
        self.xtext.textChanged.connect(self.setYText)

        button = QPushButton('Resize')
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
        """Shows the appropriate message label along with
        the correct height/width.

        Parameters
        ----------
        height, width : int

        """
        self.size_okay = False
        self.height = height
        self.width = width
        numpixels = self.height * self.width
        recommended_size = 'until it is between 900x900 and 1300x1300 pixels.'

        if numpixels < 8.1e5:  # 900x900
            message = ('The current image resolution is too low.\n'
                       'Please increase the resolution ' + recommended_size)
        elif numpixels > 1.69e6:  # 1300x1300
            message = ('The current image resolution is too high.\n'
                       'Please decrease the resolution ' + recommended_size)

        else:
            message = ('The current image resolution is perfect!\n'
                       'You can move on to the next page.')
            self.size_okay = True

        self.messageLabel.setText(message)
        self.xtext.setText(str(width))

    def setYText(self):
        """Called whenever ``xtext`` is edited.
        Automatically changes ``ylabel`` to match the correct aspect ratio.
        Also changes ``width`` and ``height``.

        """
        width = self.xtext.text()

        if width.isEmpty():
            return

        width = int(width)
        scale = int(width) / float(self.width)
        self.width = width
        self.height = self.height * scale
        self.ylabel.setText(' x {0}'.format(str(int(self.height))))

    def changeSize(self):
        """Called when the resize button is clicked.
        Calls ``AstroGUI.resizeImage()`` with the input height and width.

        """
        w = int(self.width)
        h = int(self.height)
        self.parent.resizeImage(w, h)
        self.resetUI(h, w)
        self.emit(SIGNAL('completeChanged()'))

    def initializePage(self):
        """Called right before the page is displayed.
        Since this page is created before the image is loaded,
        there is no height/width to obtain at the beginning.
        However, if the user follows the proper order, then there
        will be at the time the page is viewed, and so this method can
        obtain the correct height/width.

        """
        super(ImageResizePage, self).initializePage()
        h, w = self.parent.file.data.shape
        self.resetUI(h, w)

    def isComplete(self):
        return self.size_okay

    def nextId(self):
        """Proceed to Intensity Scaling page."""
        return ThreeDModelWizard.PG_SCALE


class IntensityScalePage(QWizardPage):
    """Contains three checkboxes for choosing between a linear,
    logarithmic, or square root filter.

    Parameters
    ----------
    parent
        The instantiating widget.

    Attributes
    ----------
    parent
        Same as input.

    choices : dict
        Maps ID to scale name.

    bgroup : QButtonGroup
        Contains the three checkboxes. It ensures that the
        boxes are exclusive and sets an ID for each box so
        it knows which one is checked.

    """
    def __init__(self, parent=None):
        super(IntensityScalePage, self).__init__()
        self.parent = parent
        self.setTitle('Scale Image Intensities')

        label = QLabel("""
Select a scaling option to better view the image.
This is for display only; It does not affect output.""")
        label.setWordWrap(True)

        self.choices = parent.IMG_TRANSFORMATIONS
        self.bgroup = QButtonGroup()
        self.bgroup.setExclusive(True)
        button_grid = QGridLayout()

        for key, val in self.choices.items():
            button = QCheckBox(val)
            self.bgroup.addButton(button)
            self.bgroup.setId(button, key)
            if key == 0:
                button.setChecked(True)
            button_grid.addWidget(button, 0, key)

        applybutton = QPushButton('Apply')
        applybutton.clicked.connect(self.apply)
        button_grid.addWidget(applybutton, 1, 2)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(button_grid)

        self.setLayout(vbox)

    def apply(self):
        """Called when user clicks the apply button.
        Gets the ID of the checked box from ``bgroup``,
        then calls ``AstroGUI.setTransformation()`` with
        the appropriate argument.

        """
        _id = self.bgroup.checkedId()
        self.parent.setTransformation(self.choices[_id])

    def nextId(self):
        """Proceed to Model Type page."""
        return ThreeDModelWizard.PG_TYPE


class ModelTypePage(QWizardPage):
    """Page to select what type of model to make.

    * Flat texture map (one-sided only)
    * Smooth intensity map (one- or two-sided)
    * Textured intensity map (one- or two-sided)

    Parameters
    ----------
    parent
        The instantiating widget.

    Attributes
    ----------
    parent
        Same as input.

    choices : dict
       Maps ID to model type.

    bgroup : QButtonGroup
       Mutually exclusive selection buttons.

    """
    def __init__(self, parent=None):
        super(ModelTypePage, self).__init__()
        self.parent = parent
        self.setTitle('Select Model Type')

        label = QLabel('Select the type of 3D model to print:')
        label.setWordWrap(True)

        self.choices = parent.MODEL_TYPES
        self.bgroup = QButtonGroup()
        self.bgroup.setExclusive(True)
        button_grid = QGridLayout()

        for key, val in self.choices.items():
            button = QCheckBox(val)
            self.bgroup.addButton(button)
            self.bgroup.setId(button, key)
            if key == 4:
                button.setChecked(True)
            button_grid.addWidget(button, key, 0)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(button_grid)

        self.setLayout(vbox)

    def validatePage(self):
        """Pass the selected value to GUI parent."""
        self.parent.model_type = self.bgroup.checkedId()
        return True

    def nextId(self):
        """Proceed to Region Selection page."""
        return ThreeDModelWizard.PG_REG


class RegionPage(QWizardPage):
    """Sets the display to the interactive ``RegionStarScene``.
    Allows the user to manipulate regions.

    .. note::

        GUI components are instantiated in :meth:`initUI` and
        :meth:`createRegionList`.

    .. todo::

        Need to enable texture selection (currently hardcoded).

        Need to allow saving more than one disk region
        (will need it for interacting galaxies).

        Need a drag circle widget for disk/star.

    Parameters
    ----------
    parent
        The instantiating widget.

    Attributes
    ----------
    parent
        Same as input.

    draw : QPushButton
        Activates the ``RegionStarScene``.

    save : QPushButton
        Saves a region.

    clear : QPushButton
        Clears a region that is being drawn, but does not exit
        the ``RegionStarScene``.

    reg_list : QListWidget
        Lists all previously drawn regions.

    show, hide : QPushButton
        Shows/hides a region. (Disabled for now.)

    delete : QPushButton
        Deletes a selected region. (Disabled for now.)

    """
    def __init__(self, parent=None):
        super(RegionPage, self).__init__(parent)
        self.parent = parent
        self.setTitle('Region Draw/Edit Page')
        self.initUI()

    def initUI(self):
        """Creates all buttons and adds them to the page
        in the proper layout.

        """
        msglabel = QLabel("""
Select region the drop-down box to draw or load from file. To draw, click on the image. If you are dissatisfied, press 'Clear Region'. Once you are satisfied, press 'Save Region'. Once saved, the region cannot be removed. You can save multiple regions of the same region type, EXCEPT for 'Disk'. To draw another region, you must explicitly select from the drop-down box again.
""")
        msglabel.setWordWrap(True)
        self.status = QLabel('Status: Ready!')
        self.status.setWordWrap(True)

        self.draw = QComboBox(self)
        for key in self.parent.REGION_TEXT.itervalues():
            self.draw.addItem(key)
        self.load = QPushButton('Load Region(s)')
        self.clear = QPushButton('Clear Region(s)')
        self.save = QPushButton('Save Region(s)')

        self.draw.activated[str].connect(self.drawRegion)
        self.load.clicked.connect(self.loadRegion)
        self.save.clicked.connect(self.saveRegion)
        self.clear.clicked.connect(self.clearRegion)

        self.save.setEnabled(False)
        self.clear.setEnabled(False)

        buttongrid = QGridLayout()
        buttongrid.addWidget(self.draw, 0, 0)
        buttongrid.addWidget(self.load, 0, 1)
        buttongrid.addWidget(self.clear, 0, 2)
        buttongrid.addWidget(self.save, 0, 3)

        hbox = self.createRegionList()

        vbox = QVBoxLayout()
        vbox.addWidget(msglabel)
        vbox.addLayout(buttongrid)
        vbox.addLayout(hbox)
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def drawRegion(self, qtext):
        """Start an interactive ``RegionStarScene``.
        Also enables save and clear buttons, while
        disabling the draw button.

        .. note::

            Only allows one disk for now.

        """
        text = str(qtext).lower()

        # Special cases where text is not exact match of key
        if 'spiral' in text:
            key = 'spiral'
        else:
            key = text

        self.parent.drawRegion(key)
        self.save.setEnabled(True)
        self.clear.setEnabled(True)

        self.status.setText('Status: Click on image to draw {0}'.format(key))
        self.status.repaint()

    def loadRegion(self):
        self.status.setText('Status: Select region file(s) to load')
        self.status.repaint()

        text = self.parent.loadRegion()
        self.save.setEnabled(True)
        self.clear.setEnabled(True)
        self.status.setText('Status: {0} loaded from file(s)'.format(text))

    def saveRegion(self):
        """Save the currently drawn region.
        Also disables save and clear buttons, while enabling
        the draw button.

        .. note::

            Only allows one disk for now.

        """
        text = self.parent.saveRegion()
        self.save.setEnabled(False)
        self.clear.setEnabled(False)
        self.add_items()
        self.status.setText('Status: Saving done (see log)')
        self.emit(SIGNAL('completeChanged()'))

    def clearRegion(self):
        """Clear the currently drawn region, without exiting
        the ``RegionStarScene``."""
        self.parent.clearRegion()
        self.clear.setEnabled(False)
        self.save.setEnabled(False)

        self.status.setText('Status: Region cleared (not saved)!')
        self.status.repaint()

    def createRegionList(self):
        """Create the region list, along with a number of
        buttons for various region operations.

        Returns
        -------
        hbox : QHBoxLayout

        """
        self.reg_list = QListWidget()
        self.add_items()
        self.reg_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # These do not work properly - disabled for now
        box = QDialogButtonBox(Qt.Vertical)
        self.show_ = box.addButton('Show', QDialogButtonBox.ActionRole)
        self.hide = box.addButton('Hide', QDialogButtonBox.ActionRole)
        self.delete = box.addButton('Delete', QDialogButtonBox.ActionRole)
        #self.show_.clicked.connect(self.show_region)
        #self.hide.clicked.connect(self.hide_region)
        #self.delete.clicked.connect(self.delete_region)
        #self.enableButtons()
        #self.reg_list.itemSelectionChanged.connect(self.enableButtons)

        hbox = QHBoxLayout()
        hbox.addWidget(self.reg_list)
        #hbox.addWidget(box)
        return hbox

    def add_items(self):
        """Clear the region list, then adds all regions from
        ``AstroGUI`` region list. This is used after adjustments
        are made to various regions, such as when regions are added
        or deleted.

        """
        items = []

        for key in sorted(self.parent.regions):
            reglist = self.parent.regions[key]
            items += ['{0}_{1}'.format(key, i + 1) for i in range(len(reglist))]

        self.reg_list.clear()
        self.reg_list.addItems(items)

    def enableButtons(self):
        """Enable/disable the show, hide, and delete buttons
        depending on which regions are selected.

        * Show is enabled if any selected regions are hidden.
        * Hide is enabled if any selected regions are visible.
        * Delete is enabled as long as at least one region is selected.

        """
        selected = self.getSelected()
        if selected:
            self.delete.setEnabled(True)

            if any([reg.visible for reg in selected]):
                self.hide.setEnabled(True)
            else:
                self.hide.setEnabled(False)

            if not all([reg.visible for reg in selected]):
                self.show_.setEnabled(True)
            else:
                self.show_.setEnabled(False)
        else:
            self.show_.setEnabled(False)
            self.hide.setEnabled(False)
            self.delete.setEnabled(False)

    def getSelected(self):
        """Get all selected regions.

        Returns
        -------
        output
           A list of Region objects for all selected regions.

        """
        output = []

        for item in self.reg_list.selectedItems():
            key, val = item.split('_')
            output.append(self.parent.regions[key][int(val)])

        return output

    def show_region(self):
        """Displays any hidden regions among the selected regions."""
        self.parent.showRegion(self.getSelected())
        self.enableButtons()

    def hide_region(self):
        """Hides any displayed regions among the selected regions."""
        self.parent.hideRegion(self.getSelected())
        self.enableButtons()

    def delete_region(self):
        """Deletes the selected region."""
        self.parent.deleteRegion(self.getSelected())
        self.add_items()
        self.enableButtons()

        self.status.setText('Status: Region deleted!')
        self.status.repaint()

        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Only proceed if there is at least one region saved."""
        has_region = False

        for reglist in self.parent.regions.itervalues():
            if len(reglist) > 0:
                has_region = True
                break

        return has_region

    def nextId(self):
        """Proceed to star clusters page."""
        if self.parent.model_type in (1, 2):  # No texture
            return ThreeDModelWizard.PG_MAKE
        else:
            return ThreeDModelWizard.PG_CLUS


class IdentifyPeakPage(QWizardPage):
    """Activates the ``ClusterStarScene``.
    Allows the user to select and save the desired number
    of brightest star clusters.

    .. note::

        Need to add de-selected stars to be smoothed out.
        This requires File.stars to handle them like clusters, not regions.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None):
        super(IdentifyPeakPage, self).__init__()
        self.parent = parent
        self._proceed_ok = False
        self.setTitle("Identify Bright Objects")

        msglabel = QLabel("""
Either load from file or find new objects, not both. When you click 'Find Objects', the given number of brightest objects will be highlighted on the image for you. THIS TAKES UP TO A MINUTE TO COMPLETE, PLEASE BE PATIENT!

Some of these objects may not actually represent star clusters, but may instead be single stars, or other objects. Clicking on the identifying circles will remove that point from the list.

Once you are satisfied, click the 'Save Objects' button.
            """)
        msglabel.setWordWrap(True)

        self.ntext = QLineEdit('25')
        self.ntext.setMaxLength(4)
        label = QLabel('objects to find')
        nobjgrid = QGridLayout()
        nobjgrid.addWidget(self.ntext, 0, 0)
        nobjgrid.addWidget(label, 0, 1)

        self.status = QLabel('Status: Ready!')

        self.findbutton = QPushButton('Find Objects')
        self.findbutton.clicked.connect(self.do_find)
        self.loadbutton = QPushButton('Load Objects')
        self.loadbutton.clicked.connect(self.do_load)
        self.savebutton = QPushButton('Save Objects')
        self.savebutton.clicked.connect(self.do_save)
        self.savebutton.setEnabled(False)
        buttongrid = QGridLayout()
        buttongrid.addWidget(self.findbutton, 0, 0)
        buttongrid.addWidget(self.loadbutton, 0, 1)
        buttongrid.addWidget(self.savebutton, 0, 2)

        vbox = QVBoxLayout()
        vbox.addWidget(msglabel)
        vbox.addLayout(nobjgrid)
        vbox.addLayout(buttongrid)
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def do_find(self):
        """Find objects. Can take few seconds to a minute or so."""
        n = int(self.ntext.text())
        self.status.setText(
            'Status: Finding {0} object(s), please wait...'.format(n))
        self.status.repaint()
        self.parent.find_clusters(n)
        self.status.setText(
            'Status: {0} object(s) found! Remember to save them.'.format(
                len(self.parent.file.clusters)))
        self.savebutton.setEnabled(True)
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_save(self):
        """Save clusters to ``File.clusters``."""
        self.parent.save_clusters()
        self.status.setText(
            'Status: {0} object(s) saved!'.format(
                len(self.parent.file.clusters)))
        self.status.repaint()
        self.savebutton.setEnabled(False)

    def do_load(self):
        """Load objects from file."""
        self.status.setText('Status: Select clusters file to load')
        self.status.repaint()

        self.parent.load_clusters()

        if self.parent.file.clusters is None:
            self.status.setText('Status: No clusters loaded!')
        else:
            self.status.setText(
                'Status: {0} object(s) loaded! Remember to save '
                'them.'.format(len(self.parent.file.clusters)))

        self.savebutton.setEnabled(True)
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Proceed only after attempt to find objects is done."""
        return self._proceed_ok

    def nextId(self):
        """Proceed to make model page."""
        return ThreeDModelWizard.PG_MAKE


class MakeModelPage(QWizardPage):
    """Allows the user to select a save location and filename,
    then creates STL file(s). This is the last page.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None):
        super(MakeModelPage, self).__init__()
        self.parent = parent
        self.setTitle('Create STL file(s)')
        self._proceed_ok = False

        label = QLabel("""
If there are no more changes you would like to make, press the 'Make Model' button to create your STL file(s)!

To accomodate MakerBot Replicator 2, it is recommended that the model to be split in halves.

""")
        label.setWordWrap(True)

        self.status = QLabel('Status: Ready!')

        self.split_halves = QCheckBox(
            'Split into two halves (recommended)', self)
        self.split_halves.setChecked(True)

        self.save_extras = QCheckBox('Save intermediate files', self)
        self.save_extras.setChecked(True)

        modelbutton = QPushButton('Make Model')
        modelbutton.clicked.connect(self.save_file)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(self.split_halves)
        vbox.addWidget(self.save_extras)
        vbox.addWidget(modelbutton)
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def save_file(self):
        """Save files. This is the end point of the GUI.

        """
        self.status.setText('Status: Please wait...')
        self.status.repaint()

        # OK to proceed even if no files saved
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

        if self.split_halves.checkState() == Qt.Checked:
            do_split = True
        else:
            do_split = False

        if self.save_extras.checkState() == Qt.Checked:
            save_all = True
        else:
            save_all = False

        msg = self.parent.fileSave(split_halves=do_split, save_all=save_all)
        self.status.setText('Status: {0}'.format(msg))

    def isComplete(self):
        """Proceed only after attempt to save model is done."""
        return self._proceed_ok

    def nextId(self):
        """Last page goes nowhere."""
        return -1
