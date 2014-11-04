"""Wizard to create 3D models."""
from __future__ import division, print_function

# STDLIB
from functools import partial

# Anaconda
from PyQt4.QtGui import *
from PyQt4.QtCore import *


class ThreeDModelWizard(QWizard):
    """This wizard is meant to guide users through the creation
    of a 3D STL file from a 2D image. It contains:

    #. Load Image - Allows the user to load an image.
    #. Resize Image - Allows the user to resize the image to
       the appropriate dimensions.
    #. Scale Intensity - Allows the user to view the image after
       linear, logarithmic, or square root filters are applied.
    #. Model Type Selection - Allows the user to select which type
       of model to make.
    #. Region Selection - Allows the user to draw and save regions.
    #. Region Layer Ordering - Allows the user to order texture layers,
       except the one used for smoothing. Where they overlap, the one
       in the foreground will take precedence.
       **(Not shown if only 1 texture selected.)**
    #. Star and/or Star Clusters Selection - Allows the user to save the
       desired amount of brightest stars or star clusters to be marked in
       the model.
       **(Not shown for intensity map without textures.)**
    #. Make Model - Allows the user to construct and save the model
       as STL file.

    Parameters
    ----------
    parent : `~astro3d.gui.astroVisual.AstroGUI`
        The instantiating widget.

    debug : bool
        If `True`, wizard jumps straight to the final page.
        All the prior pages will be auto-populated by
        :meth:`~astro3d.gui.AstroGUI.run_auto_login_script`.

    """
    NUM_PAGES = 9

    (PG_LOAD, PG_RESIZE, PG_SCALE, PG_TYPE, PG_REG, PG_LAYER, PG_CLUS, PG_STAR,
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
            self.setPage(self.PG_LAYER, LayerOrderPage(parent))
            self.setPage(self.PG_CLUS, IdentifyPeakPage(parent))
            self.setPage(self.PG_STAR, IdentifyStarPage(parent))
            self.setPage(self.PG_MAKE, MakeModelPage(parent))

        self.setWindowTitle('Astronomy 3D Model Wizard')
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

        label = QLabel("""This wizard will help you create a 3D model from a 2D image.

Click the button below to load an image of a galaxy. Currently, only FITS, JPEG, and TIFF are supported.""")
        label.setWordWrap(True)

        button = QPushButton('Load Image')
        button.clicked.connect(self.do_load)
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(button)
        hbox.addStretch()

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addStretch()
        vbox.addLayout(hbox)
        vbox.addStretch()

        self.setLayout(vbox)

    def do_load(self):
        """Load image from file."""
        self.parent.fileLoad()
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Only proceed when image is loaded."""
        return self.parent.file is not None

    def nextId(self):
        """Proceed to `ImageResizePage`."""
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

    **Intensity Map without Textures**

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

    **Image Rotation**

    If the image is rotated, then the crop function of the engine
    will remove a far greater amount of the image. It has been
    Roshan Rao's experience that a rotated image with dimensions
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
    _MIN_PIXELS = 8.1e5  # 900 x 900
    _MAX_PIXELS = 1.69e6  # 1300 x 1300

    def __init__(self, parent=None):
        super(ImageResizePage, self).__init__()
        self.parent = parent
        self.setTitle('Adjust Image Resolution')

        # See initializePage()
        self.scale = 0
        self.width = 0
        self.height = 0
        self.size_okay = False

        self.messageLabel = QLabel('')
        self.messageLabel.setWordWrap(True)

        self.xtext = QLineEdit('')
        self.xtext.setFixedWidth(80)
        self.xtext.textChanged.connect(self.setYText)
        self.ylabel = QLineEdit('')
        self.ylabel.setFixedWidth(80)
        self.ylabel.setReadOnly(True)

        grid = QHBoxLayout()
        grid.addStretch()
        grid.addWidget(QLabel('Resize to '))
        grid.addWidget(self.xtext)
        grid.addWidget(QLabel('x'))
        grid.addWidget(self.ylabel)
        grid.addStretch()

        self.button = QPushButton('Resize')
        self.button.clicked.connect(self.changeSize)
        self.button.setEnabled(False)
        hbbox = QHBoxLayout()
        hbbox.addStretch()
        hbbox.addWidget(self.button)
        hbbox.addStretch()

        self.status = QLabel('')
        self.status.setWordWrap(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self.messageLabel)
        vbox.addLayout(grid)
        vbox.addLayout(hbbox)
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def initializePage(self):
        """Called right before the page is displayed.
        Since this page is created before the image is loaded,
        there is no height/width to obtain at the beginning.
        However, if the user follows the proper order, then there
        will be at the time the page is viewed, and so this method can
        obtain the correct height/width.

        """
        super(ImageResizePage, self).initializePage()

        orig_width = self.parent.file.data.shape[1]
        orig_height = self.parent.file.data.shape[0]
        self.scale = orig_height / orig_width
        self.messageLabel.setText(
            'Original image is {0} x {1} pixels. Resize it to a dimension '
            'that is between 900 x 900 and 1300 x 1300 pixels, if necessary. '
            'Scaling up is not recommended.\n\n'.format(
                orig_width, orig_height))

        if orig_width <= orig_height:
            self.xtext.setText('1000')
        else:
            self.xtext.setText(str(int(1000 / self.scale)))

        is_okay = self._sanity_check(orig_width * orig_height)[0]

        if is_okay:
            self.status.setText('Status: Resizing is unnecessary.')
            self.size_okay = True
            self.emit(SIGNAL('completeChanged()'))
        else:
            self.status.setText('Status: Resizing must be done!')

    def setYText(self):
        """Called whenever ``xtext`` is edited.
        Automatically change ``ylabel`` to match the correct aspect ratio.
        Also update ``width`` and ``height``.

        """
        s = self.xtext.text()

        if s.isEmpty():
            return

        try:
            width = int(s)
        except ValueError:
            self.status.setText('Status: ERROR - Invalid width!')
            return

        self.width = width
        self.height = int(self.width * self.scale)
        self.ylabel.setText(str(self.height))
        self._text_postedit()

    def _sanity_check(self, numpixels):
        is_okay = False
        text = ''

        if numpixels < self._MIN_PIXELS:
            text = 'Resolution is too low!'
        elif numpixels > self._MAX_PIXELS:
            text = 'Resolution is too high!'
        else:
            is_okay = True

        return is_okay, text

    def _text_postedit(self):
        is_okay, text = self._sanity_check(self.height * self.width)

        if is_okay:
            self.status.setText('Status: Click \'Resize\' to resize image.')
            self.button.setEnabled(True)
        else:
            self.status.setText('Status: {0}'.format(text))
            self.button.setEnabled(False)

    def changeSize(self):
        """Called when the resize button is clicked.
        Calls :meth:`~astro3d.gui.AstroGUI.resizeImage` with
        the input height and width.

        """
        self.parent.resizeImage(self.width, self.height)
        self.status.setText('Status: Image resized to {0} x {1}'.format(
            self.width, self.height))
        self.size_okay = True
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Only proceed when allowed dimensions are set."""
        return self.size_okay

    def nextId(self):
        """Proceed to `IntensityScalePage`."""
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
    choices : dict
        Maps ID to scale name.

    bgroup : QButtonGroup
        Contains the three checkboxes. It ensures that the boxes are exclusive and sets an ID for each box so it knows which one is checked.

    """
    def __init__(self, parent=None):
        super(IntensityScalePage, self).__init__()
        self.parent = parent
        self.setTitle('Scale Image Intensities')

        label = QLabel("""Select a scaling option to better view the image. This is for display only; It does not affect output.""")
        label.setWordWrap(True)

        self.choices = parent.IMG_TRANSFORMATIONS
        self.bgroup = QButtonGroup()
        self.bgroup.setExclusive(True)
        button_grid = QHBoxLayout()

        for key, val in self.choices.items():
            button = QCheckBox(val)
            self.bgroup.addButton(button)
            self.bgroup.setId(button, key)
            if key == 0:
                button.setChecked(True)
            button_grid.addWidget(button)

        applybutton = QPushButton('Apply')
        applybutton.clicked.connect(self.apply)
        hbbox = QHBoxLayout()
        hbbox.addStretch()
        hbbox.addWidget(applybutton)
        hbbox.addStretch()

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addStretch()
        vbox.addLayout(button_grid)
        vbox.addStretch()
        vbox.addLayout(hbbox)
        vbox.addStretch()

        self.setLayout(vbox)

    def apply(self):
        """Called when user clicks the apply button.
        Gets the ID of the checked box from ``bgroup``,
        then calls :meth:`~astro3d.gui.AstroGUI.setTransformation`
        with the appropriate argument.

        """
        _id = self.bgroup.checkedId()
        self.parent.setTransformation(self.choices[_id])

    def nextId(self):
        """Proceed to `ModelTypePage`."""
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

        label2 = QLabel('\nCheck the box below if applicable: ')
        label2.setWordWrap(True)

        self.is_spiral = QCheckBox(
            'Special processing for single spiral galaxy', self)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(button_grid)
        vbox.addStretch()
        vbox.addWidget(label2)
        vbox.addWidget(self.is_spiral)

        self.setLayout(vbox)

    def validatePage(self):
        """Pass the selected values to GUI parent."""
        self.parent.model_type = self.bgroup.checkedId()
        self.parent.is_spiral = self.is_spiral.checkState() == Qt.Checked
        return True

    def nextId(self):
        """Proceed to `RegionPage`."""
        return ThreeDModelWizard.PG_REG


class RegionPage(QWizardPage):
    """Sets the display to the interactive
    `~astro3d.gui.star_scenes.RegionStarScene`.
    Allows the user to manipulate regions.

    .. note::

        GUI components are instantiated in :meth:`initUI` and
        :meth:`createRegionList`.

    Parameters
    ----------
    parent
        The instantiating widget.

    Attributes
    ----------
    draw : QPushButton
        Activates the `~astro3d.gui.star_scenes.RegionStarScene`.

    save : QPushButton
        Saves a region.

    clear : QPushButton
        Clears a region that is being drawn, but does not exit the `~astro3d.gui.star_scenes.RegionStarScene`.

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
        self.setTitle('Region Selection')

        msglabel = QLabel(
            """Select region the drop-down box to draw or 'Load' to load from file. To draw, click on the image. If you are dissatisfied, press 'Clear'. Once you are satisfied, press 'Save'. Once saved, the region cannot be removed. To draw another region, you must explicitly select from the drop-down box again.""")
        msglabel.setWordWrap(True)

        self.draw = QComboBox(self)
        self.draw.activated[str].connect(self.drawRegion)
        self.load = QPushButton('Load')
        self.load.clicked.connect(self.loadRegion)
        self.clear = QPushButton('Clear')
        self.clear.clicked.connect(self.clearRegion)
        self.clear.setEnabled(False)
        self.save = QPushButton('Save')
        self.save.clicked.connect(self.saveRegion)
        self.save.setEnabled(False)

        buttongrid = QVBoxLayout()
        buttongrid.addWidget(self.draw)
        buttongrid.addWidget(self.load)
        buttongrid.addWidget(self.clear)
        buttongrid.addWidget(self.save)

        hbox = QHBoxLayout()
        hbox.addLayout(buttongrid)
        hbox.addLayout(self.createRegionList())

        self.status = QLabel('Status: Ready!')
        self.status.setWordWrap(True)

        vbox = QVBoxLayout()
        vbox.addWidget(msglabel)
        vbox.addLayout(hbox)
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def initializePage(self):
        """Do this here because need value set by previous page."""
        self.draw.clear()
        for key in self.parent.REGION_TEXTURES[self.parent.is_spiral]:
            self.draw.addItem(key)

    def drawRegion(self, qtext):
        """Start an interactive `~astro3d.gui.star_scenes.RegionStarScene`.
        Also enables save and clear buttons.

        """
        key = str(qtext).lower()
        self.parent.drawRegion(key)
        self.save.setEnabled(True)
        self.clear.setEnabled(True)
        self.status.setText('Status: Click on image to select {0} '
                            'region'.format(key))

    def loadRegion(self):
        """Load saved regions."""
        self.status.setText('Status: Select region file(s) to load')
        self.status.repaint()
        text = self.parent.loadRegion()
        self.save.setEnabled(True)
        self.clear.setEnabled(True)
        self.status.setText('Status: {0} loaded from file(s)'.format(text))

    def saveRegion(self):
        """Save the currently drawn region.
        Also disables save and clear buttons.

        """
        text = self.parent.saveRegion()
        self.save.setEnabled(False)
        self.clear.setEnabled(False)
        self.add_items()
        self.status.setText('Status: Saving done (see log)')
        self.emit(SIGNAL('completeChanged()'))

    def clearRegion(self):
        """Clear the currently drawn region, without exiting
        the `~astro3d.gui.star_scenes.RegionStarScene`.

        """
        self.parent.clearRegion()
        self.clear.setEnabled(False)
        self.save.setEnabled(False)
        self.status.setText('Status: Region cleared (not saved)!')

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
        #box = QDialogButtonBox(Qt.Vertical)
        #self.show_ = box.addButton('Show', QDialogButtonBox.ActionRole)
        #self.hide = box.addButton('Hide', QDialogButtonBox.ActionRole)
        #self.delete = box.addButton('Delete', QDialogButtonBox.ActionRole)
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
        `~astro3d.gui.astroVisual.AstroGUI` region list.
        This is used after adjustments are made to various regions,
        such as when regions are added or deleted.

        """
        if self.parent.file is None:
            return

        items = []

        for key in sorted(self.parent.file.regions):
            reglist = self.parent.file.regions[key]
            items += ['{0}_{1}'.format(key, i + 1) for i in range(len(reglist))]

        self.reg_list.clear()
        self.reg_list.addItems(items)

    #def enableButtons(self):
    #    """Enable/disable the show, hide, and delete buttons
    #    depending on which regions are selected.
    #
    #    * Show is enabled if any selected regions are hidden.
    #    * Hide is enabled if any selected regions are visible.
    #    * Delete is enabled as long as at least one region is selected.
    #
    #    """
    #    selected = self.getSelected()
    #    if selected:
    #        self.delete.setEnabled(True)
    #
    #        if any([reg.visible for reg in selected]):
    #            self.hide.setEnabled(True)
    #        else:
    #            self.hide.setEnabled(False)
    #
    #        if not all([reg.visible for reg in selected]):
    #            self.show_.setEnabled(True)
    #        else:
    #            self.show_.setEnabled(False)
    #    else:
    #        self.show_.setEnabled(False)
    #        self.hide.setEnabled(False)
    #        self.delete.setEnabled(False)

    #def getSelected(self):
    #    """Get all selected regions.
    #
    #    Returns
    #    -------
    #    output
    #       A list of `~astro3d.gui.astroObjects.Region` objects
    #       for all selected regions.
    #
    #    """
    #    output = []
    #    for item in self.reg_list.selectedItems():
    #        key, val = item.split('_')
    #        output.append(self.parent.file.regions[key][int(val)])
    #    return output

    #def show_region(self):
    #    """Displays any hidden regions among the selected regions."""
    #    self.parent.showRegion(self.getSelected())
    #    self.enableButtons()

    #def hide_region(self):
    #    """Hides any displayed regions among the selected regions."""
    #    self.parent.hideRegion(self.getSelected())
    #    self.enableButtons()

    #def delete_region(self):
    #    """Deletes the selected region."""
    #    self.parent.deleteRegion(self.getSelected())
    #    self.add_items()
    #    self.enableButtons()
    #    self.status.setText('Status: Region deleted!')
    #    self.status.repaint()
    #    self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Only proceed if there is at least one region saved."""
        has_region = False

        for reglist in self.parent.file.regions.itervalues():
            if len(reglist) > 0:
                has_region = True
                break

        return has_region

    def nextId(self):
        """Proceed to final page if no texture, otherwise to
        layer ordering or cluster selection page.

        """
        if self.parent.model_type in (1, 2):  # No texture
            return ThreeDModelWizard.PG_MAKE
        elif self.parent.is_spiral or len(self.parent.file.texture_names()) < 2:
            return ThreeDModelWizard.PG_CLUS
        else:
            return ThreeDModelWizard.PG_LAYER


# http://stackoverflow.com/questions/9166087/move-row-up-and-down-in-pyqt4
class LayerOrderPage(QWizardPage):
    """Enable texture layers ordering. Particularly, dots and lines.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    _UP = -1
    _DOWN = 1

    def __init__(self, parent=None):
        super(LayerOrderPage, self).__init__()
        self.parent = parent
        self.setTitle('Texture Layers Ordering')

        label = QLabel("""Order the texture layers such that the top layer will overwrite the following layer(s).""")
        label.setWordWrap(True)

        self.names_list = QListWidget()
        self.names_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.names_list.itemClicked.connect(self._enable_buttons)

        self.move_up = QPushButton('Move Up')
        self.move_up.clicked.connect(partial(self._move_item, self._UP))
        self.move_up.setEnabled(False)

        self.move_down = QPushButton('Move Down')
        self.move_down.clicked.connect(partial(self._move_item, self._DOWN))
        self.move_down.setEnabled(False)

        buttonbox = QVBoxLayout()
        buttonbox.addWidget(self.move_up)
        buttonbox.addWidget(self.move_down)
        buttonbox.addStretch()

        hbox = QHBoxLayout()
        hbox.addWidget(self.names_list)
        hbox.addLayout(buttonbox)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hbox)
        vbox.addStretch()

        self.setLayout(vbox)

    def initializePage(self):
        """Do this here because need values from parent."""
        if self.parent is None or self.parent.file is None:
            return

        self.names_list.clear()
        names = self.parent.file.texture_names()
        if len(names) > 1:
            self.names_list.addItems(names)

    def _enable_buttons(self):
        # Only one item selection allowed at a time
        i = self.names_list.currentRow()

        if i < 1:
            self.move_up.setEnabled(False)
            self.move_down.setEnabled(True)
        elif i >= self.names_list.count() - 1:
            self.move_up.setEnabled(True)
            self.move_down.setEnabled(False)
        else:
            self.move_up.setEnabled(True)
            self.move_down.setEnabled(True)

    def _move_item(self, direction):
        if direction not in (self._DOWN, self._UP):
            return

        pos = self.names_list.currentRow()
        item = self.names_list.takeItem(pos)
        new_pos = pos + direction
        self.names_list.insertItem(new_pos, item)

        for i in range(self.names_list.count()):
            item = self.names_list.item(i)
            self.names_list.setItemSelected(item, False)

        self.move_up.setEnabled(False)
        self.move_down.setEnabled(False)

    def validatePage(self):
        """Pass the selected values to GUI parent."""
        self.parent.layer_order = [str(self.names_list.item(i).text())
                                   for i in range(self.names_list.count())]
        return True

    def nextId(self):
        """Proceed to cluster selection page."""
        return ThreeDModelWizard.PG_CLUS


class IdentifyPeakPage(QWizardPage):
    """Activate the `~astro3d.gui.star_scenes.ClusterStarScene`.
    Allow the user to select and save the desired number
    of brightest star clusters.

    .. note::

        This page is not needed for models without textures.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None):
        super(IdentifyPeakPage, self).__init__()
        self.parent = parent
        self._proceed_ok = False
        self.setTitle("Identify Star Clusters")

        msglabel = QLabel("""Choose one: Find, load, or manual. Click on existing circle to remove object, or click on new object to add. Once you are satisfied, click 'Next'.""")
        msglabel.setWordWrap(True)

        self.findbutton = QPushButton('Find')
        self.findbutton.clicked.connect(self.do_find)
        self.ntext = QLineEdit('25')
        self.ntext.setMaxLength(4)
        self.ntext.setFixedWidth(80)
        nobjgrid = QHBoxLayout()
        nobjgrid.addWidget(self.findbutton)
        nobjgrid.addWidget(self.ntext)
        nobjgrid.addWidget(QLabel('objects (might take a while)'))
        nobjgrid.addStretch()

        self.loadbutton = QPushButton('Load')
        self.loadbutton.clicked.connect(self.do_load)
        hbbox1 = QHBoxLayout()
        hbbox1.addWidget(self.loadbutton)
        hbbox1.addStretch()

        self.manbutton = QPushButton('Manual/Skip')
        self.manbutton.clicked.connect(self.do_manual)
        hbbox2 = QHBoxLayout()
        hbbox2.addWidget(self.manbutton)
        hbbox2.addStretch()

        self.r_fac_add = QLineEdit('15')
        self.r_fac_add.setMaxLength(3)
        self.r_fac_add.setFixedWidth(40)
        self.r_fac_mul = QLineEdit('1')
        self.r_fac_mul.setMaxLength(3)
        self.r_fac_mul.setFixedWidth(40)
        radbox = QHBoxLayout()
        radbox.addWidget(self.r_fac_add)
        radbox.addWidget(QLabel('+'))
        radbox.addWidget(self.r_fac_mul)
        radbox.addWidget(QLabel('x flux / flux_max'))
        radbox.addStretch()
        radframe = QGroupBox('Radius')
        radframe.setLayout(radbox)

        self.status = QLabel('Status: Ready!')

        vbox = QVBoxLayout()
        vbox.setSpacing(1)
        vbox.addWidget(msglabel)
        vbox.addStretch()
        vbox.addLayout(nobjgrid)
        vbox.addLayout(hbbox1)
        vbox.addLayout(hbbox2)
        vbox.addWidget(radframe)
        vbox.addStretch()
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
            'Status: {0} object(s) found!'.format(
                len(self.parent.file.peaks['clusters'])))
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_load(self):
        """Load objects from file."""
        self.status.setText('Status: Select clusters file to load')
        self.status.repaint()

        self.parent.load_clusters()

        if ('clusters' not in self.parent.file.peaks or
                len(self.parent.file.peaks['clusters']) < 1):
            self.status.setText('Status: No clusters loaded!')
        else:
            self.status.setText(
                'Status: {0} object(s) loaded!'.format(
                    len(self.parent.file.peaks['clusters'])))

        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_manual(self):
        """Manual selection from display."""
        self.status.setText('Status: Click on display to select clusters or Next to skip')
        self.status.repaint()
        self.parent.manual_clusters()
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Proceed only after attempt to find objects is done."""
        return self._proceed_ok

    def validatePage(self):
        """Pass the selected values to GUI parent."""
        try:
            radd = float(self.r_fac_add.text())
        except ValueError:
            self.status.setText(
                'Status: ERROR - Invalid radius additive factor!')
            return False

        try:
             rmul = float(self.r_fac_mul.text())
        except ValueError:
            self.status.setText(
                'Status: ERROR - Invalid radius multiplicative factor!')
            return False

        self.parent._clus_r_fac_add = radd
        self.parent._clus_r_fac_mul = rmul
        self.parent.save_clusters()
        return True

    def nextId(self):
        """For spiral galaxy, proceed to `MakeModelPage`, otherwise
        to `IdentifyStarPage`.

        """
        if self.parent.is_spiral:
            return ThreeDModelWizard.PG_MAKE
        else:
            return ThreeDModelWizard.PG_STAR


class IdentifyStarPage(QWizardPage):
    """Activate the `~astro3d.gui.star_scenes.ClusterStarScene`.
    Allow the user to select and save the desired number of stars.

    Similar to `IdentifyPeakPage` but without auto-find.

    .. note::

        This page is not needed for models without textures.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None):
        super(IdentifyStarPage, self).__init__()
        self.parent = parent
        self._proceed_ok = False
        self.setTitle("Identify Stars")

        msglabel = QLabel("""Choose one: Load or manual. Click on existing circle to remove object, or click on new object to add. Once you are satisfied, click 'Next'.""")
        msglabel.setWordWrap(True)

        self.loadbutton = QPushButton('Load')
        self.loadbutton.clicked.connect(self.do_load)
        hbbox1 = QHBoxLayout()
        hbbox1.addWidget(self.loadbutton)
        hbbox1.addStretch()

        self.manbutton = QPushButton('Manual/Skip')
        self.manbutton.clicked.connect(self.do_manual)
        hbbox2 = QHBoxLayout()
        hbbox2.addWidget(self.manbutton)
        hbbox2.addStretch()

        self.r_fac_add = QLineEdit('15')
        self.r_fac_add.setMaxLength(3)
        self.r_fac_add.setFixedWidth(40)
        self.r_fac_mul = QLineEdit('1')
        self.r_fac_mul.setMaxLength(3)
        self.r_fac_mul.setFixedWidth(40)
        radbox = QHBoxLayout()
        radbox.addWidget(self.r_fac_add)
        radbox.addWidget(QLabel('+'))
        radbox.addWidget(self.r_fac_mul)
        radbox.addWidget(QLabel('x flux / flux_max'))
        radbox.addStretch()
        radframe = QGroupBox('Radius')
        radframe.setLayout(radbox)

        self.status = QLabel('Status: Ready!')

        vbox = QVBoxLayout()
        vbox.setSpacing(1)
        vbox.addWidget(msglabel)
        vbox.addStretch()
        vbox.addLayout(hbbox1)
        vbox.addLayout(hbbox2)
        vbox.addWidget(radframe)
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def do_load(self):
        """Load objects from file."""
        self.status.setText('Status: Select stars file to load')
        self.status.repaint()

        self.parent.load_stars()

        if ('stars' not in self.parent.file.peaks or
                len(self.parent.file.peaks['stars']) < 1):
            self.status.setText('Status: No stars loaded!')
        else:
            self.status.setText(
                'Status: {0} object(s) loaded!'.format(
                    len(self.parent.file.peaks['stars'])))

        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_manual(self):
        """Manual selection from display."""
        self.status.setText('Status: Click on display to select stars or Next to skip')
        self.status.repaint()
        self.parent.manual_stars()
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Proceed only after attempt to find objects is done."""
        return self._proceed_ok

    def validatePage(self):
        """Pass the selected values to GUI parent."""
        try:
            radd = float(self.r_fac_add.text())
        except ValueError:
            self.status.setText(
                'Status: ERROR - Invalid radius additive factor!')
            return False

        try:
             rmul = float(self.r_fac_mul.text())
        except ValueError:
            self.status.setText(
                'Status: ERROR - Invalid radius multiplicative factor!')
            return False

        self.parent._star_r_fac_add = radd
        self.parent._star_r_fac_mul = rmul
        self.parent.save_stars()
        return True

    def nextId(self):
        """Proceed to `MakeModelPage`."""
        return ThreeDModelWizard.PG_MAKE


class MakeModelPage(QWizardPage):
    """Allow the user to select a save location and filename,
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

        label = QLabel("""If there are no more changes you would like to make, enter the max height above base for scaled intensity, and then press the 'Make Model' button to create your STL file(s)!

To accomodate MakerBot Replicator 2, it is recommended that the model to be split in halves.""")
        label.setWordWrap(True)

        self.heightbox = QLineEdit('200')
        self.heightbox.setMaxLength(4)
        self.heightbox.setFixedWidth(80)
        hgrid = QHBoxLayout()
        hgrid.addWidget(QLabel('Max height:'))
        hgrid.addWidget(self.heightbox)
        hgrid.addStretch()

        self.depthbox = QLineEdit('20')
        self.depthbox.setMaxLength(3)
        self.depthbox.setFixedWidth(80)
        dgrid = QHBoxLayout()
        dgrid.addWidget(QLabel('Base thickness:'))
        dgrid.addWidget(self.depthbox)
        dgrid.addStretch()

        self.split_halves = QCheckBox(
            'Split into two halves (recommended)', self)
        self.split_halves.setChecked(True)

        self.save_extras = QCheckBox('Save intermediate files', self)
        self.save_extras.setChecked(True)

        modelbutton = QPushButton('Make Model')
        modelbutton.clicked.connect(self.save_file)
        hmodbox = QHBoxLayout()
        hmodbox.addStretch()
        hmodbox.addWidget(modelbutton)
        hmodbox.addStretch()

        self.status = QLabel('Status: Ready!')

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hgrid)
        vbox.addLayout(dgrid)
        vbox.addWidget(self.split_halves)
        vbox.addWidget(self.save_extras)
        vbox.addLayout(hmodbox)
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def save_file(self):
        """Save files. This is the end point of the GUI."""
        self.status.setText('Status: Please wait...')
        self.status.repaint()

        try:
            height = float(self.heightbox.text())
        except ValueError:
            self.status.setText('Status: ERROR - Invalid height!')
            return

        try:
            depth = int(self.depthbox.text())
        except ValueError:
            self.status.setText('Status: ERROR - Invalid depth!')
            return

        # OK to proceed even if no files saved
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

        do_split = self.split_halves.checkState() == Qt.Checked
        save_all = self.save_extras.checkState() == Qt.Checked

        msg = self.parent.fileSave(
            height=height, depth=depth, split_halves=do_split,
            save_all=save_all)
        self.status.setText('Status: {0}'.format(msg))

    def isComplete(self):
        """Proceed only after attempt to save model is done."""
        return self._proceed_ok

    def nextId(self):
        """Last page goes nowhere."""
        return -1
