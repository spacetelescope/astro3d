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
    #. Scale Intensity - Allows the user to view the image after
       linear, logarithmic, or square root filters are applied.
       **(Shown for FITS only.)**
    #. Model Type Selection - Allows the user to select which type
       of model to make.
    #. Region Selection - Allows the user to draw and save regions.
       Preview button is available on this page and after.
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

    (PG_LOAD, PG_SCALE, PG_TYPE, PG_REG, PG_GALREG, PG_LAYER, PG_CLUS, PG_STAR,
     PG_MAKE) = range(NUM_PAGES)

    def __init__(self, parent=None, debug=False):
        super(ThreeDModelWizard, self).__init__(parent)
        self.parent = parent

        if debug:
            self.addPage(MakeModelPage(parent))
        else:
            self.setPage(self.PG_LOAD, ImageLoadPage(parent, wizard=self))
            #self.setPage(self.PG_RESIZE, ImageResizePage(parent, wizard=self))
            self.setPage(self.PG_SCALE, IntensityScalePage(parent, wizard=self))
            self.setPage(self.PG_TYPE, ModelTypePage(parent, wizard=self))
            self.setPage(self.PG_REG, RegionPage(parent, wizard=self))
            self.setPage(self.PG_GALREG, GalaxyRegionPage(parent, wizard=self))
            self.setPage(self.PG_LAYER, LayerOrderPage(parent, wizard=self))
            self.setPage(self.PG_CLUS, IdentifyPeakPage(parent, wizard=self))
            self.setPage(self.PG_STAR, IdentifyStarPage(parent, wizard=self))
            self.setPage(self.PG_MAKE, MakeModelPage(parent, wizard=self))

        self.setWindowTitle('Astronomy 3D Model Wizard')
        self.setVisible(True)

        # Help button
        self.setOption(QWizard.HaveHelpButton, True)
        self.helpRequested.connect(self._showHelp)

        # Custom preview button
        self.setButtonText(QWizard.CustomButton1, self.tr('&Preview'))
        self.setOption(QWizard.HaveCustomButton1, True)
        self.customButtonClicked.connect(self._previewButtonClicked)

        # Quit entire GUI when done
        self.button(QWizard.FinishButton).clicked.connect(
            QCoreApplication.instance().quit)

    def _showHelp(self):
        """Do this when help button is clicked."""
        page_id = self.currentId()
        title = self.tr('Astronomy 3D Model Wizard Help')

        if page_id == self.PG_LOAD:
            msg = self.tr(
                'Select a FITS or a file with RGB layers (e.g., JPEG or TIFF). '
                'If RGB image is selected, the monochrome intensity is '
                'calculated by summing all the color layers.')
        #elif page_id == self.PG_RESIZE:
        #    msg = self.tr(
        #        """Resize the image to roughly 1k x 1k pixels so that the resultant STL file(s) will not have too many triangles that could crash MakerBot software. If the original dimension is already acceptable, you can skip this step.""")
        elif page_id == self.PG_SCALE:
            msg = self.tr(
                'Scale the intensity such that it is the easiest for you to '
                'mark regions and point sources. This only affects the '
                'display, not the final result.')
        elif page_id == self.PG_TYPE:
            msg = self.tr(
                """A texture map has no intensity. A smooth intensity map has no texture. A textured intensity map has both.

Check the 'special processing' box if the image contains a single spiral galaxy.""")
        elif page_id == self.PG_REG:
            msg = self.tr("""You can either draw a new region by selecting the texture from the drop-down box, or load existing region(s) with the 'Load' button.
Note that 'smooth' or 'remove_star' is not a real texture, but rather used to mark an area to be smoothed over (e.g., foreground star or background galaxy).

To draw, click once on the image and a circular brush will appear. Adjust brush size using Alt+Plus or Alt+Minus to increase or decrease radius by 5 pixels, respectively. Left-click and drag the mouse from inside the region to draw, or from outside to erase.

To erase the region that is just drawn or loaded without saving, click 'Clear'. To save it, click 'Save'.

To draw another region, you must explicitly select a texture from the drop-down box again.

For a saved region, left-click on it in the list to highlight it on the display. You can also right-click on it for options to hide/show, rename (this does not change the texture), edit, or delete. To edit a region, use the brush as described above.""")
        elif page_id == self.PG_GALREG:
            msg = self.tr("""You can either draw a new region by selecting the texture from the drop-down box, or load existing region(s) with the 'Load' button.
Note that 'smooth' or 'remove_star' is not a real texture, but rather used to mark an area to be smoothed over (e.g., foreground star or background galaxy).

To draw, click once on the image and a circular brush will appear. Adjust brush size using Alt+Plus or Alt+Minus to increase or decrease radius by 5 pixels, respectively. Left-click and drag the mouse from inside the region to draw, or from outside to erase.

To erase the region that is just drawn or loaded without saving, click 'Clear'. To save it, click 'Save'.

To draw another region, you must explicitly select a texture from the drop-down box again.

For a saved region, left-click on it in the list to highlight it on the display. You can also right-click on it for options to hide/show, rename (this does not change the texture), edit, or delete. To edit a region, use the brush as described above.

To automatically detect spiral arms and gas regions, draw or load the disk region first, enter the desired percentile numbers, and then click 'Auto Masks'. This overwrites any saved spiral arms and gas masks. Automatically generated masks can be manipulated like other masks as above.""")
        elif page_id == self.PG_LAYER:
            msg = self.tr("""In the final model, textures cannot overlap. Therefore, when two regions of different textures overlap, the layer ordering is used to determine which texture should be given higher priority. For example, if you have:

    lines
    dots
    dots_small

When a 'lines' region is drawn in the middle of another region marked for 'dots', the area of overlap will only have 'lines' texture because in the order above, 'lines' has the highest priority. If the order is reversed, the area of overlap will only show 'dots'.""")
        elif page_id == self.PG_CLUS:
            msg = self.tr(
                'This page will be revamped soon. Help is unavailable.')
        elif page_id == self.PG_STAR:
            msg = self.tr(
                'This page will be revamped soon. Help is unavailable.')
        elif page_id == self.PG_MAKE:
            msg = self.tr("""Model height of 150 corresponds maximum intensity printed as 4.5??? cm for an one-sided intensity map on MakerBot Replicator 2 when a model is split in halves, each half oriented such that its longer side rests on the plate, and scaled to maximum size on MakerBot Desktop. This mean, it would be XXX cm for a two-sided intensity map. For texture map, this number does not affect the printed height, but it is still used for internal calculations.

Base thickness of 20 corresponds to 6 mm (regardless if it is one- or two-sided) when the model is split, oriented, and scaled as above. You want a base that is thick enough so that the model would not topple while being printed on its side, and thin enough so that it is easy to remove model from the plate. MakerBot Replicator 2 glass plate can be 'sticky' to PLA plastic.

To accomodate MakerBot Replicator 2, it is recommended that the model to be split in halves. Otherwise, textures might not be rendered properly even on maximum scale in MakerBot Desktop. In addition, this also prevents MakerBot Desktop from crashing by keeping the STL file for each half under 100 MB. When a model is split, there will be two STL files per model (i.e., '_1.stl' and '_2.stl').

Checking 'save intermediate files' will also generate one '.npz' per region (saved in a sub-directory corresponding to texture name), a text file with a list of stars (if applicable), and a text file with a list of clusters (if applicable).

When you click on 'Make Model' button, only enter the prefix of the output file(s) you wish to save, as suffix and file extension will be automatically added.""")
        else:
            msg = self.tr('Help unavailable.')

        QMessageBox.information(self, title, msg)

    def _previewButtonClicked(self):
        """Do this when preview button is clicked."""
        self.parent.render_preview()

    def hidePreviewButton(self):
        """Hide preview button."""
        self.setButtonLayout(
            [QWizard.HelpButton, QWizard.Stretch,
             QWizard.NextButton, QWizard.FinishButton, QWizard.CancelButton])

    def showPreviewButton(self):
        """Show preview button."""
        self.setButtonLayout(
            [QWizard.HelpButton, QWizard.CustomButton1, QWizard.Stretch,
             QWizard.NextButton, QWizard.FinishButton, QWizard.CancelButton])


class ImageLoadPage(QWizardPage):
    """Allows the user to load an image.

    Parameters
    ----------
    parent
        The instantiating widget.

    """
    def __init__(self, parent=None, wizard=None):
        super(ImageLoadPage, self).__init__()
        self.parent = parent
        self.wizard = wizard
        self.setTitle('Load an Image')

        label = QLabel("""This wizard will help you create a 3D model from a 2D image.

Click the button below to load an image. Currently, only FITS, JPEG, and TIFF are supported.""")
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

    def initializePage(self):
        """This is done right before page is shown."""
        self.wizard.hidePreviewButton()

    def do_load(self):
        """Load image from file."""
        self.parent.fileLoad()
        self.emit(SIGNAL('completeChanged()'))

    def isComplete(self):
        """Only proceed when image is loaded."""
        return self.parent.model3d is not None

    def nextId(self):
        """Proceed to `ImageResizePage`."""
        if self.parent.transformation is None:
            return ThreeDModelWizard.PG_TYPE
        else:
            return ThreeDModelWizard.PG_SCALE


class _ImageResizePage(QWizardPage):
    """Allows the user to change the image dimensions.

    .. note:: NOT USED.

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

    def __init__(self, parent=None, wizard=None):
        super(ImageResizePage, self).__init__()
        self.parent = parent
        self.wizard = wizard
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
        """Proceed to intensity scaling page if applicable,
        else skip to model type selection page.

        """
        if self.parent.transformation is None:
            return ThreeDModelWizard.PG_TYPE
        else:
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
    def __init__(self, parent=None, wizard=None):
        super(IntensityScalePage, self).__init__()
        self.parent = parent
        self.wizard = wizard
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
    def __init__(self, parent=None, wizard=None):
        super(ModelTypePage, self).__init__()
        self.parent = parent
        self.wizard = wizard
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
        self.parent.model3d.is_spiralgal = (self.is_spiral.checkState() ==
                                            Qt.Checked)
        return True

    def nextId(self):
        """Proceed to `RegionPage`."""
        if self.parent.model3d.is_spiralgal:
            return ThreeDModelWizard.PG_GALREG
        else:
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
    def __init__(self, parent=None, wizard=None):
        super(RegionPage, self).__init__(parent)
        self.parent = parent
        self.wizard = wizard
        self.setTitle('Region Selection')

        self._button_width = 110
        self.draw = QComboBox(self)
        self.draw.setFixedWidth(self._button_width)
        self.draw.activated[str].connect(self.drawRegion)
        self.load = QPushButton('Load')
        self.load.setFixedWidth(self._button_width)
        self.load.clicked.connect(self.loadRegion)
        self.clear = QPushButton('Clear')
        self.clear.setFixedWidth(self._button_width)
        self.clear.clicked.connect(self.clearRegion)
        self.clear.setEnabled(False)
        self.save = QPushButton('Save')
        self.save.setFixedWidth(self._button_width)
        self.save.clicked.connect(self.saveRegion)
        self.save.setEnabled(False)

        self.buttongrid = QVBoxLayout()
        self.buttongrid.addWidget(self.draw)
        self.buttongrid.addWidget(self.load)
        self.buttongrid.addWidget(self.clear)
        self.buttongrid.addWidget(self.save)

        self.status = QLabel('Status: Ready!')
        self.status.setWordWrap(True)

        self.initUI()

    def initUI(self):
        """Create the layout."""
        msglabel = QLabel(
            """Select texture from the drop-down box to draw or 'Load' to load from file. To draw another region, you must explicitly select from the drop-down box again. Click 'Help' for more info.""")
        msglabel.setWordWrap(True)

        hbox = QHBoxLayout()
        hbox.addLayout(self.buttongrid)
        hbox.addLayout(self.createRegionList())

        vbox = QVBoxLayout()
        vbox.addWidget(msglabel)
        vbox.addLayout(hbox)
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def initializePage(self):
        """Do this here because need value set by previous page."""
        self.wizard.showPreviewButton()
        self.draw.clear()
        for key in self.parent.model3d.allowed_textures():
            self.draw.addItem(key)

    def cleanupPage(self):
        """Do this when Back button is pressed."""
        self.wizard.hidePreviewButton()

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

        if text is None:
            self.status.setText('Status: No file(s) selected')
            return

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
        self._highlight_item()

    def createRegionList(self):
        """Create the region list, along with a number of
        buttons for various region operations.

        Returns
        -------
        hbox : QHBoxLayout

        """
        self.reg_list = QListWidget()
        self.add_items()
        self.reg_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.reg_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.reg_list.customContextMenuRequested.connect(self._show_item_menu)
        self.reg_list.itemClicked.connect(self._highlight_item)

        hbox = QHBoxLayout()
        hbox.addWidget(self.reg_list)

        return hbox

    def _highlight_item(self, item=None):
        """Highlight region when selected."""
        self.parent.highlightRegion(self.getSelected()[3])

    def _show_item_menu(self, pos):
        """This is shown when user right-clicks on a selected item."""
        item = self.reg_list.itemAt(pos)

        if item is None:
            return

        menu = QMenu('Context Menu', self)
        visAction = menu.addAction('Hide/Show')
        renameAction = menu.addAction('Rename')
        editrgAction = menu.addAction('Edit')
        deleteAction = menu.addAction('Delete')
        action = menu.exec_(self.reg_list.mapToGlobal(pos))

        if action == visAction:
            self.hideshow_region()
        elif action == renameAction:
            self.rename_region()
        elif action == editrgAction:
            self.edit_region()
        elif action == deleteAction:
            self.delete_region()

    def add_items(self):
        """Clear the region list, then adds all regions from
        `~astro3d.gui.astroVisual.AstroGUI` region list.
        This is used after adjustments are made to various regions,
        such as when regions are added or deleted.

        """
        if self.parent.model3d is None:
            return

        items = []

        for key in sorted(self.parent.model3d.region_masks):
            reglist = self.parent.model3d.region_masks[key]
            for i, reg in enumerate(reglist, 1):
                s = '{0}_{1} ({2})'.format(key, i, reg.description)
                if hasattr(reg, 'visible') and not reg.visible:
                    s += ' - HIDDEN'
                items.append(s)

        self.reg_list.clear()
        self.reg_list.addItems(items)

    def getSelected(self):
        """Get all selected regions.

        Returns
        -------
        outrows : list
            A list of row indices for the selected items.

        outkeys : list
            A list of keys for the corresponding regions.

        outvals : list
            A list of indices for the corresponding region lists.

        output : list
            A list of `~astro3d.gui.astroObjects.Region` objects
            for all selected regions.

        """
        outrows = []
        outkeys = []
        outvals = []
        output = []

        for item in self.reg_list.selectedItems():
            s = str(item.text()).split()[0].split('_')
            key = '_'.join(s[:-1])
            val = int(s[-1]) - 1
            outrows.append(self.reg_list.row(item))
            outkeys.append(key)
            outvals.append(val)
            output.append(self.parent.model3d.region_masks[key][val])

        return outrows, outkeys, outvals, output

    def hideshow_region(self):
        """Hide or show selection regions."""
        regions = self.getSelected()[3]
        if len(regions) < 1:
            return

        self.parent.handleRegionVisibility(regions)

        # Update visibility status
        self.add_items()

    def rename_region(self):
        """Change region description."""
        regions = self.getSelected()[3]
        if len(regions) < 1:
            return

        for reg in regions:
            self.parent.renameRegion(reg)

        # Show new name
        self.add_items()

    def edit_region(self):
        keys, vals, regions = self.getSelected()[1:]
        if len(regions) != 1:
            return
        self.parent.editRegion(keys[0], vals[0])
        self.save.setEnabled(True)
        self.status.setText(
            'Status: Click on image to edit {0}'.format(regions[0].description))

    def delete_region(self):
        """Delete the selected region."""
        rows, keys, vals, regions = self.getSelected()
        if len(rows) < 1:
            return

        item_names = []

        for row, key, reg in zip(rows, keys, regions):
            self.parent.deleteRegion(key, reg)
            item = self.reg_list.takeItem(row)
            item_names.append(str(item.text()))

        self.status.setText('Status: {0} deleted!'.format(','.join(item_names)))
        self.emit(SIGNAL('completeChanged()'))

        # Relabel remaining regions
        self.add_items()

    def validatePage(self):
        """Clear brush size message."""
        self.parent.statusBar().showMessage('')
        return True

    def isComplete(self):
        """Only proceed if there is at least one region saved."""
        has_region = False

        for reglist in self.parent.model3d.region_masks.itervalues():
            if len(reglist) > 0:
                has_region = True
                break

        return has_region

    def nextId(self):
        """Proceed to final page if no texture, otherwise to
        layer ordering or cluster selection page.

        """
        if not self.parent.model3d.has_texture:
            return ThreeDModelWizard.PG_MAKE
        elif (self.parent.model3d.is_spiralgal or
              len(self.parent.model3d.texture_names()) < 2):
            return ThreeDModelWizard.PG_CLUS
        else:
            return ThreeDModelWizard.PG_LAYER


class GalaxyRegionPage(RegionPage):
    """Like `RegionPage` but with extra options for automatically
    generating masks for spiral arms and gas.

    """
    def initUI(self):
        """Create the layout."""
        msglabel = QLabel(
            """Select texture from the drop-down box to draw or 'Load' to load from file. To draw another region, you must explicitly select from the drop-down box again. To use 'Auto Masks', draw/load disk first. Click 'Help' for more info.""")
        msglabel.setWordWrap(True)

        hbox = QHBoxLayout()
        hbox.addLayout(self.buttongrid)
        hbox.addLayout(self.createRegionList())

        vbox = QVBoxLayout()
        vbox.addWidget(msglabel)
        vbox.addLayout(hbox)
        vbox.addLayout(self.createAutoMasksLayout())
        vbox.addStretch()
        vbox.addWidget(self.status)

        self.setLayout(vbox)

    def createAutoMasksLayout(self):
        """Create the layout for automatic masks generation."""
        self.find = QPushButton('Auto Masks')
        self.find.setFixedWidth(self._button_width)
        self.find.clicked.connect(self.findRegion)
        self.ptile_hi_text = QLineEdit('75')
        self.ptile_hi_text.setFixedWidth(40)
        self.ptile_lo_text = QLineEdit('55')
        self.ptile_lo_text.setFixedWidth(40)

        hbox = QHBoxLayout()
        hbox.addWidget(self.find)
        hbox.addStretch()
        hbox.addWidget(QLabel('Spiral arms %tile:'))
        hbox.addWidget(self.ptile_hi_text)
        hbox.addStretch()
        hbox.addWidget(QLabel('Gas %tile:'))
        hbox.addWidget(self.ptile_lo_text)
        hbox.addStretch()

        return hbox

    def findRegion(self):
        """Automatically find spiral arms and gas."""
        if (len(self.parent.model3d.region_masks[
                self.parent.model3d.lines_key]) < 1):
            self.status.setText('Status: ERROR - Define the disk first!')
            return

        hi_text = self.ptile_hi_text.text()
        lo_text = self.ptile_lo_text.text()

        if hi_text.isEmpty() or lo_text.isEmpty():
            return

        try:
            hi = float(hi_text)
        except ValueError:
            self.status.setText(
                'Status: ERROR - Invalid spiral arms percentile!')
            return

        try:
            lo = float(lo_text)
        except ValueError:
            self.status.setText('Status: ERROR - Invalid gas percentile!')
            return

        self.parent.findRegion(lo, hi)
        self.save.setEnabled(False)
        self.clear.setEnabled(False)
        self.add_items()
        self.status.setText('Status: Automatically added spiral arms and gas')
        self.emit(SIGNAL('completeChanged()'))


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

    def __init__(self, parent=None, wizard=None):
        super(LayerOrderPage, self).__init__()
        self.parent = parent
        self.wizard = wizard
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
        if self.parent is None or self.parent.model3d is None:
            return

        self.names_list.clear()
        ordered_names = sorted(
            self.parent.model3d.texture_names(),
            key=self.parent.model3d.layer_order.index)
        self.names_list.addItems(ordered_names)

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
        s = [str(self.names_list.item(i).text())
             for i in range(self.names_list.count())]
        new_order = sorted(self.parent.model3d.layer_order,
                           key=lambda x: s.index(x) if x in s else 99)
        self.parent.model3d.layer_order = new_order
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
    def __init__(self, parent=None, wizard=None):
        super(IdentifyPeakPage, self).__init__()
        self.parent = parent
        self.wizard = wizard
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

        self.r_fac_add = QLineEdit('10')
        self.r_fac_add.setMaxLength(3)
        self.r_fac_add.setFixedWidth(40)
        self.r_fac_mul = QLineEdit('5')
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

    def initializePage(self):
        """Do this here because need values from parent."""
        if self.parent._enable_photutil:
            self.findbutton.setDisabled(False)
            self.ntext.setDisabled(False)
        else:
            self.findbutton.setDisabled(True)
            self.ntext.setDisabled(True)

    def do_find(self):
        """Find objects. Can take few seconds to a minute or so."""
        n = int(self.ntext.text())
        self.status.setText(
            'Status: Finding {0} object(s), please wait...'.format(n))
        self.status.repaint()
        self.emit(SIGNAL('completeChanged()'))  # So status would refresh
        self.parent.find_clusters(n)
        self.status.setText(
            'Status: {0} object(s) found!'.format(
                len(self.parent.model3d.peaks['clusters'])))
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_load(self):
        """Load objects from file."""
        self.status.setText('Status: Select clusters file to load')
        self.status.repaint()

        self.parent.load_clusters()

        if ('clusters' not in self.parent.model3d.peaks or
                len(self.parent.model3d.peaks['clusters']) < 1):
            self.status.setText('Status: No clusters loaded!')
        else:
            self.status.setText(
                'Status: {0} object(s) loaded!'.format(
                    len(self.parent.model3d.peaks['clusters'])))

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

        self.parent.model3d.clus_r_fac_add = radd
        self.parent.model3d.clus_r_fac_mul = rmul
        self.parent.save_clusters()
        return True

    def nextId(self):
        """For spiral galaxy, proceed to `MakeModelPage`, otherwise
        to `IdentifyStarPage`.

        """
        if self.parent.model3d.is_spiralgal:
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
    def __init__(self, parent=None, wizard=None):
        super(IdentifyStarPage, self).__init__()
        self.parent = parent
        self.wizard = wizard
        self._proceed_ok = False
        self.setTitle("Identify Stars")

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

        self.r_fac_add = QLineEdit('10')
        self.r_fac_add.setMaxLength(3)
        self.r_fac_add.setFixedWidth(40)
        self.r_fac_mul = QLineEdit('5')
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

    def initializePage(self):
        """Do this here because need values from parent."""
        if self.parent._enable_photutil:
            self.findbutton.setDisabled(False)
            self.ntext.setDisabled(False)
        else:
            self.findbutton.setDisabled(True)
            self.ntext.setDisabled(True)

    def do_find(self):
        """Find objects. Can take few seconds to a minute or so."""
        n = int(self.ntext.text())
        self.status.setText(
            'Status: Finding {0} object(s), please wait...'.format(n))
        self.status.repaint()
        self.parent.find_stars(n)
        self.status.setText(
            'Status: {0} object(s) found!'.format(
                len(self.parent.model3d.peaks['stars'])))
        self._proceed_ok = True
        self.emit(SIGNAL('completeChanged()'))

    def do_load(self):
        """Load objects from file."""
        self.status.setText('Status: Select stars file to load')
        self.status.repaint()

        self.parent.load_stars()

        if ('stars' not in self.parent.model3d.peaks or
                len(self.parent.model3d.peaks['stars']) < 1):
            self.status.setText('Status: No stars loaded!')
        else:
            self.status.setText(
                'Status: {0} object(s) loaded!'.format(
                    len(self.parent.model3d.peaks['stars'])))

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

        self.parent.model3d.star_r_fac_add = radd
        self.parent.model3d.star_r_fac_mul = rmul
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
    def __init__(self, parent=None, wizard=None):
        super(MakeModelPage, self).__init__()
        self.parent = parent
        self.wizard = wizard
        self.setTitle('Create STL file(s)')
        self._proceed_ok = False

        label = QLabel(
            'Adjust some final values below and click the \'Make Model\' '
            'button to create your STL file(s)! Click \'Help\' for more info.')
        label.setWordWrap(True)

        self.heightbox = QLineEdit('100')
        self.heightbox.setMaxLength(4)
        self.heightbox.setFixedWidth(80)
        hgrid = QHBoxLayout()
        hgrid.addWidget(QLabel('Model height:'))
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
