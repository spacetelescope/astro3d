import sys
import os

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from astropy.io import fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class ThreeDModelCreator(QMainWindow):

	def __init__():
		super(ThreeDModelCreator, self).__init__()

	def create_widgets(self):
		self.main_frame = QWidget()

		self.dpi = 80.0
		self.fig = Figure((6.0, 8.0), dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self.main_frame)