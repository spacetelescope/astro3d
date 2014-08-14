import sys, os
from PyQt4 import QtGui, QtCore

class Image_Loader(QtGui.QWidget):

	def __init__(self):
		super(Image_Loader, self).__init__()
		self.size = QtCore.QSize(500, 500)

		self.filename = "/Users/rrao/Documents/Internship/PyQt"
		self.view = QtGui.QGraphicsView()
		self.scene = QtGui.QGraphicsScene(self)
		self.view.setScene(self.scene)
		self.initUI()

	def initUI(self):
		vbox = QtGui.QVBoxLayout()
		vbox.addWidget(self.view)
		load_button = QtGui.QPushButton('Load Image')
		self.connect(load_button, QtCore.SIGNAL('clicked()'), self.addPixmap)
		quit_button = QtGui.QPushButton('Quit')
		self.connect(quit_button, QtCore.SIGNAL('clicked()'), QtCore.QCoreApplication.instance().quit)
		hbox = QtGui.QHBoxLayout()
		hbox.addWidget(load_button)
		hbox.addWidget(quit_button)
		vbox.addLayout(hbox)

		self.setLayout(vbox)
		self.move(300, 300)
		self.resize(self.size)
		self.setWindowTitle('Image Loader')
		self.show()

	def addPixmap(self):
		path = QtCore.QFileInfo(self.filename).path()
		fname = QtGui.QFileDialog.getOpenFileName(self, "Image Loader - Load Image", \
			path, "Pixmap Files (*.bmp *.jpg *.png *.xpm)")
		if fname.isEmpty():
			return
		pic = QtGui.QPixmap(fname)
		pic = pic.scaled(self.size, QtCore.Qt.KeepAspectRatio)
		self.scene.addItem(QtGui.QGraphicsPixmapItem(pic))


def main():
	app = QtGui.QApplication(sys.argv)
	img = Image_Loader()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
