"""astro3d UI Application
"""
import sys
import logging

from ginga import AstroImage

from .gui import (Controller, MainWindow, Model)
from .gui.start_ui_app import start_ui_app


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        self.logger = logging.getLogger('astro3d')

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)

        self.viewer = MainWindow(self.logger)
        self.model = Model()

        # Setup the connections.
        self.viewer.set_callback('drag-drop', self.drop_file)
        self.viewer.set_callback('open-file', self.open_file)
        self.viewer.set_callback('quit', self.quit)

        # Ok, let's start.
        self.viewer.show()

    def load_file(self, filepath):
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(filepath)
        self.viewer.image_viewer.set_image(image)
        self.viewer.setWindowTitle(filepath)

    def drop_file(self, whatisthis, paths):
        fileName = paths[0]
        self.load_file(fileName)

    def open_file(self):
        filename = self.viewer.get_filename()
        if len(filename) != 0:
            self.load_file(filename)

    def quit(self, *args):
        self.logger.info("Attempting to shut down the application...")
        self.viewer.deleteLater()


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
