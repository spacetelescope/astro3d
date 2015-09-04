"""astro3d UI Application
"""
import sys
import logging

from ginga import AstroImage

from .util.signal_slot import Signal
from .gui import (Controller, MainWindow, Model)
from .gui.start_ui_app import start_ui_app


class Signals(object):
    """Container for signals"""


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        self.logger = logging.getLogger('astro3d')

        # Setup the connections.
        self.signals = Signals()
        self.signals.open_file = Signal()
        self.signals.open_file.connect(self.open_file)
        self.signals.quit = Signal()
        self.signals.quit.connect(self.quit)

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)

        self.viewer = MainWindow(self.signals, self.logger)
        self.model = Model(self.signals)

        # Ok, let's start.
        self.viewer.show()

    def open_file(self, filepath):
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(filepath)
        self.signals.new_image(image)

    def quit(self, *args):
        self.logger.info("Attempting to shut down the application...")


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
