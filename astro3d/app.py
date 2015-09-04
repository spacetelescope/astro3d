"""astro3d UI Application
"""
import sys
import logging

from ginga import AstroImage

from .util.signal_slot import Signal
from .gui import (Controller, MainWindow, Model)
from .gui.start_ui_app import start_ui_app

STD_FORMAT = '%(asctime)s | %(levelname)1.1s | %(filename)s:%(lineno)d (%(funcName)s) | %(message)s'


class Signals(object):
    """Container for signals"""


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        # Log it.
        logger = logging.getLogger('astro3d')
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(STD_FORMAT)
        stderrHdlr = logging.StreamHandler()
        stderrHdlr.setFormatter(fmt)
        logger.addHandler(stderrHdlr)
        self.logger = logger

        # Setup the connections.
        self.signals = Signals()
        self.signals.open_file = Signal(logger, self.open_file)
        self.signals.quit = Signal(logger, self.quit)
        self.signals.new_image = Signal(logger)
        self.signals.model_update = Signal(logger, self.process)
        self.signals.process_start = Signal(logger)
        self.signals.process_finish = Signal(logger, self.process_finish)

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)

        self.viewer = MainWindow(self.signals, self.logger)
        self.model = Model(self.signals, self.logger)

        # Ok, let's start.
        self.viewer.show()

    def open_file(self, filepath):
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(filepath)
        self.signals.new_image(image)

    def quit(self, *args):
        self.logger.debug("Attempting to shut down the application...")

    def process(self, *args, **kwargs):
        """Do the processing."""
        self.logger.info('Starting processing...')
        self.signals.process_start()

    def process_finish(self):
        self.logger.info('3D generation completed.')



def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
