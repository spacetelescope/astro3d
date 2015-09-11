"""astro3d UI Application
"""
from __future__ import absolute_import

import sys
import logging

from ginga import AstroImage

from .gui import (Controller, MainWindow, Model, signals as sig)
from .gui.start_ui_app import start_ui_app

STD_FORMAT = '%(asctime)s | %(levelname)1.1s | %(filename)s:%(lineno)d (%(funcName)s) | %(message)s'


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
        self.signals = sig.Signals(signal_class=sig.Signal, logger=logger)
        self.signals.Quit.connect(self.quit)
        self.signals.OpenFile.connect(self.open_file)
        self.signals.ModelUpdate.connect(self.process)
        self.signals.ProcessFinish.connect(self.process_finish)

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)

        self.viewer = MainWindow(self.signals, self.logger)
        self.model = Model(self.signals, self.logger)

        # Ok, let's start.
        self.viewer.show()

    def open_file(self, filepath):
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(filepath)
        self.signals.NewImage(image)

    def quit(self, *args):
        self.logger.debug("Attempting to shut down the application...")

    def process(self, *args, **kwargs):
        """Do the processing."""
        self.logger.info('Starting processing...')
        self.signals.ProcessStart()

    def process_finish(self, mesh):
        self.logger.info('3D generation completed.')
        self.signals.UpdateMesh(mesh)


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
