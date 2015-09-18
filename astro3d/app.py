"""astro3d UI Application
"""
from __future__ import absolute_import

import sys
import logging

from ginga import AstroImage

from .util.logger import make_logger
from .gui import (Controller, MainWindow, Model, signals as sig)
from .gui.start_ui_app import start_ui_app


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        # Log it.
        self.logger = make_logger(name='astro3d', level=logging.DEBUG)

        # Setup the connections.
        self.signals = sig.Signals(
            signal_class=sig.Signal,
            logger=self.logger
        )
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

    def quit(self, *args, **kwargs):
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
