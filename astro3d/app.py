"""astro3d UI Application
"""
from __future__ import absolute_import

import sys
import logging
from argparse import ArgumentParser

from .util.logger import make_logger
from .gui.qt.process import MeshThread
from .gui import (Controller, MainWindow, Model, signaldb)
from .gui.start_ui_app import start_ui_app

logger = make_logger(__name__)


class Application(Controller):
    """Start the astro3d Qt application

    Parameters
    ----------
    argv: list or None
        Argument list to pass to `argparse.ArgumentParser`.
        If None, use `sys.argv`.
    """

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        self.mesh_thread = None
        self.logger = logger

        self.parse_command_line(argv)

        self._create_signals()
        self.model = Model(
            logger=self.logger
        )

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)
        self.viewer = MainWindow(
            model=self.model,
            logger=self.logger
        )
        self.viewer.show()
        self.__class__.ui_app.setActiveWindow(self.viewer)
        self.viewer.raise_()
        self.viewer.activateWindow()

    def quit(self, *args, **kwargs):
        self.process_force_quit()
        self.logger.debug("Attempting to shut down the application...")

    def process(self, *args, **kwargs):
        """Do the processing."""
        self.logger.debug('Starting processing...')
        self.process_force_quit()
        signaldb.ProcessStart()
        self.model.process()

    def process_force_quit(self, *args, **kwargs):
        """Force quit a process"""
        signaldb.ProcessForceQuit()

    def process_finish(self, mesh, model3d):
        self.logger.debug('3D generation completed.')

    def parse_command_line(self, argv):
        """Parse command line arguments

        Parameters
        ----------
        argv: list or None
            Argument list to pass to `argparse.ArgumentParser`.
            If None, use `sys.argv`.
        """
        parser = ArgumentParser(
            description='astro3d: Create 3D models of astronomical images'
        )
        parser.add_argument(
            '-D', '--debug',
            help='Turn on debugging information',
            action='store_true'
        )
        args = parser.parse_args(argv)

        if args.debug:
            self.logger.setLevel(logging.DEBUG)

    def _create_signals(self):
        signaldb.logger = self.logger
        signaldb.Quit.connect(self.quit)
        signaldb.ModelUpdate.connect(self.process)
        signaldb.ProcessFinish.connect(self.process_finish)


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
