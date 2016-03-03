"""astro3d UI Application
"""
from __future__ import absolute_import

import sys
import logging
from argparse import ArgumentParser

from .util.logger import make_logger
from .util.process import MeshThread
from .gui import (Controller, MainWindow, Model, signaldb)
from .gui.start_ui_app import start_ui_app


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        self.mesh_thread = None
        self.logger = make_logger(name='astro3d', level=logging.DEBUG)
        self._create_signals()
        self.model = Model(
            logger=self.logger
        )

        self._argparse(argv)

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
        self.logger.debug("Attempting to shut down the application...")

    def process(self, *args, **kwargs):
        """Do the processing."""
        self.logger.debug('Starting processing...')
        self.process_force_quit()
        self.mesh_thread = MeshThread(args=(self,))
        self.mesh_thread.start()

    def process_force_quit(self, *args, **kwargs):
        """Force quit a process"""
        t = self.mesh_thread
        if t is not None and t.is_alive():
            t.stop()
            t.join()

    def process_finish(self, mesh, model3d):
        self.logger.debug('3D generation completed.')
        if mesh is not None:
            self.model.model3d = model3d
            self.model.mesh = mesh
            signaldb.UpdateMesh(mesh)

    def _argparse(self, argv):
        parser = ArgumentParser()
        parser.add_argument(
            '-D', '--disable_autoprocessing',
            help='Disable automatic 3D generation',
            action='store_true'
        )
        args = parser.parse_args(argv)
        signaldb.ModelUpdate.set_enabled(not args.disable_autoprocessing)

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
