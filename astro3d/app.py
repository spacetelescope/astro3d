"""astro3d UI Application
"""
from __future__ import absolute_import

import sys
import logging

from .util.logger import make_logger
from .util.process import MeshThread
from .gui import (Controller, MainWindow, Model, signals as sig)
from .gui.start_ui_app import start_ui_app


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        self.mesh_thread = None
        self.logger = make_logger(name='astro3d', level=logging.WARNING)
        self._create_signals()
        self.model = Model(
            logger=self.logger,
            signals=self.signals
        )

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)
        self.viewer = MainWindow(
            model=self.model,
            signals=self.signals,
            logger=self.logger
        )
        self.viewer.show()

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

    def process_finish(self, mesh):
        self.logger.debug('3D generation completed.')
        if mesh is not None:
            self.signals.UpdateMesh(mesh)

    def _create_signals(self):
        self.signals = sig.Signals(
            signal_class=sig.Signal,
            logger=self.logger
        )
        self.signals.Quit.connect(self.quit)
        self.signals.ModelUpdate.connect(self.process)
        self.signals.ProcessFinish.connect(self.process_finish)


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
