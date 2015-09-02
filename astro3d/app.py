"""astro3d UI Application
"""
import sys

from .gui import (Controller, MainWindow)
from .gui.start_ui_app import start_ui_app


class Application(Controller):

    # The UI event loop.
    ui_app = None

    def __init__(self, argv=None):

        if self.__class__.ui_app is None:
            self.__class__.ui_app = start_ui_app(argv)

        self.viewer = MainWindow()
        self.model = None

        self.viewer.show()


def main():
    app = Application()
    sys.exit(app.ui_app.exec_())


if __name__ == '__main__':
    main()
