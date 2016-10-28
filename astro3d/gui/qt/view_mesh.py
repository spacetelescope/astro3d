"""View the 3D Model"""

import numpy as np
from qtpy import (QtCore, QtWidgets)
from qtpy.QtCore import Signal as pyqtSignal
from vispy import scene

from ...util.logger import make_logger


__all__ = ['ViewMesh']


class ViewMesh(QtWidgets.QWidget):
    """View the 3D Mesh"""

    closed = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        logger = kwargs.pop('logger', None)
        if logger is None:
            logger = make_logger('astro3d Layer Manager')
        self.logger = logger
        super(ViewMesh, self).__init__(*args, **kwargs)
        self.setLayout(QtWidgets.QVBoxLayout())

    # ---------------------------------------------
    # Need to implement the following at some point
    # ---------------------------------------------
    def process(self):
        """Display while new mesh is processing"""
        self.remove_mesh()

    def update_mesh(self, mesh, model3d):
        self.logger.debug('Called.')

        # Get the vertices and scale to unit level.
        mesh = mesh[:, 1:, :]
        scaled = ((mesh - mesh.min()) / mesh.max()) - 1.0

        # Color the mesh.
        nf = scaled.shape[0]
        fcolor = np.ones((nf, 3, 4), dtype=np.float32)

        # Show it.
        _ = scene.visuals.Mesh(
            parent=self.viewbox.scene,
            face_colors=fcolor,
            vertices=scaled,
            shading='flat',
        )

    def remove_mesh(self):
        """Remove the current mesh from display"""
        self.viewbox = None

    def toggle_view(self):
        """Toggle this view"""
        sender = self.sender()
        self.setVisible(sender.isChecked())

    def sizeHint(self):
        return QtCore.QSize(800, 800)

    def closeEvent(self, event):
        self.closed.emit(False)
        event.accept()

    @property
    def viewbox(self):
        try:
            viewbox = self._viewbox
        except AttributeError:
            viewbox = None

        if viewbox:
            return viewbox

        # Create the scene to view in.
        viewbox = scene.widgets.ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(viewbox)
        viewbox.camera = scene.TurntableCamera()
        self._viewbox = viewbox

        return viewbox

    @viewbox.setter
    def viewbox(self, viewbox):
        if viewbox:
            viewbox.add_parent(self.canvas.scene)
            self._viewbox = viewbox
        else:
            self.canvas = None
            self._viewbox = None

    @property
    def canvas(self):
        try:
            canvas = self._canvas
        except AttributeError:
            canvas = None

        if canvas:
            return canvas

        # Create the canvas.
        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 800, 800
        self.canvas = canvas

        return canvas

    @canvas.setter
    def canvas(self, canvas):
        if canvas:
            self.layout().addWidget(canvas.native)
            self._canvas = canvas
        else:
            try:
                self.layout().removeWidget(self._canvas.native)
            except (AttributeError, ValueError):
                pass
            finally:
                self._canvas = None
