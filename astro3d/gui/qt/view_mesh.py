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

        # Create the 3D canvas
        canvas = scene.SceneCanvas(keys='interactive')
        self.canvas = canvas
        canvas.size = 800, 800
        canvas.show()

        # Create the scene to view in.
        viewbox = scene.widgets.ViewBox(parent=canvas.scene)
        self.viewbox = viewbox
        viewbox.camera = scene.TurntableCamera()
        canvas.central_widget.add_widget(viewbox)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.canvas.native)

    # ---------------------------------------------
    # Need to implement the following at some point
    # ---------------------------------------------
    def process(self):
        """Display while new mesh is processing"""
        self.remove_mesh()

    def update_mesh(self, mesh, model3d):
        self.logger.debug('Called.')

        mesh = mesh[:, 1:, :]
        scaled = ((mesh - mesh.min()) / mesh.max()) - 1.0

        # Mesh with pre-indexed vertices, per-face color
        # Because vertices are pre-indexed, we get a different color
        # every time a vertex is visited, resulting in sharp color
        # differences between edges.
        nf = scaled.shape[0]
        fcolor = np.ones((nf, 3, 4), dtype=np.float32)

        # Show it.
        self.mesh_visual = scene.visuals.Mesh(
            parent=self.viewbox.scene,
            face_colors=fcolor,
            vertices=scaled,
            shading='flat',
        )

    def remove_mesh(self):
        """Remove the current mesh from display"""
        return
        try:
            self.removeItem(self.mesh_item)
        except ValueError:
            pass
        self.mesh_item = None

    def toggle_view(self):
        """Toggle this view"""
        sender = self.sender()
        self.setVisible(sender.isChecked())

    def sizeHint(self):
        return QtCore.QSize(800, 800)

    def closeEvent(self, event):
        self.closed.emit(False)
        event.accept()
