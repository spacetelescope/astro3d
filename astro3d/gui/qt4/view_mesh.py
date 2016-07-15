from math import sqrt

from pyqtgraph.opengl import (GLGridItem, GLMeshItem, GLViewWidget, MeshData)

from ...external.qt import QtCore
from .. import signaldb

# Shortcuts
Qt = QtCore.Qt


__all__ = ['ViewMesh']


class ViewMesh(GLViewWidget):
    """The 3D Mesh"""

    closed = QtCore.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(ViewMesh, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        grid = GLGridItem()
        grid.scale(2, 2, 1)
        self.addItem(grid)
        self.grid = grid
        self.mesh_item = None

    def process(self):
        """Display while new mesh is processing"""
        self.remove_mesh()

    def update_mesh(self, mesh, model3d):
        self.remove_mesh()
        scaling = mesh[:, 1:].max(axis=0)[0]
        distance = sqrt(scaling[0]**2 + scaling[1]**2)

        self.grid.setSize(x=scaling[0] * 2, y=scaling[1] * 2)

        mesh_data = MeshData(vertexes=mesh[:, 1:, :])
        mesh_item = GLMeshItem(meshdata=mesh_data,
                               shader='viewNormalColor',
                               glOptions='opaque',
                               smooth=True)

        self.setCameraPosition(distance=distance, azimuth=45, elevation=45)
        self.addItem(mesh_item)
        self.mesh_item = mesh_item

    def remove_mesh(self):
        """Remove the current mesh from display"""
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
        return QtCore.QSize(512, 512)

    def closeEvent(self, event):
        self.closed.emit(False)
        event.accept()
