from math import sqrt

from pyqtgraph.opengl import (GLGridItem, GLMeshItem, GLViewWidget, MeshData)


__all__ = ['ViewMesh']


class ViewMesh(GLViewWidget):
    """The 3D Mesh"""

    def __init__(self, *args, **kwargs):
        super(ViewMesh, self).__init__(*args, **kwargs)

        grid = GLGridItem()
        grid.scale(2, 2, 1)
        self.addItem(grid)
        self.grid = grid
        self.mesh_item = None

    def update_mesh(self, mesh):
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)

        scaling = mesh[:, 1:].max(axis=0)[0]
        distance = sqrt(scaling[0]**2 + scaling[1]**2)

        self.grid.setSize(x=scaling[0] * 2, y = scaling[1] * 2)

        mesh_data = MeshData(vertexes=mesh[:, 1:, :])
        mesh_item = GLMeshItem(meshdata=mesh_data,
                               shader='viewNormalColor',
                               glOptions='opaque',
                               smooth=True)
        mesh_item.translate(-1. * scaling[0] / 2., -1. * scaling[1] / 2., 0)

        self.setCameraPosition(distance=distance, azimuth=45, elevation=45)
        self.addItem(mesh_item)
        self.mesh_item = mesh_item
