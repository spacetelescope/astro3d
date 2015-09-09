from math import sqrt

from pyqtgraph.opengl import (GLMeshItem, GLViewWidget, MeshData)


__all__ = ['ViewMesh']


class ViewMesh(GLViewWidget):
    """The 3D Mesh"""

    def update_mesh(self, mesh):
        scaling = mesh[:, 1:].max(axis=0)[0]
        distance = sqrt(scaling[0]**2 + scaling[1]**2)
        self.setCameraPosition(distance=distance, azimuth=45, elevation=45)

        md = MeshData(vertexes=mesh[:, 1:, :])
        mi = GLMeshItem(meshdata=md,
                        shader='viewNormalColor',
                        glOptions='opaque',
                        smooth=True)
        mi.translate(-1. * scaling[0] / 2., -1. * scaling[1] / 2., 0)
        self.addItem(mi)
