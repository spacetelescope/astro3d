"""Data Model"""

from __future__ import absolute_import, print_function

from attrdict import AttrDict

from numpy import concatenate

from ..core.model3d import Model3D
from ..core.meshes import (get_triangles, reflect_mesh)
from ..util.logger import make_logger


__all__ = ['Model']


class Model(object):
    """Data model"""

    image = None

    def __init__(self, logger=None):
        if logger is None:
            logger = make_logger('astro3d model')
        self.logger = logger

        self.stages = AttrDict({
            'intensity': True,
            'textures': False,
            'spiral_galaxy': False,
            'double_sided': False
        })

        self.maskpathlist = []

    def set_image(self, image):
        """Set the image"""
        self.image = image

    def read_maskpathlist(self, pathlist):
        """Read a list of mask files"""
        self.maskpathlist = pathlist

    def process(self):
        """Create the 3D model."""
        self.logger.debug('Starting processing...')

        # Setup steps in the thread. Between each step,
        # check to see if stopped.
        m = Model3D(self.image)

        for path in self.maskpathlist:
            m.read_mask(path)

        m.read_stellar_table('features/ngc3344_clusters.txt', 'star_clusters')

        m.has_textures = self.stages.textures
        m.has_intensity = self.stages.intensity
        m.spiral_galaxy = self.stages.spiral_galaxy
        m.double_sided = self.stages.double_sided

        m.make()

        triset = get_triangles(m.data)
        if m.double_sided:
            triset = concatenate((triset, reflect_mesh(triset)))
        return triset
