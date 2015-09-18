"""Data Model"""

from __future__ import absolute_import, print_function

from collections import namedtuple

from numpy import concatenate

from ..core.model3d import Model3D
from ..core.meshes import (get_triangles, reflect_mesh)
from ..util.logger import make_logger


__all__ = ['Model']


Task = namedtuple('Task', 'func, args, result')


class Model(object):
    """Data model"""

    image = None

    def __init__(self, logger=None):
        if logger is None:
            logger = make_logger('astro3d model')
        self.logger = logger

        self.has_textures = True
        self.has_intensity = True
        self.spiral_galaxy = True
        self.double_sided = True

    def stagechange(self, stage, state):
        self.logger.debug('stagechange: stage="{}" state="{}"'.format(stage, state))

        if stage == 'textures':
            self.has_textures = state
        elif stage == 'intensity':
            self.has_intensity = state
        elif stage == 'spiral_galaxy':
            self.spiral_galaxy = state
        elif stage == 'double_sided':
            self.double_sided = state

    def set_image(self, image):
        """Set the image"""
        self.image = image

    def process(self):
        """Create the 3D model."""
        self.logger.debug('Starting processing...')

        # Setup steps in the thread. Between each step,
        # check to see if stopped.
        m = Model3D(self.image)

        m.read_all_masks('features/*.fits')

        m.read_stellar_table('features/ngc3344_clusters.txt', 'star_clusters')

        m.has_textures = self.has_textures
        m.has_intensity = self.has_intensity
        m.spiral_galaxy = self.spiral_galaxy
        m.double_sided = self.double_sided

        m.make()

        triset = get_triangles(m.data)
        if m.double_sided:
            triset = concatenate((triset, reflect_mesh(triset)))
        return triset
