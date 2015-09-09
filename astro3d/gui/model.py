"""Data Model"""
from numpy import concatenate

from ..core.model3d import Model3D
from ..core.meshes import (get_triangles, reflect_mesh)


__all__ = ['Model']


class Model(object):
    """Data model"""

    image = None

    def __init__(self, signals, logger):
        self.logger = logger

        self.signals = signals
        self.signals.new_image.connect(self.set_image)
        self.signals.process_start.connect(self.process)

    def set_image(self, image):
        """Set the image

        Parameters
        ----------
        image: `ginga.AstroImage.AstroImage`
            The image to make the model from.
        """
        self.image = image
        self.signals.model_update()

    def process(self):
        """Create the 3D model."""
        self.logger.debug('Starting processing...')

        m = Model3D(self.image.get_data())

        m.read_all_masks('features/*.fits')

        m.read_stellar_table('features/ngc3344_clusters.txt', 'star_clusters')

        m.has_textures = True
        m.has_intensity = True
        m.spiral_galaxy = True
        m.double_sided = False

        m.make()

        triset = get_triangles(m.data)
        if m.double_sided:
            triset = concatenate((triset, reflect_mesh(triset)))
        self.signals.process_finish(triset)
