"""Data Model"""

from ..core.model3d import Model3D


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
        m.double_sided = True

        m.make()

        m.write_stl('/Users/eisenham/Downloads/ngc3344_intensity_texture',
                    split_model=True, clobber=True)

        # All done
        self.signals.process_finish()
