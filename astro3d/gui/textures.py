"""Get Texture info from user config"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..core.textures import (
    DotsTexture,
    LinesTexture,
    HexagonalGrid,
    StarTexture
)

__all__ = ['TextureConfig']


class TextureConfig(object):
    """User Texture configuration"""

    def __init__(self, config):
        self.params = {
            'textures': {},
            'texture_mappings': {}
        }
        params = self.params
        for section in params:
            params[section].update({
                p: config.get(section, p)
                for p in config.options(section)
            })

        # Map to the Model3d attributes
        self.translate_texture = params['texture_mappings']
        self.texture_order = params['texture_mappings']['texture_order']

        # Acquire the textures
        self.textures = {
            name: eval(pars)
            for name, pars in params['textures'].items()
        }
