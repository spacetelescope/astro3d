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
            'texture_info': {},
            'textures': {},
            'texture_mappings': {},
            'catalog_textures': {}
        }
        params = self.params
        for section in params:
            params[section].update({
                p: config.get(section, p)
                for p in config.options(section)
            })

        # Acquire the region textures
        self.translate_texture = params['texture_mappings']
        self.texture_order = params['texture_info']['texture_order']
        self.textures = {
            name: eval(pars)
            for name, pars in params['textures'].items()
        }

        # Acquire the catalog textures
        self.catalog_textures = {
            name: eval(pars)
            for name, pars in params['catalog_textures'].items()
        }
