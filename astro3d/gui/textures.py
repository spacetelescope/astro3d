"""Get Texture info from user config"""

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

        self.texture_order = params['textures']['texture_order']
        self.translate_texture = params['texture_mappings']
