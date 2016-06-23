"""GUI global configuration
"""
from __future__ import print_function

from ast import literal_eval
from ConfigParser import SafeConfigParser as SystemConfigParser
from os.path import (abspath, dirname, expanduser)

CONFIG_NAME = 'astro3d.cfg'

__all__ = ['config']


class Config(SystemConfigParser):

    def __init__(self):
        SystemConfigParser.__init__(self)
        builtin_config = _default_config()
        home_config = expanduser('/'.join(['~', CONFIG_NAME]))

        # List of configuration files.
        # The most specific should be last in the list.
        config_sources = [
            builtin_config,
            home_config,
            '/'.join(['.', CONFIG_NAME]),
        ]

        used = self.read(config_sources)
        if len(used) <= 1:
            self.save_config = home_config
        else:
            self.save_config = used[-1]

    def save(self):
        with open(self.save_config, 'wb') as save_config:
            self.write(save_config)

    def get(self, section, option):
        """Get option with guessing at value type"""
        value = SystemConfigParser.get(self, section, option)
        try:
            evalue = literal_eval(value)
        except SyntaxError:
            evalue = value
        return evalue


def _default_config():
    # Strip off the '/gui' part.
    exec_dir = dirname(abspath(__file__))[:-4]
    exec_dir = '/'.join([exec_dir, 'data'])
    return '/'.join([exec_dir, CONFIG_NAME])


config = Config()
