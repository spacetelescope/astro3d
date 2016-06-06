"""GUI global configuration
"""
from __future__ import print_function

from ConfigParser import SafeConfigParser as SystemConfigParser
from os.path import (abspath, dirname, exists, expanduser)

CONFIG_NAME = 'astro3d.cfg'

__all__ = ['config']


class Config(SystemConfigParser):

    def __init__(self):
        SystemConfigParser.__init__(self)
        builtin_config = _default_config()
        home_config = expanduser('/'.join(['~', CONFIG_NAME]))
        config_sources = [
            ('/'.join(['.', CONFIG_NAME]), True),
            (home_config, True),
            (builtin_config, False),
        ]

        config_file, savable = choose_config(config_sources)
        self.read(config_file)
        self.save_file = config_file
        if not savable:
            self.save_file = home_config

    def save(self):
        with open(self.save_file, 'wb') as save_file:
            self.write(save_file)


def _default_config():
    # Strip off the '/gui' part.
    exec_dir = dirname(abspath(__file__))[:-4]
    exec_dir = '/'.join([exec_dir, 'data'])
    return '/'.join([exec_dir, CONFIG_NAME])


def choose_config(candidates):
    for candidate, savable in candidates:
        if exists(candidate):
            return candidate, savable
    raise IOError('No configuration files found')


config = Config()
