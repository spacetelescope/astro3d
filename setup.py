#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# command-line scripts
#entry_points = {}
#entry_points['console_scripts'] = [
#    'convert_textures = astro3d.utils.scripts.convert_textures:main'
#]

setup(
    name = 'astro3d',
    version = '0.3.0.dev0',
    description = 'Hubble Image 3D Printing',
    author = 'STScI',
    author_email = 'help@stsci.edu',
    packages = ['astro3d', 'astro3d.utils', 'astro3d.gui'],
    scripts = ['scripts/astro3dgui', 'scripts/convert_textures'],
    #entry_points=entry_points
    )
