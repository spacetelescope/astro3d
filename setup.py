#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


# command-line scripts
entry_points = {}
entry_points['console_scripts'] = [
    'astro3d = astro3d.gui.core:main',
    'convert_textures = astro3d.scripts.convert_textures:main',
]

package_name = 'astro3d'
version = '0.4.dev0'

setup(name=package_name,
      version=version,
      description='Create a 3D model from an astronomical image',
      author='STScI',
      author_email='help@stsci.edu',
      packages=['astro3d', 'astro3d.gui'],
      entry_points=entry_points,
      )
