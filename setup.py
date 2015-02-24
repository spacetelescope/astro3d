#!/usr/bin/env python
from distutils.core import setup

setup(
    name = 'astro3d',
    version = '0.3.0.dev',
    description = 'Hubble Image 3D Printing',
    author = 'STScI',
    author_email = 'help@stsci.edu',
    packages = ['astro3d', 'astro3d.utils', 'astro3d.gui'],
    scripts = ['scripts/astro3dgui'] #,
    #package_data = {'astro3d' : ['data/*.*']}
    )
