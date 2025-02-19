# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A package to create a 3D model from an astronomical image.
"""

from .core.model3d import *  # noqa

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''
