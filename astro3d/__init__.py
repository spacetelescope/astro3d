# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A package to create a 3D model from an astronomical image.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .core.model3d import *
