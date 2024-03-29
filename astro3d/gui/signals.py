from __future__ import absolute_import
import six
import warnings

from ..util import signal_slot
from ..util.register_leaf_classes import (RegisterLeafClasses)

@six.add_metaclass(RegisterLeafClasses)
class Signal(signal_slot.Signal):
    """astro3d signals"""


class Signals(signal_slot.Signals):
    '''The signal container that allows autoregistring of a
    set of predefined signals.
    '''
    def __init__(self, signal_class=Signal):
        super(Signals, self).__init__()
        if signal_class is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for signal in signal_class:
                    self.add(signal)


# Specific Signal Definitions
# Signals can proliferate without bound.
# So, define all possible signals here.

# General application events
class Quit(Signal):
    """Quit the application"""


class ProcessStart(Signal):
    """Process new mesh"""


class ProcessFinish(Signal):
    """Mesh processing complete"""


class ProcessForceQuit(Signal):
    """Force quit mesh processing"""


class ProcessFail(Signal):
    """Process has failed, expect no result"""

# Data Manipulation
class CreateGasSpiralMasks(Signal):
    """Auto-create gas and spiral masks"""


class ModelUpdate(Signal):
    """Update mesh model"""


class NewImage(Signal):
    """New Image is available"""


class OpenFile(Signal):
    """Open a new data file"""


class StageChange(Signal):
    """A stage has changed state"""


# GUI events
class CatalogFromFile(Signal):
    """Initiate a catalog from file read"""


class LayerSelected(Signal):
    """Layers have been selected/deselected"""


class NewRegion(Signal):
    """New region being created"""
