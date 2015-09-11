from __future__ import absolute_import
import six
import warnings

from ..util import signal_slot
from ..util.register_leaf_classes import (RegisterLeafClasses)


class Signals(signal_slot.Signals):
    '''The signal container that allows autoregistring of a
    set of predefined signals.
    '''
    def __init__(self, signal_class=None, logger=None):
        super(Signals, self).__init__()
        if signal_class is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for signal in signal_class:
                    self.add(signal, logger)


@six.add_metaclass(RegisterLeafClasses)
class Signal(signal_slot.Signal):
    '''Specview signals'''


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


# Data Manipulation
class OpenFile(Signal):
    """Open a new data file"""


class NewImage(Signal):
    """New Image is available"""


class ModelUpdate(Signal):
    """Update mesh model"""


# GUI events
class UpdateMesh(Signal):
    """Update Mesh view"""
