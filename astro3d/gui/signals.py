from __future__ import absolute_import
import six
import warnings

from ..util import signal_slot
from ..util.logger import make_logger
from ..util.register_leaf_classes import (RegisterLeafClasses)


@six.add_metaclass(RegisterLeafClasses)
class Signal(signal_slot.Signal):
    """Specview signals"""


class Signals(signal_slot.Signals):
    '''The signal container that allows autoregistring of a
    set of predefined signals.
    '''
    def __init__(self, signal_class=Signal, logger=None):
        super(Signals, self).__init__()
        if logger is None:
            logger = make_logger('Signals')
        self.logger = logger
        if signal_class is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for signal in signal_class:
                    self.add(signal, logger)


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


# Data Manipulation
class OpenFile(Signal):
    """Open a new data file"""


class NewImage(Signal):
    """New Image is available"""


class ModelUpdate(Signal):
    """Update mesh model"""


class StageChange(Signal):
    """A stage has changed state"""


# GUI events
class UpdateMesh(Signal):
    """Update Mesh view"""