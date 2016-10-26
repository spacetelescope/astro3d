from __future__ import absolute_import

# Force the loading of Qt
# First let the qtpy compatibility library
# force the appropriate version of Qt.
# Then force ginga to use that version.
try:
    from qtpy import QtCore
except ImportError:
    raise RuntimeError('Cannot import Qt toolkit')
try:
    from ginga.qtw import QtHelp
except ImportError:
    raise RuntimeError('Cannot import Qt toolkit')

from .signals import Signals as _Signals
signaldb = _Signals()

from .model import *
from .viewer import *
from .controller import *
