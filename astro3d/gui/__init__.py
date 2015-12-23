from __future__ import absolute_import

import ginga.toolkit as ui_toolkit

ui_toolkit.use('qt')

from .signals import Signals as _Signals
signaldb = _Signals()

from .model import *
from .viewer import *
from .controller import *
