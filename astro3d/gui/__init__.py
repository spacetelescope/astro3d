from __future__ import absolute_import

import ginga.toolkit as ui_toolkit

ui_toolkit.use('qt')

from .model import *
from .viewer import *
from .controller import *

from .signals import Signals as _Signals
signals = _Signals()
