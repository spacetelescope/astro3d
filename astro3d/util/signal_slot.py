""" A signal/slot implementation

Original: See below
File:    signal.py
Author:  Thiago Marcos P. Santos
Author:  Christopher S. Case
Author:  David H. Bronke
Created: August 28, 2008
Updated: December 12, 2011
License: MIT

"""
from __future__ import print_function
import inspect
import warnings
from weakref import WeakSet, WeakKeyDictionary

from .logger import make_logger


__all__ = ['Signal',
           'Signals',
           'SignalsNotAClass']


class Signal(object):
    def __init__(self, logger=None, *args):
        """Setup a signal

        Parameters
        ----------
        logger: logging.Logger
            Logger to use. If None, one will be created.

        *args: (func, ...)
            Remaining arguments will be functions to connect
            to this signal.
        """
        self._functions = WeakSet()
        self._methods = WeakKeyDictionary()
        if logger is None:
            logger = make_logger('Signal')
        self.logger = logger

        for arg in args:
            self.connect(arg)

    def __call__(self, *args, **kwargs):
        # Call handler functions
        self.logger.debug(
            'Signal {}: Emitting with args:"{}", kwargs:"{}"'.format(
                self.__class__.__name__,
                args,
                kwargs
            )
        )
        to_be_removed = []
        for func in self._functions.copy():
            try:
                func(*args, **kwargs)
            except RuntimeError:
                Warning.warn(
                    'Signal {}: Signals func->RuntimeError: func "{}" will be removed.'.format(
                        self.__class__.__name_,
                        func
                    )
                )
                to_be_removed.append(func)

        for remove in to_be_removed:
            self._functions.discard(remove)

        # Call handler methods
        to_be_removed = []
        emitters = self._methods.copy()
        for obj, funcs in emitters.items():
            for func in funcs.copy():
                try:
                    func(obj, *args, **kwargs)
                except RuntimeError:
                    warnings.warn(
                        'Signal {}: Signals methods->RuntimeError, obj.func "{}.{}" will be removed'.format(
                            self.__class__.__new__,
                            obj,
                            func
                        )
                    )
                    to_be_removed.append((obj, func))

        for obj, func in to_be_removed:
            self._methods[obj].discard(func)

    def connect(self, slot):
        self.logger.debug(
            'Signal {}: Connecting slot:"{}"'.format(
                self.__class__.__name__,
                slot
            )
        )
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)

    def disconnect(self, slot):
        self.logger.debug(
            'Signal {}: Disconnecting slot:"{}"'.format(
                self.__class__.__name__,
                slot
            )
        )
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)

    def clear(self):
        self.logger.debug(
            'Signal {}: Clearing slots'.format(
                self.__class__.__name__
            )
        )
        self._functions.clear()
        self._methods.clear()


class SignalsErrorBase(Exception):
    '''Base Signals Error'''

    default_message = ''

    def __init__(self, *args):
        if len(args):
            super(SignalsErrorBase, self).__init__(*args)
        else:
            super(SignalsErrorBase, self).__init__(self.default_message)


class SignalsNotAClass(SignalsErrorBase):
    '''Must add a Signal Class'''
    default_message = 'Signal must be a class.'


class Signals(dict):
    '''Manage the signals.'''

    def __setitem__(self, key, value):
        if key not in self:
            super(Signals, self).__setitem__(key, value)
        else:
            warnings.warn('Signals: signal "{}" already exists.'.format(key))

    def __getattr__(self, key):
        for signal in self:
            if signal.__name__ == key:
                return self[signal]
        raise KeyError('{}'.format(key))

    def add(self, signal_class, *args, **kwargs):
        if inspect.isclass(signal_class):
            self.__setitem__(signal_class, signal_class(*args, **kwargs))
        else:
            raise SignalsNotAClass

# Sample usage:
if __name__ == '__main__':
    class Model(object):
        def __init__(self, value):
            self.__value = value
            self.changed = Signal()

        def set_value(self, value):
            self.__value = value
            self.changed()  # Emit signal

        def get_value(self):
            return self.__value

    class View(object):
        def __init__(self, model):
            self.model = model
            model.changed.connect(self.model_changed)

        def model_changed(self, *args, **kwargs):
            print('    args: "{}"'.format(args))
            print('    kwargs: "{}"'.format(kwargs))
            print("   New value:", self.model.get_value())

    print("Beginning Tests:")
    model = Model(10)
    view1 = View(model)
    view2 = View(model)
    view3 = View(model)

    print("Setting value to 20...")
    model.set_value(20)

    print("Deleting a view, and setting value to 30...")
    del view1
    model.set_value(30)

    print('Calling changed with arguments:')
    model.changed('an arg')
    model.changed('nother arg', help='me')

    print("Clearing all listeners, and setting value to 40...")
    model.changed.clear()
    model.set_value(40)

    print("Testing non-member function...")

    def bar():
        print("   Calling Non Class Function!")

    model.changed.connect(bar)
    model.set_value(50)
