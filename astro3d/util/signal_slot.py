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
from collections import namedtuple
import inspect
import warnings

from .logger import make_logger


__all__ = ['Signal',
           'Signals',
           'SignalsNotAClass']

Slot = namedtuple('Slot', ['func', 'single_shot'])

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
        self._slots = set()
        self._methods = dict()
        self._enabled = True
        self._states = []
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

        if not self.enabled:
            self.logger.debug(
                'Signal {}: Disabled, exiting...'.format(
                    self.__class__.__name__
                )
            )
            return

        # No recursive signalling
        self.set_enabled(False)

        # Call the slots.
        try:
            to_be_removed = []
            for slot in self._slots.copy():
                try:
                    slot.func(*args, **kwargs)
                except RuntimeError:
                    Warning.warn(
                        'Signal {}: Signals func->RuntimeError: func "{}" will be removed.'.format(
                            self.__class__.__name_,
                            slot.func
                        )
                    )
                    to_be_removed.append(slot)
                finally:
                    if slot.single_shot:
                        to_be_removed.append(slot)

            for remove in to_be_removed:
                self._slots.discard(remove)

            # Call handler methods
            to_be_removed = []
            emitters = self._methods.copy()
            for obj, slots in emitters.items():
                for slot in slots.copy():
                    try:
                        slot.func(obj, *args, **kwargs)
                    except RuntimeError:
                        warnings.warn(
                            'Signal {}: Signals methods->RuntimeError, obj.func "{}.{}" will be removed'.format(
                                self.__class__.__new__,
                                obj,
                                slot.func
                            )
                        )
                        to_be_removed.append((obj, slot))
                    finally:
                        if slot.single_shot:
                            to_be_removed.append((obj, slot))

            for obj, slot in to_be_removed:
                self._methods[obj].discard(slot)
        finally:
            self.set_enabled(True)

    @property
    def enabled(self):
        return self._enabled

    def set_enabled(self, state, push=False):
        """Set whether signal is active or not

        Parameters
        ----------
        state: boolean
            New state of signal

        push: boolean
            If True, current state is saved.
        """
        if push:
            self._states.append(self._enabled)
        self._enabled = state

    def reset_enabled(self):
            self._enabled = self._states.pop()

    def connect(self, func, single_shot=False):
        """Connect a function to the signal
        Parameters
        ----------
        func: function or method
            The function/method to call when the signal is activated

        single_shot: bool
            If True, the function/method is removed after being called.
        """
        self.logger.debug(
            'Signal {}: Connecting function:"{}"'.format(
                self.__class__.__name__,
                func
            )
        )
        if inspect.ismethod(func):
            if func.__self__ not in self._methods:
                self._methods[func.__self__] = set()

            slot = Slot(
                func=func.__func__,
                single_shot=single_shot
            )
            self._methods[func.__self__].add(slot)

        else:
            slot = Slot(
                func=func,
                single_shot=single_shot
            )
            self._slots.add(slot)

    def disconnect(self, func):
        self.logger.debug(
            'Signal {}: Disconnecting func:"{}"'.format(
                self.__class__.__name__,
                func
            )
        )
        if inspect.ismethod(func):
            if func.__self__ in self._methods:
                slots = [
                    slot
                    for slot in self._methods[func.__self__]
                    if slot.func == func
                ]
                try:
                    self._methods[func.__self__].remove(slots[0])
                except IndexError:
                    pass
        else:
            slots = [
                slot
                for slot in self._slots
                if slot.func == func
            ]
            try:
                self._functions.remove(slots[0])
            except IndexError:
                pass

    def clear(self):
        self.logger.debug(
            'Signal {}: Clearing slots'.format(
                self.__class__.__name__
            )
        )
        self._slots.clear()
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
            model.changed.connect(self.single_shot, single_shot=True)

        def model_changed(self, *args, **kwargs):
            print('    args: "{}"'.format(args))
            print('    kwargs: "{}"'.format(kwargs))
            print("   New value:", self.model.get_value())

        def single_shot(self, *args, **kwargs):
            print('    Single shot')

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

    print('Setting single_shot')

    def bar_once():
        print('    bar_once')

    model.changed.connect(bar_once, single_shot=True)
    model.set_value(60)
    model.set_value(70)
