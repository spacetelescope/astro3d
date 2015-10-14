"""Qt Utilities"""

from functools import (partial, wraps)

from ...external.qt.QtCore import QTimer


class DecorMethod(object):
    def __init__(self, decor, instance):
        self.decor = decor
        self.instance = instance

    def __call__(self, *args, **kw):
        return self.decor(self.instance, *args, **kw)

    def __getattr__(self, name):
        return getattr(self.decor, name)

    def __repr__(self):
        return '<bound method {} of {}>'.format(self.decor, type(self))


def event_deferred(func):
    timer = QTimer()
    timer.setSingleShot(True)
    f_args = ()
    f_kwargs = {}
    f_func = func

    def _exec():
        global f_args, f_kwargs
        f_func(*f_args, **f_kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        global f_args, f_kwargs
        timer.stop
        f_args = args
        f_kwargs = kwargs
        timer.start(0)

    timer.timeout.connect(_exec)

    return wrapper


class EventDeferred(QTimer):
    """Defer execution until no events"""
    def __init__(self, func):
        super(EventDeferred, self).__init__()

        self.setSingleShot(True)
        self.timeout.connect(self._exec)
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return DecorMethod(self, instance)

    def __call__(self, instance, *args, **kwargs):
        self.stop()
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.start(0)

    def _exec(self):
        partial(self.func, self.instance, *self.args, **self.kwargs)()
