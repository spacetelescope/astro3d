"""Data Model"""

__all__ = ['Model']

class Model(object):
    """Data model"""

    image = None

    def __init__(self, signals):
        self.signals = signals
