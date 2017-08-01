"""Interface between configuration store and widgets"""
from ast import literal_eval
from collections import MutableMapping

from ginga.gw.Widgets import (
    CheckBox,
    ComboBox,
    build_info
)

from . import signaldb

__all__ = ['StoreWidgets']


class StoreWidget(object):
    """Widget representing a configuration parameters

    Parameters
    ----------
    widget: UI widget object
        The UI widget

    config_store: object
        The configuration store from which and too which
        widget values will be retrieved/set_index

    key: str
        The index stro the store

    callback: function
        UI widget callback to use. If not defined, the generic
        `value_update` is used.
    """
    def __init__(self, widget, config_store, key, callback=None):
        self._widget = widget
        self._config_store = config_store
        self._key = key

        # Determine interface to/from widget and store
        if isinstance(widget, CheckBox):
            self._widget_get_value = widget.get_state
            self._widget_set_value = widget.set_state
            self._store_get_value = _store_get_value_direct
            self._store_set_value = _store_set_value_direct
            self._widget_set_value(self._store_get_value(config_store, key))
        elif isinstance(widget, ComboBox):
            combo_items = config_store[key][1]
            for item in combo_items:
                widget.append_text(item)
            self._widget_get_value = widget.get_index
            self._widget_set_value = widget.set_index
            self._store_get_value = _store_get_value_combo
            self._store_set_value = _store_set_value_combo
            self._widget_set_value(self._store_get_value(config_store, key))
        else:
            self._widget_get_value = widget.get_text
            self._widget_set_value = lambda value: widget.set_text(str(value))
            self._store_get_value = _store_get_value_direct
            self._store_set_value = _store_set_value_direct
            self._widget_set_value(
                self._store_get_value(config_store, key)
            )

        if callback is None:
            callback = value_update
        widget.add_callback(
            'activated',
            callback
        )

    @property
    def value(self):
        value = self._store_get_value(self._config_store, self._key)
        self._widget_set_value(value)
        return value

    @value.setter
    def value(self, value):
        self._widget_set_value(value)
        self._store_set_value(self._config_store, self._key, value)


class StoreWidgets(MutableMapping):
    """Interface between configuration store and widgets
    """

    def __init__(self, store=None, callback=None, extra=None):
        """Initialize StoreWidgets

        Parameters
        ----------
        store: dict-like
            The originating store

        callback: callable
            The callback to attach to the widgets.
            If None, a built in callback is used which
            takes the new value and places it back
            into the store.

        extra: (name, gui_type, ...)
            Extra widgets defined in the caption
            format. No special processing is
            done beyond the creation of the widgets.
        """
        super(StoreWidgets, self).__init__()

        # Initializations
        self._originating_store = store
        self.container = None
        self.widgets = None
        self.store_widgets = {}
        if store is not None:
            self.build_widgets(store, callback=callback, extra=extra)

    def build_widgets(self, store, callback=None, extra=None):
        """Build Ginga widgets from simple captions

        Parameters
        ----------
        store: dict-like
            The dictionary of items to create widgets from.

        callback: callable
            The callback to attach to the widgets.
            If None, a built in callback is used which
            takes the new value and places it back
            into the store.

        extra: (name, gui_type, ...)
            Extra widgets defined in the caption
            format. No special processing is
            done beyond the creation of the widgets.

        Returns
        -------
        widget: The Ginga widget
            The Ginga widget to use in GUI construction

        bunch: ginga.misc.Bunch.bunch
            The dict of all the individual widgets.
        """
        if extra is None:
            extra = []

        # Build the widgets
        captions = []
        captions_notbool = []
        for key in store:
            value = store[key]
            if isinstance(value, bool):
                captions.append((key, 'checkbutton'))
            else:
                if isinstance(value, list):
                    captions.extend([
                        (key, 'label'),
                        (key, 'combobox')
                    ])
                else:
                    captions_notbool.append((
                        key, 'label',
                        key, 'entryset'
                    ))
        captions = captions + captions_notbool + extra
        self.container, self.widgets = build_info(captions)

        # Define widget/store api
        for key in store:
            widget = self.widgets[key]
            store_widget = StoreWidget(widget, store, key)
            widget.extdata.update({'store_widget': store_widget})
            self.store_widgets[key] = widget

    # ABC required methods
    def __copy__(self):
        """Produce a shallow copy"""
        new_store = type(self)()
        new_store._originating_store = self._originating_store
        new_store.container = self.container
        new_store.widgets = self.widgets.copy()
        new_store.store_widget = self.store_widgets.copy()
        return new_store

    def __getitem__(self, key):
        """Get value of widget

        Parameters
        ----------
        key: str
            Index into the list

        Returns
        -------
        obj: object
            Object at index. This is the value
            of the item
        """
        widget = self.store_widgets[key]
        return widget.extdata.store_widget.value

    def __setitem__(self, key, value):
        """Set value for widget

        Parameters
        ----------
        key: str
            Index of item to set

        value: object
            The value to set the item to
        """
        widget = self.store_widgets[key]
        widget.extdata.store_widget.value = value

    def __delitem__(self, key):
        """Delete specified item
        """
        del self.store_widgets[key]
        del self.widgets[key]

    def __len__(self):
        """Return number of widgets specified"""
        return len(self.store_widgets)

    def __iter__(self):
        for key in self.store_widgets:
            yield key


# #####################
# Basic update callback
# #####################
def value_update(widget, *args, **kwargs):
    """Update the internal store from the widget

    Parameters
    ----------
    widget: GUI widget
        The widget that initiated the callback.

    args, kwargs: 0 or more objects
        Depending on the widget, there may be extra
        information generated.
    """
    get_value_funcs = [
        lambda: args[0],
        lambda: widget.get_index(),
        lambda: literal_eval(widget.get_text()),
        lambda: widget.get_text()
    ]
    for get_value in get_value_funcs:
        try:
            widget.extdata.store_widget.value = get_value()
        except:
            continue
        else:
            break
    else:
        raise RuntimeError(
            'Cannot retrieve widget value, widget="{}"'.format(widget)
        )

    signaldb.ModelUpdate()


# ######################
# Store access utilities
# ######################
def _store_set_value_direct(store, key, value):
    """Save the value to the store

    Parameters
    ----------
    store: dict-like
        The store to place the value

    key: str
        The index in the store to access

    value: object
        Value to place into the store
    """
    store[key] = value


def _store_set_value_combo(store, key, value):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to place the value

    key: str
        The index in the store to access

    value: object
        Value to place into the store
    """
    store[key][0] = value


def _store_get_value_direct(store, key):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to retreive the value from

    key: str
        The index in the store to access

    Returns
    -------
    value: object
        Value from store
    """
    return store[key]


def _store_get_value_combo(store, key):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to retreive the value from

    key: str
        The index in the store to access

    Returns
    -------
    value: object
        Value from store
    """
    return store[key][0]
