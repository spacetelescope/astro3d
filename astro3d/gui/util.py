"""General GUI utilities"""
from __future__ import print_function

from ast import literal_eval
from functools import partial

from ginga.gw.Widgets import (
    CheckBox,
    ComboBox,
    build_info
)

from . import signaldb


__all__ = ['build_widgets']


def build_widgets(store, callback=None, extra=None):
    """Build Ginga widgets from simple captions

    Based off of ginga.gw.Widgets.build_info,
    adds to extdata the following:
        'store': The given dict
        'index': The index into the store

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
    if callback is None:
        callback = value_update
    if extra is None:
        extra = []

    # Build the widgets
    captions = []
    captions_notbool = []
    for idx in store:
        value = store[idx]
        if isinstance(value, bool):
            captions.append((idx, 'checkbutton'))
        else:
            if isinstance(value, list):
                captions.extend([
                    (idx, 'label'),
                    (idx, 'combobox')
                ])
            else:
                captions_notbool.append((
                    idx, 'label',
                    idx, 'entryset'
                ))
    captions = captions + captions_notbool + extra
    container, widget_list = build_info(captions)

    # Define widget/store api
    for idx in store:
        widget = widget_list[idx]
        widget.extdata.update({
            'store': store,
            'index': idx
        })
        if isinstance(widget, CheckBox):
            widget.update_to_store = partial(
                _update_to_store_direct, store, idx
            )
            widget.update_from_store = partial(
                _update_from_store_direct, store, idx
            )
            widget.set_state(widget.update_from_store())
        elif isinstance(widget, ComboBox):
            for item in store[idx][1]:
                widget.append_text(item)
            widget.update_to_store = partial(
                _update_to_store_combo, store, idx
            )
            widget.update_from_store = partial(
                _update_from_store_combo, store, idx
            )
            widget.set_index(widget.update_from_store())
        else:
            widget.update_to_store = partial(
                _update_to_store_direct, store, idx
            )
            widget.update_from_store = partial(
                _update_from_store_direct, store, idx
            )
            widget.set_text(str(widget.update_from_store()))
        widget.add_callback(
            'activated',
            callback
        )

    return container, widget_list


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
            print('value_update: trying get_value {}'.format(get_value))
            widget.update_to_store(get_value())
        except:
            continue
        else:
            print('\tthat one worked')
            break
    else:
        raise RuntimeError(
            'Cannot retrieve widget value, widget="{}"'.format(widget)
        )

    signaldb.ModelUpdate()


def _update_to_store_direct(store, idx, value):
    """Save the value to the store

    Parameters
    ----------
    store: dict-like
        The store to place the value

    idx: int
        The index in the store to access

    value: object
        Value to place into the store
    """
    store[idx] = value


def _update_to_store_combo(store, idx, value):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to place the value

    idx: int
        The index in the store to access

    value: object
        Value to place into the store
    """
    store[idx][0] = value


def _update_from_store_direct(store, idx):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to retreive the value from

    idx: int
        The index in the store to access

    Returns
    -------
    value: object
        Value from store
    """
    return store[idx]


def _update_from_store_combo(store, idx):
    """Save the value to the store for combo values

    Parameters
    ----------
    store: dict-like
        The store to retreive the value from

    idx: int
        The index in the store to access

    Returns
    -------
    value: object
        Value from store
    """
    return store[idx][0]
