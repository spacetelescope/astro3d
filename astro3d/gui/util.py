"""General GUI utilities"""
from __future__ import print_function

from ginga.gw.Widgets import build_info

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

    captions = []
    captions_notbool = []
    for idx in store:
        value = store[idx]
        if isinstance(value, bool):
            captions.append((idx, 'checkbutton'))
        else:
            captions_notbool.append((
                idx, 'label',
                idx, 'entry'
            ))
    captions = captions + captions_notbool + extra
    container, widget_list = build_info(captions)
    for idx in store:
        value = store[idx]
        widget = widget_list[idx]
        widget.extdata.update({
            'store': store,
            'index': idx
        })
        if isinstance(value, bool):
            widget.set_state(value)
        else:
            widget.set_text(str(value))
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
        lambda: widget.get_text()
    ]
    idx = widget.extdata['index']
    store = widget.extdata['store']
    for get_value in get_value_funcs:
        try:
            store[idx] = get_value()
        except:
            continue
        else:
            break
    else:
        raise RuntimeError(
            'Cannot retrieve widget value, widget="{}"'.format(widget)
        )

    signaldb.ModelUpdate()
