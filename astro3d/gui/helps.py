"""Helper texts"""

from collections import defaultdict

__all__ = ['instructions']

INSTRUCTIONS_DEFAULT = (
    'Use File menu to read in the image and various region'
    ' definitions which define what to render in 3D.\n'
    '\n'
    'Use View-Mesh View to show/hide the 3D renderered view.\n'
    '\n'
    'Stages toggle the various processing stages.\n'
    '\n'
    'By default, processing occurs automatically.'
    ' This can be toggled on/off through Preferences->Auto Reprocess\n'
    '\n'
    'When the mouse is on the image, type "H" for a cheat sheat on'
    ' shortcut keys to change the display.\n'
    '\n'
    'To create new regions, right-click on "Regions".'
    ' To delete, hide, or edit existing regions,'
    ' click on or right-click on the desired region.\n'
    '\n'
    'When complete, use File->Save to save all regions and the'
    ' STL model files.'
)

INSTRUCTIONS_DRAW = (
    'Select the shape desired.\n'
    '\n'
    'To create/modify a pixel mask, select "paint".'
)

INSTRUCTIONS_EDIT = (
    'Click-drag to move the region.\n'
    '\n'
    'Click-drag any of the "control points" to'
    ' to change the size or shape of the region.\n'
    '\n'
    'Enter an angle and hit RETURN to change region rotation.\n'
    '\n'
    'Enter a scale factor and hit RETURN to change region size.'
)

INSTRUCTIONS_PAINT = (
    'Click-drag to either add to or erase a mask.\n'
    '\n'
    'Choose whether you are "Paint"ing or "Erase"ing.\n'
    '\n'
    'Brush size determines how much area is affected while painting.'
)

instructions = defaultdict(
    lambda: INSTRUCTIONS_DEFAULT,
    {
        'draw': INSTRUCTIONS_DRAW,
        'edit': INSTRUCTIONS_EDIT,
        'edit_select': INSTRUCTIONS_EDIT,
        'paint': INSTRUCTIONS_PAINT,
        'paint_edit': INSTRUCTIONS_PAINT,
    }
)
