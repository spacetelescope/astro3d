.. _astro3d-doc-index:

Hubble Image 3D Printing ``astro3d``
====================================

This package contains ad-hoc code to produce textured 3-dimensional images for
a 3D printer.

General process is to generate an image to be rendered as surface, and then
turn it into an STL file for a MakerBot Replicator 2 3D printer.


Dependencies
------------

* `Anaconda <http://continuum.io/downloads>`_
    * astropy
    * matplotlib
    * numpy
    * numpydoc (for documentation build only)
    * PIL
    * PyQt4
    * scipy
    * sphinx (for documentation build only)
* `imageutils <https://github.com/astropy/imageutils>`_
* `photutils <https://github.com/astropy/photutils>`_
* `qimage2ndarray <https://github.com/spacetelescope/qimage2ndarray>`_


Installation
------------

To install::

    > python setup.py install [--prefix=/my/install/dir]


Running GUI
-----------

To run GUI::

    > astro3dgui


Using ``astro3d``
-----------------

Contents:

.. toctree::
   :maxdepth: 2

   api


Version
-------

.. autodata:: astro3d.gui.astroVisual.__version__
.. autodata:: astro3d.gui.astroVisual.__vdate__


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
