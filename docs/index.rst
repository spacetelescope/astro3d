Astro3d
=======

Astro3d provides a GUI and engine to create a `3D model`_ from an
astronomical image.  The 3D model is saved in a `STL`_ file, which can
then be sent to a 3D printer.

.. _3D Model: https://en.wikipedia.org/wiki/3D_modeling
.. _STL: https://en.wikipedia.org/wiki/STL_(file_format)


Installation
============

Requirements
------------

Astro3d requires:

* `Python <https://www.python.org/>`_ 2.6 (>=2.6.5), 2.7, 3.3, or 3.4

* `Numpy <https://numpy.org/>`_ 1.6 or later

* `Scipy <https://scipy.org/>`_

* `matplotlib <https://matplotlib.org/>`_

* `Astropy`_ 1.0 or later

* `Pillow <https://python-pillow.org/>`_

* `PyQt5 <https://pypi.org/project/PyQt5/>`_

* `photutils <https://photutils.readthedocs.io/en/latest/>`_


Obtaining the Source Package
----------------------------

The latest development version of ``astro3d`` can be cloned from
github using this command::

    git clone git://github.com/STScI-SSB/astro3d.git


Installing from the Source Package
----------------------------------

To install from the root of the source package::

    python setup.py install


Installing using pip
--------------------

To install the current *development* version using `pip
<https://pip.pypa.io/en/stable/>`_::

    pip install git+https://github.com/STScI-SSB/astro3d


Using ``astro3d``
=================

.. toctree::
   :maxdepth: 2

   high-level_API
