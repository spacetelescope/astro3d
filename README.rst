Astro3d
=======

Astro3d provides a GUI and engine to create a 3D model from an
astronomical image.  The 3D model is saved in a `STL`_ file, which can
then be sent to a 3D printer.

.. _STL: https://en.wikipedia.org/wiki/STL_(file_format)

Requirements
------------
- Python 3
- Qt5

Installation
------------
In general, the installation procecure follows the Astroconda installation directions http://astroconda.readthedocs.io

Also note, all command line actions need to be done in the `bash` shell. One can always start a bash shell as follows::
    
    $ bash

Start a terminal window
^^^^^^^^^^^^^^^^^^^^^^^

In whatever system you have, start a terminal window that will get you to the command line.

Remember, you need to be in the `bash` shell. If not sure, start `bash`::

    $ bash
    
Install Anaconda
^^^^^^^^^^^^^^^^
If you have never installed anaconda before, or want to start fresh, this is the place to start. First, remove any previous instance of anaconda::

    $ cd
    $ rm -rf anaconda3
    
Go to the Anaconda download site,https://www.anaconda.com/download/,  and download the command line installer appropriate for your system. Choose the Python3.x version.

After downloading, perform the installation::

    $ cd <download_directory_here>
    $ bash <install_script_here>
    
Accept all the defaults for any questions asked.

Now, quit the terminal window you were in and start another one. Remember to start `bash` if necessary.

Create Astroconda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup for Astroconda by doing the following::

    $ conda config --add channels http://ssb.stsci.edu/astroconda
    
Now we can create the `astro3d` Python environment::

    $ conda create -n astro3d stsci
    
The `astro3d` environment is now ready. To activate, do the following::

    $ source activate astro3d
    
Always Do
^^^^^^^^^

At this point, the environment is setup. From now on, always make sure you are running `bash` and in the `astro3d` environment whenever you need to start a new terminal window::

    $ bash
    $ source activate astro3d
    
astro3d code install
^^^^^^^^^^^^^^^^^^^^

If one has never installed `astro3d` or one wishes to get a fresh install, do the following. Remember to start bash and activate the environment before doing this::

    $ cd
    $ rm -rf astro3d
    $ git clone https://github.com/spacetelescope/astro3d.git
    $ cd astro3d
    $ python setup.py install
    
The application should now be runnable. Note, you can run the application from whatever directory you wish::

    $ astro3d
    
Updating astro3d
^^^^^^^^^^^^^^^^

To update the code, do the following. Remember to start bash and activate the environment::

    $ cd
    $ cd astro3d
    $ git pull
    $ python setup.py install

This should do it.
