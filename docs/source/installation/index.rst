.. _Installation:

Installation
=============

.. _require:

Requirements
------------

To run pysmFISH you will need python >=3.6 (we have no plans to support python<3.6).
The code has still some dependencies from MPI (we are currently removing them).  
Install mpich2 on linux using
::

    apt-get install -y mpich

or on OSX using
::
    
    brew install mpich2


.. _pypi:

Install using PyPI
------------------

Every new stable version of ``pysmFISH`` gets immediately released on PyPI, so running the following command will install on your system the cutting-edge version:

::

    pip install --no-cache-dir pysmFISH


To get started with ``pysmFISH`` you can follow :ref:`our guide <tutorial>`. 


.. _fromsource:

Install from source
-------------------

If you plan to explore and make changes to the source code, or you have requested some bug-fix that is temporarily available only on the github ``dev`` branch, then you need to install ``pysmFISH`` directly from source.


First of all, make sure all the dependencies are installed, and that `git` is installed on your system. 
Then, run the following commands to complete the installation:

::

    git clone https://github.com/linnarsson-lab/pysmFISH.git
    cd pysmFISH.py
    pip install -e .  # note the trailing dot

You can test whether the installation was successful by running ``pysmFISH --help``.

.. tip::
    ``pysmFISH`` |version| is an alpha release, we recommend pulling in the latest bufixes and feature improvements often. Note that adding the ``-e`` flag to the pip command installs the software in `development` mode, when a package is installed this way each change to the source immediatelly reflects to changes in the installed library. This means that you can simply use ``git pull`` to update your installation.

To get started with ``pysmFISH`` you can follow :ref:`our guide <tutorial>`. 


.. _conda:

Install using conda
-------------------

.. note::
   This installation method is not currently available. Our plan is make it available asap.