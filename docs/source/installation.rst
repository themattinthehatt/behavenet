############
Installation
############

Before you begin, ensure that you have downloaded both Git (for cloning the repository) and `Anaconda <https://www.anaconda.com/distribution/>`_ (for managing the code environment). The following steps will show you how to:

1. Set up a virtual environment using Anaconda
2. Install the BehaveNet and ssm packages
3. Store user paths in a local json file

Environment setup
=================
First set up the anaconda environment from the command line.

.. code-block:: console

    $: conda create --name=behavenet python=3.7.2
    $: conda activate behavenet

Package installation
====================

BehaveNet
---------

``cd`` to the directory that will contain the BehaveNet code, and then clone the BehaveNet repository from github:

.. code-block:: console

    (behavenet) $: git clone https://github.com/ebatty/behavenet

``cd`` into the ``behavenet`` repository and install the required dependencies:

.. code-block:: console

    (behavenet) $: pip install -r requirements.txt

To make the package modules visible to the python interpreter, locally run pip install from inside the main behavenet directory:

.. code-block:: console

    (behavenet) $: pip install -e .

You can test the installation by running

.. code-block:: console

    (behavenet) $: python -c "import behavenet"

If this command does not return an error the package has been successfully installed.

Installing the BehaveNet package automatically installed the ``ipykernel`` package, which allows you to work with python code in Jupyter notebooks. To be able to use the behavenet conda environment for Jupyter notebooks, run the following command from the terminal:

.. code-block:: console

    (behavenet) $: python -m ipykernel install --user --name behavenet


ssm
---

The ``ssm`` package is the backend state space modeling code used by BehaveNet. To install ssm, ``cd`` to any directory where you would like to keep the ssm code and run the following:

.. code-block:: console

    (behavenet) $: git clone https://github.com/slinderman/ssm.git --branch behavenet-no-cython --single-branch
    (behavenet) $: cd ssm
    (behavenet) $: pip install cython
    (behavenet) $: pip install -e .


PyTorch
-------

PyTorch is automatically pip installed during the BehaveNet installation process; however, if you have issues running PyTorch, first uninstall the existing package:

.. code-block:: console

    (behavenet) $: pip uninstall torch

and reinstall following the directions `here <https://pytorch.org/get-started/locally/>`_ using the ``Pip`` package option.


ffmpeg
------

The BehaveNet package uses the ffmpeg backend to produce movies. ffmpeg is 
automatically installed on many systems, and is not automatically installed with 
BehaveNet. If you are trying to make movies and run into issues with ffmpeg,
install using the conda package manager:

.. code-block:: console
    
    (behavenet) $: conda install -c conda-forge ffmpeg


Set user paths
==============

Next, set up your paths to the directories where data, results, and figures will be stored. To do so, launch python from the behavenet environment, and type:

.. code-block:: python

    from behavenet import setup
    setup()

You will be asked to input a base data directory; all data should be stored in the form ``base_data_dir/lab_id/expt_id/animal_id/session_id/data.hdf5``. More information on the structure of the hdf5 file can be found :ref:`here<data_structure>`. You will also be asked to input a base results directory, which will store all of the model fits. Finally, the base figure directory will be used to store figure and video outputs.

The ``behavenet.setup()`` function will create a hidden directory named ``.behavenet`` in your user directory.

* In Linux, ``~/.behavenet``
* In MacOS, ``/Users/CurrentUser/.behavenet``

Within this directory the function will create a json file named ``directories`` which you can manually edit at any point.

