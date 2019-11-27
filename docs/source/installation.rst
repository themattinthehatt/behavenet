############
Installation
############

Before you begin, ensure that you have downloaded both Git (for cloning the repository) and `Anaconda <https://www.anaconda.com/distribution/>`_ (for managing the code environment).

Environment setup
=================
First set up the anaconda environment from the command line.

.. code-block:: console

    $: conda create --name=behavenet python=3.7.2
    $: source activate behavenet

To be able to use this environment for jupyter notebooks:

.. code-block:: console

    (behavenet) $: python -m ipykernel install --user --name behavenet

Package installation
====================

BehaveNet
---------

:code:`cd` to the directory that will contain the BehaveNet code, and then clone the BehaveNet repository from github:

.. code-block:: console

    (behavenet) $: git clone https://github.com/ebatty/behavenet

:code:`cd` into the :code:`behavenet` repository and install the required dependencies:

.. code-block:: console

    (behavenet) $: pip install -r requirements.txt

To make the package modules visible to the python interpreter, locally run pip install from inside the main behavenet directory:

.. code-block:: console

    (behavenet) $: pip install -e .


SSM
---

The :code:`ssm` package is the backend state space modeling code used by BehaveNet. To install ssm, :code:`cd` to any directory where you would like to keep the ssm code and run the following:

.. code-block:: console

    (behavenet) $: git clone git@github.com:slinderman/ssm.git
    (behavenet) $: cd ssm
    (behavenet) $: pip install cython
    (behavenet) $: pip install -e .

Set user paths
==============

Next, set up your paths to the directories where data, results, and figures will be stored. To do so, launch python from the behavenet environment, and type:

.. code-block:: python

    from behavenet import setup
    setup()

You will be asked to input a base data directory; all data should be stored in the form :code:`base_data_dir/lab_name/expt_name/animal_id/session_id/data.hdf5`. More information on the structure of the hdf5 file can be found :ref:`here<data_structure>`. You will also be asked to input a base results directory, which will store all of the model fits. Finally, the base figure directory will be used to store figure and video outputs.

The :code:`behavenet.setup()` method will create a hidden directory named :code:`.behavenet` in your user directory.

* In Linux, :code:`~/.behavenet`
* In MacOS, :code:`/Users/CurrentUser/.behavenet`

Within this directory the method will create a json file named :code:`directories` which you can manually edit at any point.


Adding a new dataset
====================

Next you will input some prior information about the dataset to avoid supplying this information at all intermediate steps (examples shown for Musall dataset):

* lab or experimenter name (:code:`musall`)
* experiment name (:code:`vistrained`)
* example animal name (:code:`mSM36`)
* example session name (:code:`05-Dec-2017`)
* trial splits (:code:`8;1;1;0`) - this is how trials will be split among training, validation, testing, and gap trials, respectively. Typically we use training data to train the models; validation data to choose the best model from a collection of models using different hyperparameters; test data to produce plots and videos; and gap trials can optionally be inserted between training, validation, and test trials if desired.
* x pixels (:code:`128`)
* y pixels (:code:`128`)
* input channels (:code:`2`) - this can refer to color channels (for RGB data) and/or multiple camera views, which should be concatenated along the color channel dimension. In the Musall dataset we use grayscale images from two camera views, so a trial with 189 frames will have a block of video data of shape (189, 2, 128, 128)
* use output mask (:code:`False`) - an optional output mask can be applied to each video frame if desired; these output masks must also be stored in the :code:`data.hdf5` files as :code:`masks`.
* frame rate (:code:`30`) - in Hz; behavenet assumes that the video data and neural data are binned at the same temporal resolution
* neural data type (:code:`ca`) - either :code:`ca` for 2-photon/widefield data, or :code:`spikes` for ephys data. This parameter controls the noise distribution for encoding models, as well as several other model hyperparameters.

To input this information, launch python from the behavenet environment and type:

.. code-block:: python

    from behavenet import add_dataset
    add_dataset()

This function will create a json file named :code:`[lab name]_[experiment name]` which you can manually edit at any point.
 
