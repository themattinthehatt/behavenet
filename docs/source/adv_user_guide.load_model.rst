Loading a trained model
=======================

After you've fit one or more models, often you'll want to load these models and their associated data generator to perform further analyses. BehaveNet provides three methods for doing so:

* :ref:`Method 1<load_best_model>`: load the "best" model from a test-tube experiment
* :ref:`Method 2<specify_version>`: specify the model version in a test-tube experiment
* :ref:`Method 3<specify_hparams>`: specify the model hyperparameters in a test-tube experiment

To illustrate these three methods we'll use an autoencoder as an example. Let's assume that we've trained 5 convolutional autoencoders with 10 latents, each with a different random seed for initializing the weights, and these have all been saved in the test-tube experiment ``ae-example``.

.. _load_best_model:

Method 1: load best model
-------------------------
The first option is to load the best model from ``ae-example``. The "best" model is defined as the one with the smallest loss value computed on validation data. If you set the parameter ``val_check_interval`` in the ae training json to a nonzero value before fitting, this information has already been computed and saved in a csv file, so this is a relatively fast option. The following code block shows how to load the best model, as well as the associated data generator, from ``ae-example``.

.. code-block:: python

    # imports
    from behavenet import get_user_dir
    from behavenet.fitting.utils import get_best_model_and_data
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_lab_example
    from behavenet.fitting.utils import get_session_dir
    from behavenet.models import AE as Model

    # define necessary hyperparameters
    hparams = {
        'data_dir': get_user_dir('data'),
        'save_dir': get_user_dir('save'),
        'experiment_name': 'ae-example',
        'model_class': 'ae',
        'model_type': 'conv',
        'n_ae_latents': 10,
    }

    # programmatically fill out other hparams options
    get_lab_example(hparams, 'musall', 'vistrained')
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)

    # use helper function to load model and data generator
    model, data_generator = get_best_model_and_data(hparams, Model, version='best')


.. _specify_version:

Method 2: specify the model version
-----------------------------------
The next option requires that you know in advance which test-tube version you want to load. In this example, we'll load version 3. All you need to do is replace ``version='best'`` with ``version=3`` in the final line above.

.. code-block:: python

    # use helper function to load model and data generator
    model, data_generator = get_best_model_and_data(hparams, Model, version=3)


.. _specify_hparams:

Method 3: specify model hyperparameters
---------------------------------------
The final option gives you the most control - you can specify all relevant hyperparameters needed to define the model and the data generator, and load that specific model.

.. code-block:: python

    # imports
    from behavenet import get_user_dir
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_best_model_and_data
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_lab_example
    from behavenet.fitting.utils import get_session_dir
    from behavenet.models import AE as Model

    # define necessary hyperparameters
    hparams = {
        'data_dir': get_user_dir('data'),
        'save_dir': get_user_dir('save'),
        'experiment_name': 'ae-example',
        'model_class': 'ae',
        'model_type': 'conv',
        'n_ae_latents': 10,
        'rng_seed_data': 0,
        'trial_splits': '8;1;1;0',
        'train_frac': 1,
        'rng_seed_model': 0,
        'fit_sess_io_layers': False,
        'learning_rate': 1e-4,
        'l2_reg': 0,
    }

    # programmatically fill out other hparams options
    get_lab_example(hparams, 'musall', 'vistrained')
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)

    # find the version for these hyperparameters; returns None for version if it doesn't exist
    exists, version = experiment_exists(hparams, which_version=True)

    # use helper function to load model and data generator
    model, data_generator = get_best_model_and_data(hparams, Model, version=version)

You will need to specify the following entries in ``hparams`` regardless of the model class:

* 'rng_seed_data'
* 'trial_splits'
* 'train_frac'
* 'rng_seed_model'
* 'model_class'
* 'model_type'

For the autencoder, we need to additionally specify ``n_ae_latents``, ``fit_sess_io_layers``, ``learning_rate``, and ``l2_reg``. Check out the source code for :py:func:`behavenet.fitting.utils.get_model_params` to see which entries are required for other model classes.


Iterating through the data
--------------------------

Below is an example of how to iterate through the data generator and load batches of data:

.. code-block:: python

    # select data type to load
    dtype = 'train'  # 'train' | 'val' | 'test'

    # reset data iterator for this data type
    data_generator.reset_iterators(dtype)

    # loop through all batches for this data type
    for i in range(data_generator.n_tot_batches[dtype]):

        batch, sess = data_generator.next_batch(dtype)
        # "batch" is a dict with keys for the relevant signal, e.g. 'images', 'neural', etc.
        # "sess" is an integer denoting the dataset this batch comes from

        # ... perform analyses ...
