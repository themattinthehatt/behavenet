Autoencoders
============

BehaveNet uses convolutional autoencoders to perform nonlinear dimensionality reduction on behavioral videos. The steps below demonstrate how to fit these models on an arbitrary dataset.

Within the behavenet package there is a directory named ``behavenet/fitting/json_configs``, which contains example config files. First copy the following config files to the ``.behavenet`` directory that was automatically created in your home directory: ``ae_compute.json``, ``ae_model.json``, and ``ae_training.json``. You can then update the hyperparameters in these files in a text editor.

To fit a single model with the default CAE BehaveNet architecture (details in paper), edit the ``ae_model.json`` file to look like the following:

.. code-block:: json

    {
    "experiment_name": "ae-example",
    "model_type": "conv",
    "n_ae_latents": 12,
    "l2_reg": 0.0,
    "rng_seed_model": 0,
    "fit_sess_io_layers": false,
    "arch_types": "default",
    "model_class": "ae"
    }

Then to fit the model, ``cd`` to the ``behavenet`` directory in the terminal and run

.. code-block:: console

    $: python behavenet/fitting/ae_grid_search.py --data_config /user_home/.behavenet/musall_vistrained_params.json --model_config /user_home/.behavenet/ae_model.json --training_config /user_home/.behavenet/ae_training.json --compute_config /user_home/.behavenet/ae_compute.json

where ``~/.behavenet/musall_vistrained_params.json`` can be replaced by any dataset config file created by running the ``behavenet.add_dataset()`` function (example :ref:`here <add_dataset>`).

Performing a search over multiple latents is as simple as editing the ``ae_model.json`` as below and rerunning the same command.

.. code-block:: json

    {
    "experiment_name": "latent-search",
    "model_type": "conv",
    "n_ae_latents": [6, 9, 12, 15],
    "l2_reg": 0.0,
    "rng_seed_model": 0,
    "fit_sess_io_layers": false,
    "arch_types": "default",
    "model_class": "ae"
    }

