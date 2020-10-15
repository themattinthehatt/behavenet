Introduction
============

BehaveNet is a software package that provides tools for analyzing behavioral video and neural activity. Currently BehaveNet supports:

* Video compression using convolutional autoencoders
* Video segmentation (and generation) using autoregressive hidden Markov models
* Neural network decoding of videos from neural activity
* Bayesian decoding of videos from neural activity

BehaveNet automatically saves models using a well-defined and flexible directory structure, allowing for easy management of many models and multiple datasets.


The command line interface
--------------------------

Users interact with BehaveNet using a command line interface, so all model fitting is done from the terminal. To simplify this process all necessary parameters are defined in four configuration files that can be manually updated using a text editor:

* **data_config** - dataset ids, video frames sizes, etc. You can automatically generate this configuration file for a new dataset by following the instructions in the following section.
* **model_config** - model hyperparameters
* **training_config** - learning rate, number of epochs, etc.
* **compute_config** - gpu vs cpu, gpu ids, etc.

Example configuration files can be found `here <https://github.com/ebatty/behavenet/tree/master/configs>`_.

For example, the command line call to fit an autoencoder would be (using the default json files):

.. code-block:: console
    
    $: cd /behavenet/code/directory/
    $: cd behavenet
    $: python fitting/ae_grid_search.py --data_config ../configs/data_default.json --model_config ../configs/ae_model.json --training_config ../configs/ae_training.json --compute_config ../configs/ae_compute.json

We recommend that you copy the default config files in the behavenet repo into a separate directory on your local machine and make edits there. For more information on the different hyperparameters, see the :ref:`hyperparameters glossary<glossary>`.


.. _add_dataset:

Adding a new dataset
--------------------

When using BehaveNet with a new dataset you will need to make a new data config json file, which can be automatically generated using a BehaveNet helper function. You will be asked to enter the following information (examples shown for Musall dataset):

* lab or experimenter name (:code:`musall`)
* experiment name (:code:`vistrained`)
* example animal name (:code:`mSM36`)
* example session name (:code:`05-Dec-2017`)
* input channels (:code:`2`) - this can refer to color channels (for RGB data) and/or number of camera views, which should be concatenated along the color channel dimension. In the Musall dataset we use grayscale images from two camera views, so a trial with 189 frames will have a block of video data of shape (189, 2, 128, 128)
* y pixels (:code:`128`)
* x pixels (:code:`128`)
* use output mask (:code:`False`) - an optional output mask can be applied to each video frame if desired; these output masks must also be stored in the :code:`data.hdf5` files as :code:`masks`.
* frame rate (:code:`30`) - in Hz; BehaveNet assumes that the video data and neural data are binned at the same temporal resolution
* neural data type (:code:`ca`) - either :code:`ca` for 2-photon/widefield data, or :code:`spikes` for ephys data. This parameter controls the noise distribution for encoding models, as well as several other model hyperparameters.

To enter this information, launch python from the behavenet environment and type:

.. code-block:: python

    from behavenet import add_dataset
    add_dataset()

This function will create a json file named ``[lab_id]_[expt_id].json`` in the ``.behavenet`` directory in your user home directory, which you can manually update at any point using a text editor.


Organizing model fits with test-tube
------------------------------------

BehaveNet uses the `test-tube package <https://williamfalcon.github.io/test-tube/>`_ to organize model fits into user-defined experiments, log meta and training data, and perform grid searches over model hyperparameters. Most of this occurs behind the scenes, but there are a couple of important pieces of information that will improve your model fitting experience.

BehaveNet organizes model fits using a combination of hyperparameters and user-defined experiment names. For example, let's say you want to fit 5 different convolutional autoencoder architectures, all with 12 latents, to find the best one. Let's call this experiment "arch_search", which you will set in the ``model_config`` json in the ``experiment_name`` field. The results will then be stored in the directory ``results_dir/lab_id/expt_id/animal_id/session_id/ae/conv/12_latents/arch_search/``.

Each model will automatically be assigned it's own "version" by test-tube, so the ``arch_search`` directory will have subdirectories ``version_0``, ..., ``version_4``. If an additional CAE model is later fit with 12 latents (and using the "arch_search" experiment name), test-tube will add it to the ``arch_search`` directory as ``version_5``. Different versions may have different architectures, learning rates, regularization values, etc. Each model class (autoencoder, arhmm, decoders) has a set of hyperparameters that are used for directory names, and another set that are used to distinguish test-tube versions within the user-defined experiment.

Within the ``version_x`` directory, there are various files saved during training. Here are some of the files automatically output when training an autoencoder:

* **best_val_model.pt**: the best model (not necessarily from the final training epoch) as determined by computing the loss on validation data
* **meta_tags.csv**: hyperparameters associated with data, computational resources, training, and model
* **metrics.csv**: metrics computed on dataset as a function of epochs; the default is that metrics are computed on training and validation data every epoch (and reported as a mean over all batches) while metrics are computed on test data only at the end of training using the best model (and reported per batch).
* **session_info.csv**: experimental sessions used to fit the model

Additionally, if you set ``export_latents`` to ``True`` in the training config file, you will see

* **[lab_id]_[expt_id]_[animal_id]_[session_id]_latents.pkl**: list of np.ndarrays of CAE latents computed using the best model

and if you set ``export_train_plots`` to ``True`` in the training config file, you will see

* **loss_training.png**: plot of MSE as a function of training epoch on training data
* **loss_validation.png**: same as above using validation data


.. _grid_search_tt:

Grid searching with test-tube
-----------------------------

Beyond organizing model fits, test-tube is also useful for performing grid searches over model hyperparameters, using multiple cpus or gpus. All you as the user need to do is enter the relevant hyperparameter choices as a list instead of a single value in the associated configuration file. 

Again using the autoencoder as an example, let's say you want to fit a single AE architecture using 4 different numbers of latents, all with the same regularization value. In the model config file, you will set these values as:

.. code-block:: json

    {
    ...
    "n_ae_latents": [4, 8, 12, 16],
    "l2_reg": 0.0,
    ...
    }

To specify the computing resources for this job, you will next edit the compute config file, which looks like this:

.. code-block:: json

    {
    ...
    "device": "cuda", # "cpu" or "cuda"
    "gpus_viz": "0", # "add multiple gpus as e.g. "0;1;4"
    "tt_n_gpu_trials": 1000,
    "tt_n_cpu_trials": 1000,
    "tt_n_cpu_workers": 5,
    ...
    }

With the ``device`` field set to ``cuda``, test-tube will use gpus to run this job. The ``gpus_viz`` field can further specify which subset of gpus to use. The ``tt_n_gpu_trials`` defines the maximum number of jobs to run. If this number is larger than the total number of hyperparameter configurations, all configurations are fit; if this number is smaller than the total number (say if ``"tt_n_gpu_trials": 2`` in this example) then this number of configurations is randomly sampled from all possible choices.

To fit models using the cpu instead, set the ``device`` field to ``cpu``; then ``tt_n_cpu_workers`` defines the total number of cpus to run the job (total number of models fitting at any one time) and ``tt_n_cpu_trials`` is analogous to ``tt_n_gpu_trials``.

Finally, multiple hyperparameters can be searched over simultaneously; for example, to search over both AE latents and regularization values, set these parameters in the model config file like so:

.. code-block:: json

    {
    ...
    "n_ae_latents": [4, 8, 12, 16],
    "l2_reg": [1e-5, 1e-4, 1e-3],
    ...
    }

This job would then fit a total of 4 latent values x 3 regularization values = 12 models.
