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

Example configuration files can be found `here <https://github.com/ebatty/behavenet/tree/master/behavenet/json_configs>`_.

For example, the command line call to fit an autoencoder would be (using the default json files):

.. code-block:: console
    
    $: cd /behavenet/code/directory/
    $: cd behavenet
    $: python fitting/ae_grid_search.py --data_config ./json_configs/data_default.json --model_config ./json_configs/ae_model.json --training_config ./json_configs/ae_training.json --compute_config ./json_configs/ae_compute.json

We recommend that you copy the default config files in the behavenet repo into a separate directory on your local machine and make edits there. For more information on the different hyperparameters, see the :ref:`hyperparameters glossary<glossary>`.


Adding a new dataset
--------------------

When using BehaveNet with a new dataset you will need to make a new data config json file, which can be automatically generated using a BehaveNet helper function. You will be asked to enter the following information (examples shown for Musall dataset):

* lab or experimenter name (:code:`musall`)
* experiment name (:code:`vistrained`)
* example animal name (:code:`mSM36`)
* example session name (:code:`05-Dec-2017`)
* x pixels (:code:`128`)
* y pixels (:code:`128`)
* input channels (:code:`2`) - this can refer to color channels (for RGB data) and/or number of camera views, which should be concatenated along the color channel dimension. In the Musall dataset we use grayscale images from two camera views, so a trial with 189 frames will have a block of video data of shape (189, 2, 128, 128)
* use output mask (:code:`False`) - an optional output mask can be applied to each video frame if desired; these output masks must also be stored in the :code:`data.hdf5` files as :code:`masks`.
* frame rate (:code:`30`) - in Hz; BehaveNet assumes that the video data and neural data are binned at the same temporal resolution
* neural data type (:code:`ca`) - either :code:`ca` for 2-photon/widefield data, or :code:`spikes` for ephys data. This parameter controls the noise distribution for encoding models, as well as several other model hyperparameters.

To enter this information, launch python from the behavenet environment and type:

.. code-block:: python

    from behavenet import add_dataset
    add_dataset()

This function will create a json file named :code:`[lab_id]_[expt_id]` in the :code:`.behavenet` directory in your user home directory, which you can manually update at any point using a text editor.


Organizing model fits with test-tube
------------------------------------

BehaveNet uses the `test-tube package <https://williamfalcon.github.io/test-tube/>`_ to organize model fits into user-defined experiments, log meta and training data, and perform grid searches over model hyperparameters. Most of this occurs behind the scenes, but there are a couple of important pieces of information that will improve your model fitting experience.

BehaveNet organizes model fits using a combination of hyperparameters and user-defined experiment names. For example, let's say you want to fit 5 different convolutional autoencoder architectures, all with 12 latents, to find the best one. Let's call this experiment "arch_search", which you will set in the :obj:`model_config` json in the :obj:`experiment_name` field. The results will then be stored in the directory :obj:`results_dir/lab_id/expt_id/animal_id/session_id/ae/conv/12_latents/arch_search/`.

Each model will automatically be assigned it's own "version" by test-tube, so the :obj:`arch_search` directory will have subdirectories :obj:`version_0`, ..., :obj:`version_4`. If an additional CAE model is later fit with 12 latents (and using the "arch_search" experiment name), test-tube will add it to the :obj:`arch_search` directory as :obj:`version_5`. This includes changing the architecture, learning rate, regularization values, etc. Each model class (autoencoder, arhmm, decoders) have a set of hyperparameters that are used for directory names, and another set that are used to distinguish test-tube versions within the final test-tube experiment.

Within the :obj:`version_x` directory, there are various files saved during training. Here are some of the files automatically output when training an autoencoder:

* **best_val_model.pt**: the best model as determined by computing the loss on validation data
* **meta_tags.csv**: hyperparameters associated with data, computational resources, training, and model
* **metrics.csv**: metrics computed on dataset as a function of epochs; the default is that metrics are computed on training and validation data every epoch (and reported as a mean over all batches) while metrics are computed on test data only at the end of training using the best model (and reported per batch).
* **session_info.csv**: experimental sessions used to fit the model

Additionally, if you set :obj:`export_latents` to :obj:`True` in the training config file, you will see

* **lab-id_expt-id_animal-id_session-id_latents.pkl**: list of np.ndarrays of CAE latents computed using the best model

and if you set :obj:`export_train_plots` to :obj:`True` in the training config file, you will see

* **loss_training.png**: MSE as a function of training epoch on training data
* **loss_validation.png**: MSE as a function of training epoch on validation data

