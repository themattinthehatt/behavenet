.. _glossary:

#######################
Hyperparameter glossary
#######################

The BehaveNet code requires a diverse array of hyperparameters (hparams) to specify details about
the data, computational resources, training algorithms, and the models themselves. This glossary
contains a brief description for each of the hparams options. See the
`example json files <https://github.com/ebatty/behavenet/tree/master/configs>`_ for reasonable
hparams defaults.

Data
====

* **data_dir** (*str*): absolute path to data directory
* **save_dir** (*str*): absolute path to save directory, where models are stored
* **lab** (*str*): lab id
* **expt** (*str*): experiment id
* **animal** (*str*): animal id
* **session** (*str*): session id
* **sessions_csv** (*str*): path to csv file that contains a list of sessions to use for model fitting. The 4 column headers in the csv should be ``lab``, ``expt``, ``animal``, ``session``. If this is not an empty string, it supercedes the information provided in the ``lab``, ``expt``, ``animal``, and ``session`` fields above.
* **all_source** (*str*): one of the ``expt``, ``animal``, or ``session`` fields can optionally be set to the string ``"all"``. For example, if ``expt`` is ``"all"``, then for the specified ``lab`` all sessions from all experiments/animals are collected and fit with the same model. If ``animal`` is ``"all"``, then for the specified ``lab`` and ``expt`` all sessions are collected. The field ``all_source`` tells the code where to look for the corresponding sessions: ``"data"`` will search for all sessions in ``data_dir``; ``"save"`` will search for all sessions in ``save_dir``, and as such will only find the sessions that have been previously used to fit models.
* **n_input_channels** (*str*): number of colors channel/camera views in behavioral video
* **y_pixels** (*int*): number of behavioral video pixels in y dimension
* **x_pixels** (*int*): number of behavioral video pixels in x dimension
* **use_output_mask** (*bool*): `True`` to apply frame-wise output masks (must be a key ``masks`` in data HDF5 file)
* **use_label_mask** (*bool*): `True`` to apply frame-wise masks to labels in conditional ae models (must be a key ``labels_masks`` in data HDF5 file)
* **n_labels** (*bool*): specify number of labels when model_class is 'neural-labels' or 'labels-neural'
* **neural_bin_size** (*float*): bin size of neural/video data (ms)
* **neural_type** (*str*): 'spikes' | 'ca'
* **approx_batch_size** (*str*): approximate batch size (number of frames) for gpu memory calculation

For encoders/decoders, additional information can be supplied to control which subsets of neurons
are used for encoding/decoding. See the :ref:`data structure documentation<data_structure_subsets>`
for detailed instructions on how to incorporate this information into your HDF5 data file. The
following options must be added to the data json file (an example can be found
`here <https://github.com/ebatty/behavenet/blob/master/configs/decoding_jsons/decoding_data.json>`__):

* **subsample_idxs_group_0** (*str*): name of the top-level HDF5 group that contains index groups
* **subsample_idxs_group_1** (*str*): name of the second-level HDF5 group that contains index datasets
* **subsample_idxs_dataset** (*str*): use "all" to have test tube loop over each index dataset in ``subsample_idxs_group_0/subsample_idxs_group_1``, or specify a single user-defined index dataset as a string
* **subsample_method** (*str*): determines how different index datasets are subsampled

    * 'none': no subsampling; all neural data is used for encoding/decoding
    * 'single': for the index dataset specified by 'subsample_idxs_dataset', use *just* these indices for decoding
    * 'loo': leave-one-out; for the index dataset specified by 'subsample_idxs_dataset', use all *except* this dataset for decoding

Computational resources
=======================

* **device** (*str*): where to fit pytorch models; 'cpu' | 'cuda'
* **n_parallel_gpus** (*int*): number of gpus to use per model, currently only implemented for AEs 
* **tt_n_gpu_trials** (*int*): total number of hyperparameter combinations to fit with test-tube on gpus
* **tt_n_cpu_trials** (*int*): total number of hyperparameter combinations to fit with test-tube on cpus
* **tt_n_cpu_workers** (*int*): total number of cpu cores to use with test-tube for hyperparameter searching
* **mem_limit_gb** (*float*): maximum size of gpu memory; used to filter out randomly generated CAEs that are too large

If using machine without slurm:

* **gpus_viz** (*str*): which gpus are visible to test-tube; multiple gpus are identified as e.g. '0;1;4'

If using slurm:

* **slurm** (*bool*): true if using slurm, false otherwise
* **slurm_log_path** (*str*): directory in which to save slurm outputs (.err/.out files)
* **slurm_param_file** (*str*): file name of the .sh file with slurm params for job submission

Training
========

All models:

* **as_numpy** (*bool*): ``True`` to load data as numpy arrays, ``False`` to load as pytorch tensors
* **batch_load** (*bool*): ``True`` to load data one batch at a time, ``False`` to load all data into memory (the data is still served to models in batches)
* **rng_seed_data** (*int*): control randomness when splitting data into train, val, and test trials
* **train_frac** (*float*): if ``0 < train_frac < 1.0``, defines the *fraction* of assigned training trials to actually use; if ``train_frac > 1.0``, defines the *number* of assigned training trials to actually use (rounded to the nearest integer)
* **trial_splits** (*str*): determines number of train/val/test/gap trials; entered as `8;1;1;0`, for example. See :func:`behavenet.data.data_generator.split_trials` for how these values are used.
* **export_train_plots** (*bool*): ``True`` to automatically export training/validation loss as a function of epoch upon completion of training [AEs and ARHMMs only]
* **export_latents** (*bool*): ``True`` to automatically export train/val/test autoencoder latents using best model upon completion of training [analogous parameters **export_states** and **export_predictions** exist for arhmms and decoders, respectively)
* **rng_seed_train** (*int*): control randomness in batching data

Pytorch models (all but 'arhmm' and 'bayesian-decoding'):

* **val_check_interval**: (*float*): frequency with which metrics are calculated on validation data. These metrics are logged in a csv file via test-tube, and can also be used for early stopping if enabled. If ``0 < val_check_interval < 1.0``, metrics are computed multiple times per epoch (val_check_interval=0.5 corresponds to checking every half epoch); if ``val_check_interval > 1.0``, defines number of epochs between metric computation.
* **learning_rate** (*float*): learning rate of adam optimizer
* **max_n_epochs** (*int*): maximum number of training epochs
* **min_n_epochs** (*int*): minimum number of training epochs, even when early stopping is used
* **enable_early_stop** (*bool*): if ``False``, training proceeds until maximum number of epochs is reached
* **early_stop_history** (*int*): number of epochs over which to average validation loss

ARHMM:

* **n_iters** (*int*): number of EM iterations (currently no early stopping)
* **arhmm_es_tol** (*float*): relative tolerance for early stopping; training terminates if the absolute value of the difference between the previous log likelihood (ll) and current ll, divided by the current ll, is less than this value


Models
======

All models:

* **experiment_name** (*str*): name of the test-tube experiment
* **rng_seed_model** (*int*): control initialization of model parameters
* **model_class**: (*str*): name of the model class

    * 'ae': autoencoder
    * 'vae': variational autoencoder
    * 'beta-tcvae': variational autoencoder with beta tc-vae decomposition of elbo
    * 'cond-ae': conditional autoencoder
    * 'cond-vae': conditional variational autoencoder
    * 'cond-ae-msp': autoencoder with matrix subspace projection loss
    * 'ps-vae': partitioned subspace variational autoencoder
    * 'msps-vae': multi-session partitioned subspace variational autoencoder
    * 'hmm': hidden Markov model
    * 'arhmm': autoregressive hidden Markov model
    * 'neural-ae': decode AE latents from neural activity
    * 'neural-ae-me': decode motion energy of AE latents (absolute value of temporal difference) from neural activity
    * 'neural-arhmm': decode arhmm states from neural activity
    * 'ae-neural': predict neural activity from AE latents
    * 'arhmm-neural': predict neural activity from arhmm states
    * 'labels-images': decode images from labels with a convolutional decoder
    * 'bayesian-decoding': baysian decoding of AE latents and arhmm states from neural activity


Pytorch models (all but 'arhmm' and 'bayesian-decoding'):

* **l2_reg** (*float*): L2 regularization value applied to all model weights


Autoencoder
-----------

* **model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): output dimensions of AE encoder network
* **fit_sess_io_layers** (*bool*): ``True`` to fit session-specific input and output layers; all other layers are shared across all sessions
* **ae_arch_json** (*str*): ``null`` to use the default convolutional autoencoder architecture from the original behavenet paper; otherwise, a string that defines the path to a json file that defines the architecture. An example can be found `here <https://github.com/ebatty/behavenet/tree/master/configs>`__.


Variational autoencoders
------------------------

In addition to the autoencoder parameters defined above,

* **vae.beta** (*float*): weight on KL divergence term in VAE ELBO
* **vae.beta_anneal_epochs** (*int*): number of epochs over which to linearly increase VAE beta
* **beta_tcvae.beta** (*float*) weight on total correlation term in Beta TC-VAE ELBO
* **beta_tcvae.beta_anneal_epochs** (*int*): number of epochs over which to linearly increase Beta TC-VAE beta
* **ps_vae.alpha** (*float*) weight on label reconstruction term in (MS)PS-VAE ELBO
* **ps_vae.beta** (*float*) weight on unsupervised disentangling term in (MS)PS-VAE ELBO
* **ps_vae.delta** (*float*): weight on triplet loss for the background subspace of the MSPS-VAE
* **ps_vae.anneal_epochs** (*int*): number of epochs over which to linearly increase (MS)PS-VAE beta
* **n_background** (*int*): dimensionality of background subspace in the MSPS-VAE
* **n_sessions_per_batch** (*int*): number of sessions to use for a single batch when training the MSPS-VAE (triplet loss needs data from multiple sessions)


Conditional autoencoders
------------------------

In addition to the autoencoder parameters defined above,

* **conditional_encoder** (*bool*): ``True`` to condition encoder on labels when model class is 'cond-ae'
* **msp.alpha** (*float*): weight on label reconstruction term when model class is 'cond-ae-msp'


ARHMM
-----

* **model_type** (*NoneType*): not used for ARHMMs
* **n_arhmm_lags** (*int*): number of autoregressive lags (order of AR process)
* **noise_type** (*str*): observation noise; 'gaussian' | 'studentst' | 'diagonal_gaussian' | 'diagonal_studentst'
* **transitions** (*float*): transition model; 'stationary' | 'sticky' | 'recurrent' | 'recurrent_only'
* **kappa** (*float*): stickiness parameter that biases diagonal of Markov transition matrix, which increases average state durations

* **ae_experiment_name** (*str*): name of AE test-tube experiment
* **ae_version** (*str* or *int*): 'best' to choose best version in AE experiment, otherwise an integer specifying test-tube version number
* **ae_model_class** (*str*): 'ae' | 'vae' | 'beta-tcvae' | ...
* **ae_model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): number of autoencoder latents; this will be the observation dimension in the ARHMM
* **export_train_plots** ('*bool*): ``True`` to automatically export training/validation log probability as a function of epoch upon completion of training
* **export_states** (*bool*): ``True`` to automatically export train/val/test states using best model upon completion of training


Decoder
-------

For both continuous and discrete decoders:

* **model_type**: 

    * 'mlp' - standard feedforward neural network; use ``n_hid_layers=0`` (see below) for linear regression
    * 'mlp-mv' - use the neural network to estimate both the mean and the covariance matrix of the AE latents
    * 'lstm' - currently not implemented

* **n_hid_layers** (*int*): number of hidden layers in decoder, not counting data or output layer
* **n_hid_units** (*int*): number of units in all hidden layers; the code will automatically choose the correct number of units for the output layer based on the data size
* **n_lags** (*int*): number of time lags in neural activity to use in predicting outputs; if ``n_lags=n``, then the window of neural activity ``t-n:t+n`` is used to predict the outputs at time ``t`` (and therefore ``2n+1`` total time points are used to predict each time point)
* **n_max_lags** (*int*): maximum number of lags the user thinks they may search over; the first ``n_max_lags`` and final ``n_max_lags`` time points of each batch are not used in the calculation of metrics to make models with differing numbers of lags directly comparable
* **activation** (*str*): activation function of hidden layers; activation function of final layer is automatically chosen based on decoder/data type; 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
* **export_predictions** (*bool*): ``True`` to automatically export train/val/test predictions using best model upon completion of training


For the continuous decoder:

* **ae_experiment_name** (*str*): name of AE test-tube experiment
* **ae_version** (*str* or *int*): 'best' to choose best version in AE experiment, otherwise an integer specifying test-tube version number
* **ae_model_class** (*str*): 'ae' | 'vae' | 'beta-tcvae' | ...
* **ae_model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): number of autoencoder latents; this will be the dimension of the data predicted by the decoder
* **ae_multisession** (*int*): use if loading latents from an AE that was trained on multiple datasets


For the discrete decoder:

* **n_ae_latents** (*int*): number of autoencoder latents that the ARHMM was trained on
* **ae_model_class** (*str*): 'ae' | 'vae' | 'beta-tcvae' | ...
* **ae_model_type** (*str*): 'conv' | 'linear'
* **arhmm_experiment_name** (*str*): name of ARHMM test-tube experiment
* **n_arhmm_states** (*int*): number of ARHMM discrete states; this will be the number of classes the decoder is trained on
* **n_arhmm_lags** (*int*): number of autoregressive lags (order of AR process)
* **kappa** (*float*): 'kappa' parameter of the desired ARHMM
* **noise_type** (*str*): 'noise_type' parameter of the desired ARHMM; 'gaussian' | 'studentst'
* **arhmm_version** (*str* or *int*): 'best' to choose best version in ARHMM experiment, otherwise an integer specifying test-tube version number
* **arhmm_multisession** (*int*): use if loading states from an ARHMM that was trained on multiple datasets


Bayesian decoder
----------------

TODO


