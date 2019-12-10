.. _glossary:

#######################
Hyperparameter glossary
#######################

The BehaveNet code requires a diverse array of hyperparameters (hparams) that specify details about the data, computational resources to be used for model fitting, and the models themselves. This glossary contains a brief description for each of the hparams options. See the example json files for hparams defaults.

Data
====

* **data_dir** (*str*): absolute path to data directory
* **save_dir** (*str*): absolute path to save directory, where models are stored
* **sessions_csv** (*str*): list of sessions to use for model fitting in csv file. The 4 column headers should be :obj:`lab`, :obj:`expt`, :obj:`animal`, :obj:`session`.
* **as_numpy** (*bool*): :obj:`True` to load data as numpy arrays, :obj:`False` to load as pytorch tensors
* **batch_load** (*bool*): :obj:`True` to load data one batch at a time, :obj:`False` to load all data into memory (the data is still served to models in batches)
* **rng_seed_data** (*int*): control randomness when splitting data into train, val, and test trials
* **train_frac** (*float*): if :obj:`0 < train_frac < 1.0`, defines the *fraction* of assigned training trials to actually use; if :obj:`train_frac > 1.0`, defines the *number* of assigned training trials to actually use (rounded to the nearest integer)


Computational resources
=======================

* **device** (*str*): where to fit pytorch models; 'cpu' | 'cuda'
* **tt_n_gpu_trials** (*int*): total number of hyperparameter combinations to fit with test-tube on gpus
* **tt_n_cpu_trials** (*int*): total number of hyperparameter combinations to fit with test-tube on cpus
* **tt_n_cpu_workers** (*int*): total number of cpu cores to use with test-tube for hyperparameter searching
* **mem_limit_gb** (*float*): maximum size of gpu memory; used to filter out randomly generated CAEs that are too large
* **gpus_viz** (*str*): which gpus are visible to test-tube; multiple gpus are identified as e.g. '0;1;4'


Models
======

All models:

* **experiment_name** (*str*): name of the test-tube experiment
* **rng_seed_model** (*int*): control initialization of model parameters
* **model_class**: (*str*): name of the model class

    * 'ae': autoencoder
    * 'vae': variational autoencoder
    * 'hmm': hidden Markov model
    * 'arhmm': autoregressive hidden Markov model
    * 'neural-ae': decode AE latents from neural activity
    * 'neural-arhmm': decode arhmm states from neural activity
    * 'ae-neural': predict neural activity from AE latents
    * 'arhmm-neural': predict neural activity from arhmm states
    * 'bayesian-decoding': baysian decoding of AE latents and arhmm states from neural activity


Pytorch models (all but 'arhmm' and 'bayesian-decoding'):

* **learning_rate** (*float*): learning rate of adam optimizer
* **min_n_epochs** (*int*): minimum number of training epochs, even when early stopping is used
* **max_n_epochs** (*int*): maximum number of training epochs
* **val_check_interval**: (*float*): frequency with which metrics are calculated on validation data. These metrics are logged in a csv file via test-tube, and can also be used for early stopping if enabled. If :obj:`0 < val_check_interval < 1.0`, metrics are computed multiple times per epoch (val_check_interval=0.5 corresponds to checking every half epoch); if :obj:`val_check_interval > 1.0`, defines number of epochs between metric computation.
* **enable_early_stop** (*bool*): if :obj:`False`, training proceeds until maximum number of epochs is reached
* **early_stop_history** (*int*): number of epochs over which to average validation loss
* **l2_reg** (*float*): L2 regularization value applied to all model weights


Autoencoder
-----------

* **model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): output dimensions of AE encoder network
* **fit_sess_io_layers** (*bool*): :obj:`True` to fit session-specific input and output layers; all other layers are shared across all sessions
* **export_train_plots** (*bool*): :obj:`True` to automatically export training/validation loss as a function of epoch upon completion of training
* **export_latents** (*bool*): :obj:`True` to automatically export train/val/test latents using best model upon completion of training


ARHMM
-----

* **model_type** (*NoneType*): not used for ARHMMs
* **n_arhmm_lags** (*int*): number of autoregressive lags (order of AR process)
* **noise_type** (*str*): observation noise; 'gaussian' | 'studentst'
* **kappa** (*float*): stickiness parameter that biases diagonal of Markov transition matrix, which increases average state durations
* **n_iters** (*int*): number of EM iterations (currently no early stopping)
* **ae_experiment_name** (*str*): name of AE test-tube experiment
* **ae_version** (*str* or *int*): 'best' to choose best version in AE experiment, otherwise an integer specifying test-tube version number
* **ae_model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): number of autoencoder latents; this will be the observation dimension in the ARHMM
* **export_train_plots** ('*bool*): :obj:`True` to automatically export training/validation log probability as a function of epoch upon completion of training
* **export_states** (*bool*): :obj:`True` to automatically export train/val/test states using best model upon completion of training


Decoder
-------

For both continuous and discrete decoders:

* **model_type**: 

    * 'ff' - standard feedforward neural network; use :obj:`n_hid_layers=0` (see below) for linear regression
    * 'ff-mv' - use the neural network to estimate both the mean and the covariance matrix of the AE latents
    * 'lstm' - currently not implemented

* **n_hid_layers** (*int*): number of hidden layers in decoder, not counting data or output layer
* **n_final_units** (*int*): number of units in the final hidden layer; the code will automatically choose the correct number of units for the output layer based on the data size
* **n_int_units** (*int*): number of units in all hidden layers except the final
* **n_lags** (*int*): number of time lags in neural activity to use in predicting outputs; if :obj:`n_lags=n`, then the window of neural activity :obj:`t-n:t+n` is used to predict the outputs at time :obj:`t` (and therefore :obj:`2n+1` total time points are used to predict each time point)
* **n_max_lags** (*int*): maximum number of lags the user thinks they may search over; the first :obj:`n_max_lags` and final :obj:`n_max_lags` time points of each batch are not used in the calculation of metrics to make models with differing numbers of lags directly comparable
* **activation** (*str*): activation function of hidden layers; activation function of final layer is automatically chosen based on decoder/data type; 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
* **export_predictions** (*bool*): :obj:`True` to automatically export train/val/test predictions using best model upon completion of training
* **reg_list** (*str*):  
* **subsample_regions** (*str*): determines how neural regions are subsampled

    * 'none': no subsampling
    * 'single': for each region in 'reg_list', use *just* this region for decoding
    * 'loo': leave-one-out; for each region in 'reg_list', use all *except* this region for decoding


For the continuous decoder:

* **ae_experiment_name** (*str*): name of AE test-tube experiment
* **ae_version** (*str* or *int*): 'best' to choose best version in AE experiment, otherwise an integer specifying test-tube version number
* **ae_model_type** (*str*): 'conv' | 'linear'
* **n_ae_latents** (*int*): number of autoencoder latents; this will be the dimension of the data predicted by the decoder
* **ae_multisession** (*int*): use if loading latents from an AE that was trained on multiple datasets


For the discrete decoder:

* **n_ae_latents** (*int*): number of autoencoder latents that the ARHMM was trained on
* **ae_model_type** (*str*): 'conv' | 'linear'
* **arhmm_experiment_name** (*str*): name of ARHMM test-tube experiment
* **n_arhmm_states** (*int*): number of ARHMM discrete states; this will be the number of classes the decoder is trained on
* **kappa** (*float*): 'kappa' parameter of the desired ARHMM
* **noise_type** (*str*): 'noise_type' parameter of the desired ARHMM; 'gaussian' | 'studentst'
* **arhmm_version** (*str* or *int*): 'best' to choose best version in ARHMM experiment, otherwise an integer specifying test-tube version number
* **arhmm_multisession** (*int*): use if loading states from an ARHMM that was trained on multiple datasets


Bayesian decoder
----------------

TODO

