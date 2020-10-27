Decoders
========

The next step of the BehaveNet pipeline uses the neural activity to decode (or reconstruct) aspects
of behavior. In particular, you may decode either the AE latents or the ARHMM states on a
frame-by-frame basis given the surrounding window of neural activity.

The architecture options consist of a linear model or feedforward neural network: exact
architecture parameters such as number of layers in the neural network can be specified in
``decoding_ae_model.json`` or ``decoding_arhmm_model.json``. The size of the window of neural
activity used to reconstruct each frame of AE latents or ARHMM states is set by ``n_lags``: the
neural activity from ``t-n_lags:t+n_lags`` will be used to predict the latents or states at time
``t``.

To begin fitting decoding models, copy the example json files ``decoding_ae_model.json``,
``decoding_arhmm_model.json``, ``decoding_compute.json``, and ``decoding_training.json`` into your
``.behavenet`` directory. ``cd`` to the ``behavenet`` directory in the terminal, and run:

Decoding ARHMM states:

.. code-block:: console

    $: python behavenet/fitting/decoding_grid_search.py --data_config ~/.behavenet/musall_vistrained_params.json --model_config ~/.behavenet/decoding_arhmm_model.json --training_config ~/.behavenet/decoding_training.json --compute_config ~/.behavenet/decoding_compute.json
    
or

Decoding AE states:

.. code-block:: console

    $: python behavenet/fitting/decoding_grid_search.py --data_config ~/.behavenet/musall_vistrained_params.json --model_config ~/.behavenet/decoding_ae_model.json --training_config ~/.behavenet/decoding_training.json --compute_config ~/.behavenet/decoding_compute.json


.. _decoding_with_subsets:

Decoding with subsets of neurons
--------------------------------

Continuing with the toy dataset introduced in the :ref:`data structure<data_structure_subsets>`
documentation, below are some examples for how to modify the decoding data json file to decode from
user-specified groups of neurons:

**Example 0**: 

Use all neurons:

.. code-block:: javascript

    {
    "subsample_idxs_group_0": null, // not used
    "subsample_idxs_group_1": null, // not used
    "subsample_idxs_dataset": null, // not used
    "subsample_method": "none"      // no subsampling, use all neurons
    }

**Example 1**: 

Use the indices in the HDF5 dataset ``regions/idxs_lr/AUD_L``:

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group 
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "AUD_L",   // dataset name
    "subsample_method": "single"         // subsample, use single region
    }

**Example 2**: 

Fit separate decoders for each dataset of indices in the HDF5 group ``regions/idxs_lr``:

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "all",     // dataset name
    "subsample_method": "single"         // subsample, use single regions
    }

In this toy example, these options will fit 4 decoders, each using a different set of indices:
``AUD_R``, ``AUD_L``, ``VIS_L``, and ``VIS_R``.

.. note::
    
    At this time the option ``subsample_idxs_dataset`` can only accept a single string as an
    argument; therefore you can use ``all`` to fit decoders using all datasets in the specified
    index group, or you can specify a single dataset (e.g. ``AUD_L`` in this example). You cannot,
    for example, provide a list of strings.

**Example 3**: 

Use all indices *except* those in the HDF5 dataset ``regions/idxs_lr/AUD_L`` ("loo" stands for
"leave-one-out"):

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "AUD_L",   // dataset name
    "subsample_method": "loo"            // subsample, use all but specified region
    }

In this toy example, the combined neurons from ``AUD_R``, ``VIS_L`` and ``VIS_R`` would be used for
decoding (i.e. not the neurons in the specified region ``AUD_L``).

**Example 4**:

For each dataset in ``regions/indxs_lr``, fit a decoder that uses all indices *except* those in the
dataset:

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "all",     // dataset name
    "subsample_method": "loo"            // subsample, use all but specified region
    }

Again referring to the toy example, these options will fit 4 decoders, each using a different set
of indices:

1. ``AUD_L``, ``VIS_L``, and ``VIS_R`` (not ``AUD_R``)
2. ``AUD_R``, ``VIS_L``, and ``VIS_R`` (not ``AUD_L``)
3. ``AUD_R``, ``AUD_L``, and ``VIS_L`` (not ``VIS_R``)
4. ``AUD_R``, ``AUD_L``, and ``VIS_R`` (not ``VIS_L``)


.. _decoding_labels:

Decoding arbitrary covariates
-----------------------------
BehaveNet also uses the above decoding infrastructure to allow users to decode an arbitrary set of
labels from neural activity; these could be markers from pose estimation software, stimulus
information, or other task variables. In order to fit these models, the data HDF5 needs to be
augmented to include a new HDF5 group named ``labels``, which contains an HDF5 dataset for each
trial. See the :ref:`data structure documentation <data_structure_labels>` for more information.

Once the labels have been added to the data file, you can decode labels as you would CAE latents
above; the only changes that are necessary is the addition of the field ``n_labels`` in the data
json, and changing the model class in the model json from either ``neural-ae`` or ``neural-arhmm``
to ``neural-labels``.

.. note::

    The current BehaveNet implementation only allows for decoding continuous labels using a
    Gaussian noise distribution; support for binary and count data forthcoming.
