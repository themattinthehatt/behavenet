Decoders
========

Coming soon!


.. _decoding_with_subsets:

Decoding with subsets of neurons
--------------------------------

Continuing with the toy dataset introduced in the :ref:`data structure<data_structure_subsets>` documentation, below are some examples for how to modify the decoding data json file to decode from user-specified groups of neurons:

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

In this toy example, these options will fit 4 decoders, each using a different set of indices: ``AUD_R``, ``AUD_L``, ``VIS_L``, and ``VIS_R``.

.. note::
    
    At this time the option ``subsample_idxs_dataset`` can only accept a single string as an argument; therefore you can use ``all`` to fit decoders using all datasets in the specified index group, or you can specify a single dataset (e.g. ``AUD_L`` in this example). You cannot, for example, provide a list of strings.

**Example 3**: 

Use all indices *except* those in the HDF5 dataset ``regions/idxs_lr/AUD_L`` ("loo" stands for "leave-one-out"):

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "AUD_L",   // dataset name
    "subsample_method": "loo"            // subsample, use all but specified region
    }

In this toy example, the combined neurons from ``AUD_R``, ``VIS_L`` and ``VIS_R`` would be used for decoding (i.e. not the neurons in the specified region ``AUD_L``).

**Example 3**: 

For each dataset in ``regions/indxs_lr``, fit a decoder that uses all indices *except* those in the dataset:

.. code-block:: javascript

    {
    "subsample_idxs_group_0": "regions", // top-level group
    "subsample_idxs_group_1": "idxs_lr", // second-level group
    "subsample_idxs_dataset": "all",     // dataset name
    "subsample_method": "loo"            // subsample, use all but specified region
    }

Again referring to the toy example, these options will fit 4 decoders, each using a different set of indices:

1. ``AUD_L``, ``VIS_L``, and ``VIS_R`` (not ``AUD_R``)
2. ``AUD_R``, ``VIS_L``, and ``VIS_R`` (not ``AUD_L``)
3. ``AUD_R``, ``AUD_L``, and ``VIS_L`` (not ``VIS_R``)
4. ``AUD_R``, ``AUD_L``, and ``VIS_R`` (not ``VIS_L``)

