.. _data_structure:

########################
BehaveNet data structure
########################

Introduction
============
In order to quickly and easily fit many models, BehaveNet uses a standardized data structure. "Raw"
experimental data such as behavioral videos and (processed) neural data are stored in the
`HDF5 file format <https://support.hdfgroup.org/HDF5/whatishdf5.html>`_. This file format can
accomodate large and complex datasets, and is easy to work with thanks to a high-level
`python API <https://www.h5py.org/>`_.

HDF is an acronym for Hierarchical Data Format, and one can think of it like a full directory
structure inside of a single file. HDF5 "groups" are analogous to directories, while HDF5
"datasets" are analogous to files. The BehaveNet code uses up to 3 HDF5 groups: ``images``,
``masks`` (for masking images), and ``neural``. Each of these HDF5 groups contains multiple HDF5
datasets - one for each experimental trial. These datasets should have names that follow the
pattern ``trial_%04i`` - datasets with more than 10000 trials are not currently supported with this
naming convention.

BehaveNet models are trained on batches of data, which here are defined as one trial per batch. For
datasets that do not have a trial structure (i.e. spontaneous behavior) we recommend splitting
frames into arbitrarily defined "trials", the length of which should depend on the autocorrelation
of the behavior (i.e. trials should not be shorter than the temporal extent of relevant behaviors).
For the NP dataset in the original paper we used batch sizes of 1000 frames (~25 sec), and inserted
additional gap trials between training, validation, and testing trials to minimize the possibility
that good model performance was due to similarity of trials.

Below is a sample python script demonstrating how to create an HDF5 file with video data and neural
data. Video data is assumed to be in a list, where each list element corresponds to a single trial,
and is a numpy array of shape (n_frames, n_channels, y_pix, x_pix). Neural data is assumed to be in
the same format; a corresponding list of numpy arrays of shape (n_frames, n_neurons). BehaveNet
does not require all trials to be of the same length, but does require that for each trial the
images and neural activity have the same number of frames. This may require you to interpolate/bin
video or neural data differently than the rate at which it was acquired.

**Notes**:

* for large experiments, having all of this (video) data in memory to create the HDF5 file might be infeasible, and more sophisticated processing will be required
* neural data is only required for fitting decoding models; it is still possible to fit autoencoders and ARHMMs when the HDF5 file only contains images
* masks should be the same size as images; a value of 0 excludes the pixel from the loss function, a value of 1 includes it
* the python package ``h5py`` is required for creating the HDF5 file, and is automatically installed with the BehaveNet package.

.. code-block:: python

    import h5py
    
    # assume images are in an np array named "images_np"; this should be a an array of dtype
    # 'uint8', and values should be between 0 and 255. The data type is converted to float
    # and values are divided by 255 during the data loading process.
    
    # assume neural activity is in an np array named "neural_np"; this can be spike count data
    # or continuous-valued data for e.g. calcium imaging experiments

    hdf5_file = '/path/to/data.hdf5'  # path needs to exist, but not 'data.hdf5' file
    
    with h5py.File(hdf5_file, 'w', libver='latest', swmr=True) as f:

        # enable single write, multi-read - needed for simultaneous model fitting
        f.swmr_mode = True  

        # create "image" HDF5 group
        group_i = f.create_group('images')

        # create "neural" HDF5 group
        group_n = f.create_group('neural')

        # create a dataset for each trial within groups
        for trial in range(len(images_np)):
            
            # create dataset in "image" group
            # images_np[trial] should be of shape (n_frames, n_channels, y_pix, x_pix)
            group_i.create_dataset('trial_%04i' % trial, data=images_np[trial], dtype='uint8')

            # create dataset in "neural" group
            # neural_np[trial] should be of shape (n_frames, n_neurons)
            group_n.create_dataset('trial_%04i' % trial, data=neural_np[trial], dtype='float32')

A more in-depth example can be found in the function :func:`behavenet.data.preprocess.build_hdf5`.

.. _data_structure_subsets:

Identifying subsets of neurons
==============================
It is possible that the neural data used for encoding and decoding models will have natural partitions - for example, neurons belonging to different brain regions or cell types. In this case you may be interested in, say, decoding behavior from each brain region individually, as well as all together. BehaveNet provides this capability through the addition of another HDF5 group. This group can have any name, but for illustration purposes we will use the name "regions" (this name will be later be provided in the updated data json file).

The "regions" group contains a second level of (again user-defined) groups, which will define different index groupings. As a concrete example, let's say we have neural data with 100 neurons:

* indices 00-24 are neurons in left auditory cortex
* indices 25-49 are neurons in right auditory cortex
* indices 50-74 are neurons in left visual cortex
* indices 75-99 are neurons in right visual cortex

We will define this "grouping" of indices in a python dict:

.. code-block:: python

    neural_idxs_lr = {
        'AUD_L': np.arange(0, 25),
        'AUD_R': np.arange(25, 50),
        'VIS_L': np.arange(50, 75),
        'VIS_R': np.arange(75, 100)
    }

We can also define another "grouping" of indices that ignores hemisphere information:

.. code-block:: python 

    neural_idxs = {
        'AUD': np.arange(0, 50),
        'VIS': np.arange(50, 100)
    }

We can then store these indices in the data HDF5 by modifying the above script:

.. code-block:: python

    ...

    # create "neural" HDF5 group
    group_n = f.create_group('neural')

    # create "regions" HDF5 group
    group_r0 = f.create_group('regions')

    # create "idxs_lr" HDF5 group inside the "regions" group
    group_r1a = group_r0.create_group('idxs_lr')
    # insert the index info into datasets inside the regions/idxs_lr group
    for region_name, region_idxs in neural_idxs_lr.items():
        group_r1a.create_dataset(region_name, data=region_idxs)

    # create "idxs" HDF5 group inside the "regions" group
    group_r1b = group_r0.create_group('idxs')
    # insert the index info into datasets inside the regions/idxs group
    for region_name, region_idxs in neural_idxs.items():
        group_r1b.create_dataset(region_name, data=region_idxs)
    
    # create a dataset for each trial within groups
    for trial in range(len(images_np)):
    
    ...

This HDF5 file will now have the following addtional datasets:

* regions/idxs_lr/AUD_L
* regions/idxs_lr/AUD_R
* regions/idxs_lr/VIS_L
* regions/idxs_lr/VIS_R
* regions/idxs/AUD
* regions/idxs/VIS

Just as the top-level group (here named "regions") can have an arbitrary name (later specified in
the data json file), the second-level groups (here named "idxs_lr" and "idxs") can also have
arbitrary names, and there can be any number of them, as long as the datasets within them contain
valid indices into the neural data. The specific set of indices used for any analyses will be
specified in the data json file. See the :ref:`decoding documentation<decoding_with_subsets>` for
an example of how to decode behavior using specified subsets of neurons.


.. _data_structure_labels:

Including labels for ARHMMs and conditional autoencoders
========================================================
In order to fit :ref:`conditional autoencoder models<conditional_aes>`, you will need to include
additional information about labels in the HDF5 file. These labels can be outputs from pose
estimation software, or other behavior-related signals such as pupil diameter or lick times. These
labels should be stored in an HDF5 group named ``labels``. As before, the ``labels`` group contains
multiple HDF5 datasets - one for each experimental trial. These datasets should also follow the
pattern ``trial_%04i``, and match the image data in the corresponding image dataset
``images/trial_%04i``. If the image data in a given trial is of shape
(n_frames, n_channels, y_pix, x_pix), then the corresponding label data should be of shape
(n_frames, n_markers). Note that, when using pose estimation software, each marker has an x- and
y-coordinate, so tracking four body parts will result in an 8-dimensional set of labels.

It is also possible to fit ARHMMs directly to labels rather than the outputs of an autoencoder. In
this case ``labels`` is the only necessary HDF5 group, though including a corresponding ``images``
group will allow you to utilize more of the ARHMM visualization tools. To fit an ARHMM on label
data, you simply need to change the ``model_class`` entry of the arhmm model json from ``arhmm`` to
``arhmm-labels`` (see the json config ``arhmm_labels_model.json``).


.. note::
    
    The matrix subspace projection model implemented in BehaveNet learns a linear mapping from the
    original latent space to the predicted labels that **does not contain a bias term**. Therefore
    you should center each label before adding them to the HDF5 file. Additionally, normalizing
    each label by its standard deviation can make searching across msp weights less dependent on
    the size of the input image.
