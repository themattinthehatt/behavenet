.. _data_structure:

########################
BehaveNet data structure
########################

In order to quickly and easily fit many models, BehaveNet uses a standardized data structure. "Raw" experimental data such as behavioral videos and (processed) neural data are stored in the `HDF5 file format <https://support.hdfgroup.org/HDF5/whatishdf5.html>`_. This file format can accomodate large and complex datasets, and is easy to work with thanks to a high-level `python API <https://www.h5py.org/>`_.

HDF is an acronym for Hierarchical Data Format, and one can think of it like a full directory structure inside of a single file. HDF5 "groups" are analogous to directories, while HDF5 "datasets" are analogous to files. The BehaveNet code uses up to 3 HDF5 groups: ``images``, ``masks`` (for masking images), and ``neural``. Each of these HDF5 groups contains multiple HDF5 datasets - one for each experimental trial. These datasets should have names that follow the pattern ``trial_%04i`` - datasets with more than 10000 trials are not currently supported with this naming convention.

BehaveNet models are trained on batches of data, which here are defined as one trial per batch. For datasets that do not have a trial structure (i.e. spontaneous behavior) we recommend splitting frames into arbitrarily defined "trials", the length of which should depend on the autocorrelation of the behavior (i.e. trials should not be shorter than the temporal extent of relevant behaviors). For the NP dataset in the original paper we used batch sizes of 1000 frames (~25 sec), and inserted additional gap trials between training, validation, and testing trials to minimize the possibility that good model performance was due to similarity of trials.

Below is a sample python script demonstrating how to create an HDF5 file with video data and neural data. Video data is assumed to be in a numpy array of shape (n_trials, n_frames, n_channels, y_pix, x_pix) and neural data is assumed to be in a numpy array of shape (n_trials, n_frames, n_neurons).

**Note 1**: for large experiments having all of this data in memory might be infeasible, and more sophisticated processing will be required

**Note 2**: BehaveNet does not require all trials to be of the same length, but does require that for each trial the shape of image/mask data is (n_frames, n_channels, y_pix, x_pix) (n_channels=1 for a single grayscale video) and the shape of neural data is (n_frames, n_neurons)

**Note 3**: the python package ``h5py`` is required for creating the HDF5 file, and is automatically installed with the BehaveNet package.

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
        for trial in range(n_trials):
            
            # create dataset in "image" group
            group_i.create_dataset('trial_%04i' % trial, data=images_np[trial], dtype='uint8')

            # create dataset in "neural" group
            group_n.create_dataset('trial_%04i' % trial, data=neural_np[trial], dtype='float32')

