"""Classes for splitting and serving data to models.

The data generator classes contained in this module inherit from the
:class:`torch.utils.data.Dataset` class. The user-facing class is the
:class:`ConcatSessionsGenerator`, which can manage one or more datasets. Each dataset is composed
of trials, which are split into training, validation, and testing trials using the
:func:`split_trials`. The default data generator can handle the following data types:

* **images**: individual frames of the behavioral video
* **masks**: binary mask for each frame
* **labels**: i.e. DLC labels
* **neural activity**
* **AE latents**
* **AE predictions**: predictions of AE latents from neural activity
* **ARHMM states**
* **ARHMM predictions**: predictions of ARHMM states from neural activity

Please see the online documentation at
`Read the Docs <https://behavenet.readthedocs.io/en/latest/index.html>`_ for detailed examples of
how to use the data generators.

"""

from collections import OrderedDict
import h5py
import numpy as np
import os
import pickle
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler


__all__ = [
    'split_trials',
    'SingleSessionDatasetBatchedLoad',
    'SingleSessionDataset',
    'ConcatSessionsGenerator',
    'ConcatSessionsGeneratorMulti']


def split_trials(n_trials, rng_seed=0, train_tr=8, val_tr=1, test_tr=1, gap_tr=0):
    """Split trials into train/val/test blocks.

    The data is split into blocks that have gap trials between tr/val/test:

    :obj:`train tr | gap tr | val tr | gap tr | test tr | gap tr`

    Parameters
    ----------
    n_trials : :obj:`int`
        total number of trials to be split
    rng_seed : :obj:`int`, optional
        random seed for reproducibility
    train_tr : :obj:`int`, optional
        number of train trials per block
    val_tr : :obj:`int`, optional
        number of validation trials per block
    test_tr : :obj:`int`, optional
        number of test trials per block
    gap_tr : :obj:`int`, optional
        number of gap trials between tr/val/test; there will be a total of 3 * `gap_tr` gap trials
        per block; can be zero if no gap trials are desired.

    Returns
    -------
    :obj:`dict`
        Split trial indices are stored in a dict with keys `train`, `test`, and `val`

    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = train_tr + gap_tr + val_tr + gap_tr + test_tr + gap_tr

    n_blocks = int(np.floor(n_trials / tr_per_block))
    if n_blocks == 0:
        raise ValueError(
            'Not enough trials (n=%i) for the train/test/val/gap values %i/%i/%i/%i' %
            (n_trials, train_tr, val_tr, test_tr, gap_tr))

    leftover_trials = n_trials - tr_per_block * n_blocks
    if leftover_trials > 0:
        offset = np.random.randint(0, high=leftover_trials)
    else:
        offset = 0
    idxs_block = np.random.permutation(n_blocks)

    batch_idxs = {'train': [], 'test': [], 'val': []}
    for block in idxs_block:

        curr_tr = block * tr_per_block + offset
        batch_idxs['train'].append(np.arange(curr_tr, curr_tr + train_tr))
        curr_tr += (train_tr + gap_tr)
        batch_idxs['val'].append(np.arange(curr_tr, curr_tr + val_tr))
        curr_tr += (val_tr + gap_tr)
        batch_idxs['test'].append(np.arange(curr_tr, curr_tr + test_tr))

    for dtype in ['train', 'val', 'test']:
        batch_idxs[dtype] = np.concatenate(batch_idxs[dtype], axis=0)

    return batch_idxs


def _load_pkl_dict(path, key, idx=None, dtype='float32'):
    """Helper function to load pickled data.

    Parameters
    ----------
    path : :obj:`str`
        full file name including `.pkl` extention
    key : :obj:`str`
        data is returned from this key of the pickled dictionary
    idx : :obj:`int` or :obj:`NoneType`
        if :obj:`NoneType` return all data, else return data from this index
    dtype : :obj:`str`
        numpy data type of data

    Returns
    -------
    :obj:`list` of :obj:`numpy.ndarray` if :obj:`idx=None`
    :obj:`numpy.ndarray` is :obj:`idx=int`

    """
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)

    if idx is None:
        samp = [data.astype(dtype) for data in data_dict[key]]
    else:
        samp = [data_dict[key][idx].astype(dtype)]

    return samp


class SingleSessionDatasetBatchedLoad(data.Dataset):
    """Dataset class for a single session with batch loading of data."""

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='', signals=None, transforms=None,
            paths=None, device='cpu', as_numpy=False):
        """

        Parameters
        ----------
        data_dir : :obj:`str`
            root directory of data
        lab : :obj:`str`
            lab id
        expt : :obj:`str`
            expt id
        animal : :obj:`str`
            animal id
        session : :obj:`str`
            session id
        signals : :obj:`list` of :obj:`str`
            e.g. 'images' | 'masks' | 'neural' | .... See
            :func:`behavenet.fitting.utils.get_data_generator_inputs` for examples.
        transforms : :obj:`list` of :obj:`behavenet.data.transform` objects
            each element corresponds to an entry in :obj:`signals`; for multiple transforms, chain
            together using :obj:`behavenet.data.transform.Compose` class. See
            :mod:`behavenet.data.transforms` for available transform options.
        paths : :obj:`list` of :obj:`str`
            each element corresponds to an entry in :obj:`signals`; filename (using absolute path)
            of data
        device : :obj:`str`, optional
            location of data; options are :obj:`cpu | cuda`
        as_numpy : bool
            if :obj:`True` return data as a numpy array, else return as a torch tensor

        """

        # specify data
        self.lab = lab
        self.expt = expt
        self.animal = animal
        self.session = session
        self.data_dir = os.path.join(
            data_dir, self.lab, self.expt, self.animal, self.session)
        self.name = os.path.join(self.lab, self.expt, self.animal, self.session)
        self.sess_str = str('%s_%s_%s_%s' % (self.lab, self.expt, self.animal, self.session))

        # get data paths
        self.signals = signals
        self.transforms = OrderedDict()
        self.paths = OrderedDict()
        for signal, transform, path in zip(signals, transforms, paths):
            self.transforms[signal] = transform
            self.paths[signal] = path

        # get total number of trials by loading images/neural data
        self.n_trials = None
        for i, signal in enumerate(signals):
            if signal == 'images' or signal == 'neural' or signal == 'labels' or \
                    signal == 'labels_sc' or signal == 'labels_masks':
                data_file = paths[i]
                with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
                    self.n_trials = len(f[signal])
                    break
            elif signal == 'ae_latents':
                try:
                    latents = _load_pkl_dict(self.paths[signal], 'latents')
                except FileNotFoundError:
                    raise NotImplementedError(
                        ('Could not open %s\nMust create ae latents from model;' +
                         ' currently not implemented') % self.paths[signal])

                self.n_trials = len(latents)

        # meta data about train/test/xv splits; set by ConcatSessionsGenerator
        self.batch_idxs = None
        self.n_batches = None

        self.device = device
        self.as_numpy = as_numpy

    def __str__(self):
        """Pretty printing of dataset info"""
        format_str = str('%s\n' % self.sess_str)
        format_str += str('    signals: {}\n'.format(self.signals))
        format_str += str('    transforms: {}\n'.format(self.transforms))
        format_str += str('    paths: {}\n'.format(self.paths))
        return format_str

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        """Return batch of data; if idx is None, return all data

        Parameters
        ----------
        idx : :obj:`int` or :obj:`NoneType`
            trial index to load; if :obj:`NoneType`, return all data.

        Returns
        -------
        :obj:`dict`
            data sample

        """

        if idx is None and not self.as_numpy:
            raise NotImplementedError('Cannot currently load all data as torch tensors')

        sample = OrderedDict()
        for signal in self.signals:

            # index correct trial
            if signal == 'images':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if idx is None:
                        print('Warning: loading all images!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype) / 255)
                        sample[signal] = temp_data
                    else:
                        sample[signal] = [f[signal][str(
                            'trial_%04i' % idx)][()].astype(dtype) / 255]

            elif signal == 'masks':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if idx is None:
                        print('Warning: loading all masks!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype))
                        sample[signal] = temp_data
                    else:
                        sample[signal] = f[signal][str('trial_%04i' % idx)][()].astype(dtype)

            elif signal == 'neural' or signal == 'labels' or signal == 'labels_sc' \
                    or signal == 'labels_masks':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if idx is None:
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype))
                        sample[signal] = temp_data
                    else:
                        sample[signal] = [f[signal][str('trial_%04i' % idx)][()].astype(dtype)]

            elif signal == 'ae_latents' or signal == 'latents':
                dtype = 'float32'
                sample[signal] = self._try_to_load(signal, key='latents', idx=idx, dtype=dtype)

            elif signal == 'ae_predictions':
                dtype = 'float32'
                sample[signal] = self._try_to_load(signal, key='predictions', idx=idx, dtype=dtype)

            elif signal == 'arhmm' or signal == 'arhmm_states':
                dtype = 'int32'
                sample[signal] = self._try_to_load(signal, key='states', idx=idx, dtype=dtype)

            elif signal == 'arhmm_predictions':
                dtype = 'float32'
                sample[signal] = self._try_to_load(signal, key='predictions', idx=idx, dtype=dtype)

            else:
                raise ValueError('"%s" is an invalid signal type' % signal)

            # apply transforms
            if self.transforms[signal]:
                sample[signal] = [self.transforms[signal](samp) for samp in sample[signal]]

            # transform into tensor
            if not self.as_numpy:
                if dtype == 'float32':
                    sample[signal] = torch.from_numpy(sample[signal][0]).float()
                else:
                    sample[signal] = torch.from_numpy(sample[signal][0]).long()

        sample['batch_idx'] = idx

        return sample

    def _try_to_load(self, signal, key, idx, dtype):
        # try:
        #     data = _load_pkl_dict(self.paths[signal], key, idx=idx, dtype=dtype)
        # except FileNotFoundError:
        #     # try prepending session string
        #     try:
        #         self.paths[signal] = _prepend_sess_id(self.paths[signal], self.sess_str)
        #         data = _load_pkl_dict(self.paths[signal], key, idx=idx, dtype=dtype)
        #     except FileNotFoundError:
        #         raise NotImplementedError(
        #             ('Could not open %s\nMust create %s from model;' +
        #              ' currently not implemented') % (self.paths[signal], key))
        try:
            data = _load_pkl_dict(self.paths[signal], key, idx=idx, dtype=dtype)
        except FileNotFoundError:
            raise NotImplementedError(
                ('Could not open %s\nMust create %s from model;' +
                 ' currently not implemented') % (self.paths[signal], key))
        return data


class SingleSessionDataset(SingleSessionDatasetBatchedLoad):
    """Dataset class for a single session.

    Loads all data during Dataset creation and saves as an attribute. Batches are then sampled from
    this stored data. All data transformations are applied to the full dataset upon load, *not*
    for each batch. This automatically returns data as lists of numpy arrays.

    Note
    ----
    This data loader cannot be used to fit pytorch models, only ssm models.

    """

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='', signals=None, transforms=None,
            paths=None, device='cuda', as_numpy=False):
        """

        Parameters
        ----------
        data_dir : :obj:`str`
            root directory of data
        lab : :obj:`str`
            lab id
        expt : :obj:`str`
            expt id
        animal : :obj:`str`
            animal id
        session : :obj:`str`
            session id
        signals : :obj:`list` of :obj:`str`
            e.g. 'images' | 'masks' | 'neural' | .... See
            :func:`behavenet.fitting.utils.get_data_generator_inputs` for examples.
        transforms : :obj:`list` of :obj:`behavenet.data.transform` objects
            each element corresponds to an entry in :obj:`signals`; for multiple transforms, chain
            together using :obj:`behavenet.data.transform.Compose` class. See
            :mod:`behavenet.data.transforms` for available transform options.
        paths : :obj:`list` of :obj:`str`
            each element corresponds to an entry in :obj:`signals`; filename (using absolute path)
            of data
        device : :obj:`str`, optional
            location of data; options are :obj:`cpu | cuda`

        """

        super().__init__(data_dir, lab, expt, animal, session, signals, transforms, paths, device)

        # grab all data as a single batch
        self.as_numpy = as_numpy
        self.data = super(SingleSessionDataset, self).__getitem__(idx=None)
        _ = self.data.pop('batch_idx')

        # collect dims for easy reference
        # self.dims = OrderedDict()
        # for signal, data in self.data.items():
        #     self.dims[signal] = data.shape

        # if self.n_trials is None:
        #     self.n_trials = self.dims[signal][0]

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        """Return batch of data.

        Parameters
        ----------
        idx : :obj:`int` or :obj:`NoneType`
            trial index to load; if :obj:`NoneType`, return all data.

        Returns
        -------
        :obj:`dict`
            data sample

        """

        sample = OrderedDict()
        for signal in self.signals:
            sample[signal] = [self.data[signal][idx]]

        sample['batch_idx'] = idx
        return sample


class ConcatSessionsGenerator(object):
    """Dataset class for multiple sessions.

    This class contains a list of single session data generators. It handles shuffling and
    iterating over these sessions.
    """

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids_list, signals_list=None, transforms_list=None, paths_list=None,
            device='cuda', as_numpy=False, batch_load=True, rng_seed=0, trial_splits=None,
            train_frac=1.0):
        """

        Parameters
        ----------
        data_dir : :obj:`str`
            root directory of data
        ids_list : :obj:`list` of :obj:`dict`
            each element has the following keys: 'lab', 'expt', 'animal', and 'session'; the data
            (images, masks, neural activity) is assumed to be located in:
            :obj:`data_dir/lab/expt/animal/session/data.hdf5`
        signals_list : :obj:`list` of :obj:`list`
            list of signals for each session
        transforms_list : :obj:`list` of :obj:`list`
            list of transforms for each session
        paths_list : :obj:`list` of :obj:`list`
            list of paths for each session
        device : :obj:`str`, optional
            location of data; options are :obj:`cpu | cuda`
        as_numpy : bool, optional
            if :obj:`True` return data as a numpy array, else return as a torch tensor
        batch_load : :obj:`bool`, optional
            :obj:`True` to load data one batch at a time, :obj:`False` to load all data at once and
            store in memory (data is still served one trial at a time).
        rng_seed : :obj:`int`, optional
            controls split of train/val/test trials
        trial_splits : :obj:`dict`, optional
            determines number of train/val/test trials using the keys 'train_tr', 'val_tr',
            'test_tr', and 'gap_tr'; see :func:`split_trials` for how these are used.
        train_frac : :obj:`float`, optional
            if :obj:`0 < train_frac < 1.0`, defines the fraction of assigned training trials to
            actually use; if :obj:`train_frac > 1.0`, defines the number of assigned training
            trials to actually use

        """
        if isinstance(ids_list, dict):
            ids_list = [ids_list]
        self.ids = ids_list
        self.as_numpy = as_numpy
        self.device = device

        self.batch_load = batch_load
        if self.batch_load:
            SingleSession = SingleSessionDatasetBatchedLoad
        else:
            SingleSession = SingleSessionDataset

        self.datasets = []
        self.datasets_info = []

        self.signals = signals_list
        self.transforms = transforms_list
        self.paths = paths_list
        for ids, signals, transforms, paths in zip(
                ids_list, signals_list, transforms_list, paths_list):
            self.datasets.append(SingleSession(
                data_dir, lab=ids['lab'], expt=ids['expt'], animal=ids['animal'],
                session=ids['session'], signals=signals, transforms=transforms, paths=paths,
                device=device, as_numpy=self.as_numpy))
            self.datasets_info.append({
                'lab': ids['lab'], 'expt': ids['expt'], 'animal': ids['animal'],
                'session': ids['session']})

        # collect info about datasets
        self.n_datasets = len(self.datasets)

        # get train/val/test batch indices for each dataset
        if trial_splits is None:
            trial_splits = {'train_tr': 8, 'val_tr': 1, 'test_tr': 1, 'gap_tr': 0}
        self.batch_ratios = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            dataset.batch_idxs = split_trials(len(dataset), rng_seed=rng_seed, **trial_splits)
            dataset.n_batches = {}
            for dtype in self._dtypes:
                if dtype == 'train':
                    # subsample training data if requested
                    if train_frac != 1.0:
                        n_batches = len(dataset.batch_idxs[dtype])
                        if train_frac < 1.0:
                            # subsample as fraction of total batches
                            n_idxs = int(np.floor(train_frac * n_batches))
                            if n_idxs <= 0:
                                print(
                                    'warning: attempting to use invalid number of training ' +
                                    'batches; defaulting to all training batches')
                                n_idxs = n_batches
                        else:
                            # subsample fixed number of batches
                            train_frac = n_batches if train_frac > n_batches else train_frac
                            n_idxs = int(train_frac)
                        idxs_rand = np.random.choice(n_batches, size=n_idxs, replace=False)
                        dataset.batch_idxs[dtype] = dataset.batch_idxs[dtype][idxs_rand]
                    self.batch_ratios[i] = len(dataset.batch_idxs[dtype])
                dataset.n_batches[dtype] = len(dataset.batch_idxs[dtype])
        self.batch_ratios = np.array(self.batch_ratios) / np.sum(self.batch_ratios)

        # find total number of batches per data type; this will be iterated over in the train loop
        self.n_tot_batches = {}
        for dtype in self._dtypes:
            self.n_tot_batches[dtype] = np.sum(
                [dataset.n_batches[dtype] for dataset in self.datasets])

        # create data loaders (will shuffle/batch/etc datasets)
        self.dataset_loaders = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            self.dataset_loaders[i] = {}
            for dtype in self._dtypes:
                self.dataset_loaders[i][dtype] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=SubsetRandomSampler(dataset.batch_idxs[dtype]),
                    num_workers=0,
                    pin_memory=False)

        # create all iterators (will iterate through data loaders)
        self.dataset_iters = [None] * self.n_datasets
        for i in range(self.n_datasets):
            self.dataset_iters[i] = {}
            for dtype in self._dtypes:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def __str__(self):
        """Pretty printing of dataset info"""
        if self.batch_load:
            dataset_type = 'SingleSessionDatasetBatchedLoad'
        else:
            dataset_type = 'SingleSessionDataset'
        format_str = str('Generator contains %i %s objects:\n' % (self.n_datasets, dataset_type))
        for dataset in self.datasets:
            format_str += dataset.__str__()
        return format_str

    def __len__(self):
        return self.n_datasets

    def reset_iterators(self, dtype):
        """Reset iterators so that all data is available.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test' | 'all'

        """

        for i in range(self.n_datasets):
            if dtype == 'all':
                for dtype_ in self._dtypes:
                    self.dataset_iters[i][dtype_] = iter(self.dataset_loaders[i][dtype_])
            else:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def next_batch(self, dtype):
        """Return next batch of data.

        The data generator iterates randomly through sessions and trials. Once a session runs out
        of trials it is skipped.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'

        Returns
        -------
        :obj:`tuple`
            - **sample** (:obj:`dict`): data batch with keys given by :obj:`signals` input to class
            - **dataset** (:obj:`int`): dataset from which data batch is drawn

        """
        while True:
            # get next session
            dataset = int(np.random.choice(np.arange(self.n_datasets), p=self.batch_ratios))

            # get this session data
            try:
                sample = next(self.dataset_iters[dataset][dtype])
                break
            except StopIteration:
                continue

        if self.as_numpy:
            for i, signal in enumerate(sample):
                if signal != 'batch_idx':
                    sample[signal] = [ss.cpu().detach().numpy() for ss in sample[signal]]
        else:
            if self.device == 'cuda':
                sample = {key: val.to('cuda') for key, val in sample.items()}

        return sample, dataset


class ConcatSessionsGeneratorMulti(ConcatSessionsGenerator):
    """Dataset class for multiple sessions, which returns multiple sessions per training batch.

    This class contains a list of single session data generators. It handles shuffling and
    iterating over these sessions.
    """

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids_list, signals_list=None, transforms_list=None, paths_list=None,
            device='cuda', as_numpy=False, batch_load=True, rng_seed=0, trial_splits=None,
            train_frac=1.0, n_sessions_per_batch=2):
        """

        Parameters
        ----------
        data_dir : :obj:`str`
            root directory of data
        ids_list : :obj:`list` of :obj:`dict`
            each element has the following keys: 'lab', 'expt', 'animal', and 'session'; the data
            (images, masks, neural activity) is assumed to be located in:
            :obj:`data_dir/lab/expt/animal/session/data.hdf5`
        signals_list : :obj:`list` of :obj:`list`
            list of signals for each session
        transforms_list : :obj:`list` of :obj:`list`
            list of transforms for each session
        paths_list : :obj:`list` of :obj:`list`
            list of paths for each session
        device : :obj:`str`, optional
            location of data; options are :obj:`cpu | cuda`
        as_numpy : bool, optional
            if :obj:`True` return data as a numpy array, else return as a torch tensor
        batch_load : :obj:`bool`, optional
            :obj:`True` to load data one batch at a time, :obj:`False` to load all data at once and
            store in memory (data is still served one trial at a time).
        rng_seed : :obj:`int`, optional
            controls split of train/val/test trials
        trial_splits : :obj:`dict`, optional
            determines number of train/val/test trials using the keys 'train_tr', 'val_tr',
            'test_tr', and 'gap_tr'; see :func:`split_trials` for how these are used.
        train_frac : :obj:`float`, optional
            if :obj:`0 < train_frac < 1.0`, defines the fraction of assigned training trials to
            actually use; if :obj:`train_frac > 1.0`, defines the number of assigned training
            trials to actually use
        n_sessions_per_batch : :obj:`int`, optional
            number of session per training batch to serve model; the combination of datasets and
            batches will be shuffled when the data iterator is reset

        """

        if n_sessions_per_batch > 4:
            # requires more implementation in behavenet.fitting.losses.triplet_loss()
            raise NotImplementedError
        self.n_sessions_per_batch = n_sessions_per_batch

        super().__init__(
            data_dir, ids_list, signals_list=signals_list, transforms_list=transforms_list,
            paths_list=paths_list, device=device, as_numpy=as_numpy, batch_load=batch_load,
            rng_seed=rng_seed, trial_splits=trial_splits, train_frac=train_frac)

        # redefine total number of training batches to reflect the fact that multiple batches are
        # served per iteration (but only for training data)
        self.n_tot_batches['train'] = int(self.n_tot_batches['train'] / n_sessions_per_batch)

    def __str__(self):
        """Pretty printing of dataset info"""
        if self.batch_load:
            dataset_type = 'SingleSessionDatasetBatchedLoad'
        else:
            dataset_type = 'SingleSessionDataset'
        format_str = 'MultiGenerator contains %i %s objects:\n' % (self.n_datasets, dataset_type)
        for dataset in self.datasets:
            format_str += dataset.__str__()
        return format_str

    def __len__(self):
        return self.n_datasets

    def next_batch(self, dtype, return_multiple=True):
        """Return next batch of data.

        The data generator iterates randomly through sessions and trials. Once a session runs out
        of trials it is skipped.

        Parameters
        ----------
        dtype : :obj:`str`
            'train' | 'val' | 'test'
        return_multiple : :obj:`bool`
            True to return multiple batches for train data

        Returns
        -------
        :obj:`tuple`
            - **samples** (:obj:`dict`): data batch with keys given by :obj:`signals` input to class
            - **datasets** (:obj:`int`): dataset from which data batch is drawn

        """

        def renormalize(array):
            if np.sum(array) == 0:
                return array
            else:
                return array / np.sum(array)

        if dtype == 'train' and return_multiple:

            samples = []
            datasets = []

            curr_batch_ratios = np.copy(self.batch_ratios)

            for sess in range(self.n_sessions_per_batch):

                while True:

                    # check to see if there are enough available batches
                    if np.sum(curr_batch_ratios > 0) < (self.n_sessions_per_batch - sess):
                        return None, None

                    # get next dataset
                    dataset = np.random.choice(np.arange(self.n_datasets), p=curr_batch_ratios)

                    # don't choose this dataset in the future
                    curr_batch_ratios[dataset] = 0
                    curr_batch_ratios = renormalize(curr_batch_ratios)

                    # get this session data
                    try:
                        sample = next(self.dataset_iters[dataset][dtype])
                        break
                    except StopIteration:
                        continue

                if self.as_numpy:
                    raise NotImplementedError
                    # for i, signal in enumerate(sample):
                    #     if signal != 'batch_idx':
                    #         sample[signal] = [ss.cpu().detach().numpy() for ss in sample[signal]]
                else:
                    if self.device == 'cuda':
                        sample = {key: val.to('cuda') for key, val in sample.items()}

                samples.append(sample)
                datasets.append(dataset)

            # print(datasets)
            # print([s['batch_idx'].item() for s in samples])

        else:

            while True:
                # get next session
                dataset = np.random.choice(np.arange(self.n_datasets), p=self.batch_ratios)

                # get this session data
                try:
                    sample = next(self.dataset_iters[dataset][dtype])
                    break
                except StopIteration:
                    continue

            if self.as_numpy:
                for i, signal in enumerate(sample):
                    if signal != 'batch_idx':
                        sample[signal] = [ss.cpu().detach().numpy() for ss in sample[signal]]
            else:
                if self.device == 'cuda':
                    sample = {key: val.to('cuda') for key, val in sample.items()}

            datasets = int(dataset)
            samples = sample

        return samples, datasets
