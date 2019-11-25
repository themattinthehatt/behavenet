import os
import numpy as np
import pickle
from collections import OrderedDict
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import h5py

# TODO: rename indx to idx when updating documentation


def split_trials(n_trials, rng_seed=0, train_tr=8, val_tr=1, test_tr=1, gap_tr=0):
    """
    Split trials into train/val/test blocks.

    The data is split into blocks that have gap trials between tr/val/test:
    train tr | gap tr | val tr | gap tr | test tr | gap tr

    Args:
        n_trials (int): number of trials to use in the split
        rng_seed (int): numpy random seed for reproducibility
        train_tr (int): number of train trials per block
        val_tr (int): number of validation trials per block
        test_tr (int): number of test trials per block
        gap_tr (int): number of gap trials between tr/val/test; there will be
            a total of 3 * `gap_tr` gap trials per block

    Returns:
        (dict)
    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = train_tr + gap_tr + val_tr + gap_tr + test_tr + gap_tr

    n_blocks = int(np.floor(n_trials / tr_per_block))
    leftover_trials = n_trials - tr_per_block * n_blocks
    if leftover_trials > 0:
        offset = np.random.randint(0, high=leftover_trials)
    else:
        offset = 0
    indxs_block = np.random.permutation(n_blocks)

    batch_indxs = {'train': [], 'test': [], 'val': []}
    for block in indxs_block:

        curr_tr = block * tr_per_block + offset
        batch_indxs['train'].append(np.arange(curr_tr, curr_tr + train_tr))
        curr_tr += (train_tr + gap_tr)
        batch_indxs['val'].append(np.arange(curr_tr, curr_tr + val_tr))
        curr_tr += (val_tr + gap_tr)
        batch_indxs['test'].append(np.arange(curr_tr, curr_tr + test_tr))

    for dtype in ['train', 'val', 'test']:
        batch_indxs[dtype] = np.concatenate(batch_indxs[dtype], axis=0)

    return batch_indxs


def load_pkl_dict(path, key, indx=None, dtype='float32'):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    if indx is None:
        samp = [data.astype(dtype) for data in data_dict[key]]
    else:
        samp = data_dict[key][indx].astype(dtype)
    return samp


def prepend_sess_id(path, sess_str):
    pathname = os.path.dirname(path)
    filename = os.path.basename(path)
    return os.path.join(pathname, str('%s_%s' % (sess_str, filename)))


class SingleSessionDatasetBatchedLoad(data.Dataset):
    """
    Dataset class for a single session

    Loads data one batch at a time; data transformations are applied to each batch upon load.
    """

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='', signals=None, transforms=None,
            paths=None, device='cpu'):
        """
        Args:
            data_dir (str): root directory of data
            lab (str)
            expt (str)
            animal (str)
            session (str)
            signals (list of strs): e.g. 'images' | 'masks' | 'neural' | ...
                see behavenet.fitting.utils.get_data_generator_inputs for
                examples
            transforms (list of transforms): each element corresponds to an
                entry in `signals`; for multiple transforms, chain together
                using pt transforms.Compose; see behavenet.data.transforms.py
                for available transform options
            paths (list of strs): each element corresponds to file
                location for an entry in `signals`; see
                behavenet.fitting.utils.get_data_generator_inputs for examples
            device (str, optional): location of data
                'cpu' | 'cuda'
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
            if signal == 'images' or signal == 'neural':
                data_file = paths[i]
                with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
                    self.n_trials = len(f[signal])
                    break
            elif signal == 'ae_latents':
                try:
                    latents = load_pkl_dict(self.paths[signal], 'latents')
                except FileNotFoundError:
                    # try prepending session string
                    try:
                        latents = load_pkl_dict(
                            prepend_sess_id(self.paths[signal], self.sess_str), 'latents')
                    except FileNotFoundError:
                        raise NotImplementedError(
                            ('Could not open %s\nMust create ae latents from model;' +
                             ' currently not implemented') % self.paths[signal])
                self.n_trials = len(latents)

        # meta data about train/test/xv splits; set by ConcatSessionsGenerator
        self.batch_indxs = None
        self.n_batches = None

        self.device = device

        # # TODO: not all signals are stored in hdf5 file
        # self.dims = OrderedDict()
        # for signal in self.signals:
        #     if signal in f:
        #         key_list = list(f[signal].keys())
        #         self.dims[signal] = f[signal][key_list[0]].shape
        #     else:
        #         self.dims[signal] = []

    def __str__(self):
        format_str = str('%s\n' % self.sess_str)
        format_str += str('    signals: {}\n'.format(self.signals))
        format_str += str('    transforms: {}\n'.format(self.transforms))
        format_str += str('    paths: {}\n'.format(self.paths))
        return format_str

    def __len__(self):
        return self.n_trials

    def __getitem__(self, indx):
        """
        Return batch of data; if indx is None, return all data

        Args:
            indx (int or NoneType): trial index

        Returns:
            (dict): data sample
        """

        sample = OrderedDict()
        for signal in self.signals:

            # index correct trial
            if signal == 'images':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        print('Warning: loading all images!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][str(
                            'trial_%04i' % indx)][()].astype(dtype)
                # normalize to range [0, 1]
                sample[signal] /= 255.0

            elif signal == 'masks':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        print('Warning: loading all masks!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][str('trial_%04i' % indx)][()].astype(dtype)

            elif signal == 'neural':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][str(
                                'trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][str(
                            'trial_%04i' % indx)][()].astype(dtype)

            elif signal == 'ae_latents':
                dtype = 'float32'
                sample[signal] = self.try_to_load(
                    signal, key='latents', indx=indx, dtype=dtype)

            elif signal == 'ae_predictions':
                dtype = 'float32'
                sample[signal] = self.try_to_load(
                    signal, key='predictions', indx=indx, dtype=dtype)

            elif signal == 'arhmm' or signal == 'arhmm_states':
                dtype = 'int32'
                sample[signal] = self.try_to_load(
                    signal, key='states', indx=indx, dtype=dtype)

            elif signal == 'arhmm_predictions':
                dtype = 'float32'
                sample[signal] = self.try_to_load(
                    signal, key='predictions', indx=indx, dtype=dtype)

            else:
                raise ValueError('"%s" is an invalid signal type' % signal)

            # apply transforms
            if self.transforms[signal]:
                sample[signal] = self.transforms[signal](sample[signal])

            # transform into tensor
            if dtype == 'float32':
                sample[signal] = torch.from_numpy(sample[signal]).float()
            else:
                sample[signal] = torch.from_numpy(sample[signal]).long()

            sample[signal] = sample[signal].to(self.device)

        sample['batch_indx'] = indx

        return sample

    def try_to_load(self, signal, key, indx, dtype):
        try:
            data = load_pkl_dict(self.paths[signal], key, indx=indx, dtype=dtype)
        except FileNotFoundError:
            # try prepending session string
            try:
                self.paths[signal] = prepend_sess_id(self.paths[signal], self.sess_str)
                data = load_pkl_dict(self.paths[signal], key, indx=indx, dtype=dtype)
            except FileNotFoundError:
                raise NotImplementedError(
                    ('Could not open %s\nMust create %s from model;' +
                     ' currently not implemented') % (self.paths[signal], key))
        return data


class SingleSessionDataset(SingleSessionDatasetBatchedLoad):
    """
    Dataset class for a single session

    Loads all data during Dataset creation and saves as an attribute. Batches are then sampled from
    this stored data. All data transformations are applied to the full dataset upon load, *not*
    for each batch.
    """

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='', signals=None, transforms=None,
            paths=None, device='cuda'):
        """
        Args:
            data_dir (str): root directory of data
            lab (str)
            expt (str)
            animal (str)
            session (str)
            signals (list of strs): e.g. 'images' | 'masks' | 'neural' | ...
                see behavenet.fitting.utils.get_data_generator_inputs for examples
            transforms (list of transforms): each element corresponds to an entry in `signals`;
                for multiple transforms, chain together using pt transforms.Compose;
                see behavenet.data.transforms.py for available transform options
            paths (list of strs): each element corresponds to file location for an entry in
                `signals`; see behavenet.fitting.utils.get_data_generator_inputs for examples
            device (str, optional): location of data
                'cpu' | 'cuda'
        """

        super().__init__(data_dir, lab, expt, animal, session, signals, transforms, paths, device)

        # grab all data as a single batch
        self.data = super(SingleSessionDataset, self).__getitem__(indx=None)
        _ = self.data.pop('batch_indx')

        # collect dims for easy reference
        self.dims = OrderedDict()
        for signal, data in self.data.items():
            self.dims[signal] = data.shape

        if self.n_trials is None:
            self.n_trials = self.dims[signal][0]

    def __len__(self):
        return self.n_trials

    def __getitem__(self, indx):
        """
        Return batch of data; if indx is None, return all data

        Args:
            indx (int or NoneType): trial index

        Returns:
            (dict): data sample
        """

        sample = OrderedDict()
        for signal in self.signals:
            sample[signal] = self.data[signal][indx]
        sample['batch_indx'] = indx
        return sample


class ConcatSessionsGenerator(object):
    """
    Dataset class for multiple sessions

    Handles shuffling and iterating over sessions
    """

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids_list, signals_list=None, transforms_list=None, paths_list=None,
            device='cuda', as_numpy=False, batch_load=True, rng_seed=0, trial_splits=None,
            train_frac=1.0):
        """

        Args:
            data_dir (str): base directory for data
            ids_list (list of dicts): each element has the following keys:
                'lab', 'expt', 'animal', 'session';
                the data (neural activity, images, masks) is assumed to be located in:
                data_dir/lab/expt/animal/session/data.hdf5
            signals_list (list of lists): list of signals for each session
            transforms_list (list of lists): list of transforms for each session
            paths_list (list of lists): list of paths for each session
            device (str, optional): location of data
                'cpu' | 'cuda'
            as_numpy (bool, optional): `True` to return numpy array, `False` to return pytorch
                tensor
            batch_load (bool, optional): `True` to load data in batches as model is training,
                otherwise all data is loaded at once and stored on `device`
            rng_seed (int, optional): controls train/test/xv fold splits
            trial_splits (dict, optional): defines number of train/text/xv folds using the keys
                'train_tr', 'val_tr', 'test_tr', and 'gap_tr'; see `split_trials` for how these are
                used.
            train_frac (float, optional): if 0 < train_frac < 1.0, defines the fraction of assigned
                training trials to actually use; if >1.0, defines the number of assigned training
                trials to actually use
        """

        if isinstance(ids_list, dict):
            ids_list = [ids_list]
        self.ids = ids_list
        self.as_numpy = as_numpy

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
                device=device))
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
            dataset.batch_indxs = split_trials(len(dataset), rng_seed=rng_seed, **trial_splits)
            dataset.n_batches = {}
            for dtype in self._dtypes:
                if dtype == 'train':
                    # subsample training data if requested
                    if train_frac != 1.0:
                        n_batches = len(dataset.batch_indxs[dtype])
                        if train_frac < 1.0:
                            # subsample as fraction of total batches
                            n_indxs = int(np.floor(train_frac * n_batches))
                            if n_indxs <= 0:
                                print(
                                    'warning: attempting to use invalid number of training ' +
                                    'batches; defaulting to all training batches')
                                n_indxs = n_batches
                        else:
                            # subsample fixed number of batches
                            train_frac = n_batches if train_frac > n_batches else train_frac
                            n_indxs = int(train_frac)
                        indxs_rand = np.random.choice(n_batches, size=n_indxs, replace=False)
                        dataset.batch_indxs[dtype] = dataset.batch_indxs[dtype][indxs_rand]
                    self.batch_ratios[i] = len(dataset.batch_indxs[dtype])
                dataset.n_batches[dtype] = len(dataset.batch_indxs[dtype])
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
                    sampler=SubsetRandomSampler(dataset.batch_indxs[dtype]),
                    num_workers=0)

        # create all iterators (will iterate through data loaders)
        self.dataset_iters = [None] * self.n_datasets
        for i in range(self.n_datasets):
            self.dataset_iters[i] = {}
            for dtype in self._dtypes:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def __str__(self):
        # return info about number of datasets
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
        """
        Reset iterators so that all data is available

        Args:
            dtype (str): 'train' | 'val' | 'test' | 'all'
        """

        for i in range(self.n_datasets):
            if dtype == 'all':
                for dtype_ in self._dtypes:
                    self.dataset_iters[i][dtype_] = iter(self.dataset_loaders[i][dtype_])
            else:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def next_batch(self, dtype):
        """
        Iterate randomly through sessions and trials; a batch from each session
        is used before resetting the session iterator. Once a session runs out
        of trials it is skipped.

        Args:
            dtype (str): 'train' | 'val' | 'test'

        Returns:
            (tuple)
                - sample (dict): sample batch with keys given by `signals`
                    input to class constructor
                - dataset (int): dataset from which data sample is drawn
        """

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
                sample[signal] = sample[signal].cpu().detach().numpy()

        return sample, dataset
