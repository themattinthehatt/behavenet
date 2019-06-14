import os
import numpy as np
import glob
import pickle
from collections import OrderedDict
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import h5py
from behavenet.fitting.utils import get_best_model_version


def split_trials(
        n_trials, rng_seed=0, train_tr=5, val_tr=1, test_tr=1, gap_tr=1):
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


class SingleSessionDatasetBatchedLoad(data.Dataset):
    """
    Dataset class for a single session

    Loads data one batch at a time; data transformations are applied to each
    batch upon load.
    """

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='',
            signals=None, transforms=None, load_kwargs=None, device='cpu'):
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
            load_kwargs (list of dicts): each element corresponds to loading
                parameters for an entry in `signals`; see
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

        self.signals = signals
        self.transforms = transforms
        self.load_kwargs = load_kwargs

        # get total number of trials by loading images/neural data
        signal = 'images' if 'images' in signals else 'neural'
        data_file = os.path.join(self.data_dir, 'data.hdf5')
        with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
            self.n_trials = len(f[signal])
            key_list = list(f[signal].keys())
            self.trial_len = f[signal][key_list[0]].shape[0]

        # meta data about train/test/xv splits; set by ConcatSessionsGenerator
        self.batch_indxs = None
        self.n_batches = None

        self.device = device

        # TODO: not all signals are stored in hdf5 file
        # self.dims = OrderedDict()
        # for signal in self.signals:
        #     key_list = list(f[signal].keys())
        #     self.dims[signal] = f[signal][key_list[0]].shape

        # get data paths
        self.paths = OrderedDict()
        for signal, transform, load_kwarg in zip(
                self.signals, self.transforms, self.load_kwargs):
            if signal == 'images':
                self.paths[signal] = os.path.join(self.data_dir, 'data.hdf5')
            elif signal == 'masks':
                self.paths[signal] = os.path.join(self.data_dir, 'data.hdf5')
            elif signal == 'neural':
                self.paths[signal] = os.path.join(self.data_dir, 'data.hdf5')
            elif signal == 'ae':
                # build path to latents
                if 'latents_file' in load_kwarg:
                    self.paths[signal] = load_kwarg['latents_file']
                else:
                    if load_kwarg['model_dir'] is None:
                        raise IOError(
                            'Must supply ae directory or latents file')
                    if 'model_version' in load_kwarg and isinstance(
                            load_kwarg['model_version'], int):
                        model_dir = os.path.join(
                            load_kwarg['model_dir'],
                            'version_%i' % load_kwarg['model_version'])
                    else:
                        model_version = get_best_model_version(
                            load_kwarg['model_dir'], 'val_loss')[0]
                        model_dir = os.path.join(
                            load_kwarg['model_dir'], model_version)
                    # find file with "latents" in name
                    self.paths[signal] = glob.glob(os.path.join(
                        model_dir, '*latents*.pkl'))[0]
            elif signal == 'ae_predictions':
                # build path to latents
                if 'predictions_file' in load_kwarg:
                    self.paths[signal] = load_kwarg['predictions_file']
                else:
                    if load_kwarg['model_dir'] is None:
                        raise IOError(
                            'Must supply ae directory or predictions file')
                    if 'model_version' in load_kwarg and isinstance(
                            load_kwarg['model_version'], int):
                        model_dir = os.path.join(
                            load_kwarg['model_dir'],
                            'version_%i' % load_kwarg['model_version'])
                    else:
                        model_version = get_best_model_version(
                            load_kwarg['model_dir'], 'val_loss')[0]
                        model_dir = os.path.join(
                            load_kwarg['model_dir'], model_version)
                    # find file with "latents" in name
                    self.paths[signal] = glob.glob(os.path.join(
                        model_dir, '*predictions*.pkl'))[0]
            elif signal == 'arhmm':
                # build path to latents
                if 'states_file' in load_kwarg:
                    self.paths[signal] = load_kwarg['states_file']
                else:
                    if load_kwarg['model_dir'] is None:
                        raise IOError(
                            'Must supply arhmm directory or states file')
                    if 'model_version' in load_kwarg and isinstance(
                            load_kwarg['model_version'], int):
                        model_dir = os.path.join(
                            load_kwarg['model_dir'],
                            'version_%i' % load_kwarg['model_version'])
                    else:
                        model_version = get_best_model_version(
                            load_kwarg['model_dir'], 'val_ll', best_def='max')[0]
                        model_dir = os.path.join(
                            load_kwarg['model_dir'], model_version)
                    # find file with "latents" in name
                    self.paths[signal] = glob.glob(os.path.join(
                        model_dir, '*states*.pkl'))[0]
            elif signal == 'arhmm_predictions':
                # build path to latents
                if 'predictions_file' in load_kwarg:
                    self.paths[signal] = load_kwarg['predictions_file']
                else:
                    if load_kwarg['model_dir'] is None:
                        raise IOError(
                            'Must supply arhmm directory or predictions file')
                    if 'model_version' in load_kwarg and isinstance(
                            load_kwarg['model_version'], int):
                        model_dir = os.path.join(
                            load_kwarg['model_dir'],
                            'version_%i' % load_kwarg['model_version'])
                    else:
                        model_version = get_best_model_version(
                            load_kwarg['model_dir'], 'val_loss')[0]
                        model_dir = os.path.join(
                            load_kwarg['model_dir'], model_version)
                    # find file with "latents" in name
                    self.paths[signal] = glob.glob(os.path.join(
                        model_dir, '*predictions*.pkl'))[0]
            else:
                raise ValueError('"%s" is an invalid signal type')

    def __len__(self):
        return self.n_trials

    def __getitem__(self, indx):
        """
        Return batch of data; if indx is None, return all data

        Args:
            indx (int): trial index

        Returns:
            (dict): data sample
        """

        sample = OrderedDict()
        for signal, transform, load_kwargs in zip(
                self.signals, self.transforms, self.load_kwargs):

            # index correct trial
            if signal == 'images':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        print('Warning: loading all images!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][
                                str('trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][
                            str('trial_%04i' % indx)][()].astype(dtype)
                # normalize to range [0, 1]
                sample[signal] /= 255.0

            elif signal == 'masks':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        print('Warning: loading all masks!')
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][
                                str('trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][
                            str('trial_%04i' % indx)][()].astype(dtype)

            elif signal == 'neural':
                dtype = 'float32'
                with h5py.File(self.paths[signal], 'r', libver='latest', swmr=True) as f:
                    if indx is None:
                        temp_data = []
                        for tr in range(self.n_trials):
                            temp_data.append(f[signal][
                                str('trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][
                            str('trial_%04i' % indx)][()].astype(dtype)

            elif signal == 'ae':
                dtype = 'float32'
                try:
                    with open(self.paths[signal], 'rb') as f:
                        latents_dict = pickle.load(f)
                    if indx is None:
                        sample[signal] = latents_dict['latents']
                    else:
                        sample[signal] = latents_dict['latents'][indx]
                    sample[signal] = sample[signal].astype(dtype)
                except IOError:
                    raise NotImplementedError(
                        'Must create ae latents from model; currently not' +
                        ' implemented')

            elif signal == 'ae_predictions':
                dtype = 'float32'
                try:
                    with open(self.paths[signal], 'rb') as f:
                        latents_dict = pickle.load(f)
                    if indx is None:
                        sample[signal] = latents_dict['predictions']
                    else:
                        sample[signal] = latents_dict['predictions'][indx]
                    sample[signal] = sample[signal].astype(dtype)
                except IOError:
                    raise NotImplementedError(
                        'Must create ae predictions from model; currently not' +
                        ' implemented')

            elif signal == 'arhmm':
                dtype = 'int32'
                try:
                    with open(self.paths[signal], 'rb') as f:
                        latents_dict = pickle.load(f)
                    if indx is None:
                        sample[signal] = latents_dict['states']
                    else:
                        sample[signal] = latents_dict['states'][indx]
                    sample[signal] = sample[signal]
                except IOError:
                    raise NotImplementedError(
                        'Must create arhmm latents from model; currently not' +
                        ' implemented')

            elif signal == 'arhmm_predictions':
                dtype = 'float32'
                try:
                    with open(self.paths[signal], 'rb') as f:
                        latents_dict = pickle.load(f)
                    if indx is None:
                        sample[signal] = latents_dict['predictions']
                    else:
                        sample[signal] = latents_dict['predictions'][indx]
                    sample[signal] = sample[signal].astype(dtype)
                except IOError:
                    raise NotImplementedError(
                        'Must create arhmm predictions from model; currently not' +
                        ' implemented')

            else:
                raise ValueError('"%s" is an invalid signal type' % signal)

            # apply transforms
            if transform:
                sample[signal] = transform(sample[signal])
                
            # transform into tensor
            if dtype == 'float32':
                sample[signal] = torch.from_numpy(sample[signal]).float()
            else:
                sample[signal] = torch.from_numpy(sample[signal]).long()

            sample[signal] = sample[signal].to(self.device)

        sample['batch_indx'] = indx

        return sample


class SingleSessionDataset(SingleSessionDatasetBatchedLoad):
    """
    Dataset class for a single session

    Loads all data during Dataset creation and saves as an attribute. Batches
    are then sampled from this stored data. All data transformations are
    applied to the full dataset upon load, *not* for each batch.
    """

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='',
            signals=None, transforms=None, load_kwargs=None, device='cuda'):
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
            load_kwargs (list of dicts): each element corresponds to loading
                parameters for an entry in `signals`; see
                behavenet.fitting.utils.get_data_generator_inputs for examples
            device (str, optional): location of data
                'cpu' | 'cuda'
        """

        super().__init__(
            data_dir, lab, expt, animal, session, signals, transforms,
            load_kwargs, device)

        # grab all data as a single batch
        self.data = super(SingleSessionDataset, self).__getitem__(indx=None)
        self.data.pop('batch_indx')

        # collect dims for easy reference
        self.dims = OrderedDict()
        for signal, data in self.data.items():
            self.dims[signal] = data.shape

    def __len__(self):
        return self.n_trials

    def __getitem__(self, indx):
        """
        Return batch of data; if indx is None, return all data

        Args:
            indx (int): trial index

        Returns:
            (dict): data sample
        """

        sample = OrderedDict()
        for signal in self.signals:
            sample[signal] = self.data[signal][indx]
        sample['batch_indx'] = indx
        return sample


class ConcatSessionsGenerator(object):

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids_list, signals=None, transforms=None,
            load_kwargs=None, device='cuda', as_numpy=False, batch_load=True,
            rng_seed=0):
        """

        Args:
            data_dir (str): base directory for data
            ids_list (list of dicts): each element has the following keys:
                'lab', 'expt', 'animal', 'session';
                data (neural, images) is assumed to be located in:
                data_dir/lab/expt/animal/session/data.hdf5
            signals (list of strs): e.g. 'images' | 'masks' | 'neural' | ...
                see `behavenet.fitting.utils.get_data_generator_inputs` for
                examples
            transforms (list of transforms): each element corresponds to an
                entry in `signals`; for multiple transforms, chain together
                using pt transforms.Compose; see `behavenet.data.transforms.py`
                for available transform options
            load_kwargs (list of dicts): each element corresponds to loading
                parameters for an entry in `signals`; see
                `behavenet.fitting.utils.get_data_generator_inputs` for
                examples
            device (str, optional): location of data
                'cpu' | 'cuda'
            as_numpy (bool, optional): `True` to return numpy array, `False` to
                return pytorch tensor
            batch_load (bool, optional): `True` to load data in batches as
                model is training, otherwise all data is loaded at once and
                stored on `device`
            rng_seed (int, optional): controls train/test/xv fold splits
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

        self.signals = signals
        self.transforms = transforms
        self.load_kwargs = load_kwargs
        for ids in ids_list:
            self.datasets.append(SingleSession(
                data_dir, lab=ids['lab'], expt=ids['expt'],
                animal=ids['animal'], session=ids['session'],
                signals=signals, transforms=transforms,
                load_kwargs=load_kwargs, device=device))
            self.datasets_info.append({
                'lab': ids['lab'], 'expt': ids['expt'], 'animal': ids['animal'],
                'session': ids['session']})

        # collect info about datasets
        self.n_datasets = len(self.datasets)

        # get train/val/test batch indices for each dataset
        self.batch_ratios = [None] * self.n_datasets
        for i, dataset in enumerate(self.datasets):
            dataset.batch_indxs = split_trials(len(dataset), rng_seed=rng_seed)
            dataset.n_batches = {}
            for dtype in self._dtypes:
                dataset.n_batches[dtype] = len(dataset.batch_indxs[dtype])
                if dtype == 'train':
                    self.batch_ratios[i] = len(dataset.batch_indxs[dtype])
        self.batch_ratios = np.array(self.batch_ratios) / np.sum(self.batch_ratios)

        # find total number of batches per data type; this will be iterated
        # over in the training loop
        self.n_tot_batches = {}
        for dtype in self._dtypes:
            self.n_tot_batches[dtype] = np.sum([
                dataset.n_batches[dtype] for dataset in self.datasets])

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
                self.dataset_iters[i][dtype] = iter(
                    self.dataset_loaders[i][dtype])

    def __repr__(self):
        # return info about number of datasets
        if self.batch_load:
            single_sess_str = 'SingleSessionDatasetBatchedLoad'
        else:
            single_sess_str = 'SingleSessionDataset'
        format_str = str('Generator contains %i %s objects\n' %
                         (self.n_datasets, single_sess_str))
        for i in range(len(self.signals)):
            format_str += str('\tsignal:\n\t\t{}\n'.format(self.signals[i]))
            format_str += str('\ttransform:\n\t\t{}\n'.format(self.transforms[i]))
            format_str += str('\tload_kwargs:\n')
            if self.load_kwargs[i] is not None:
                for key, value in self.load_kwargs[i].items():
                    format_str += str('\t\t{}: {}\n'.format(key, value))
            else:
                format_str += str('\t\t{}'.format(None))
            format_str += '\n\n'
        return format_str

    def reset_iterators(self, dtype):
        """
        Reset iterators so that all data is available

        Args:
            dtype (str): 'train' | 'val' | 'test' | 'all'
        """

        for i in range(self.n_datasets):
            if dtype == 'all':
                for dtype_ in self._dtypes:
                    self.dataset_iters[i][dtype_] = iter(
                        self.dataset_loaders[i][dtype_])
            else:
                self.dataset_iters[i][dtype] = iter(
                    self.dataset_loaders[i][dtype])

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
            dataset = np.random.choice(
                np.arange(self.n_datasets), p=self.batch_ratios)

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
