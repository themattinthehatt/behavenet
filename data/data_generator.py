import os
import numpy as np
import glob
import pickle
from collections import OrderedDict
from scipy.io import loadmat
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import h5py
from fitting.utils import get_best_model_version


def split_trials(
        num_trials, rng_seed=0, train_tr=5, val_tr=1, test_tr=1, gap_tr=1):
    """
    Split trials into train/val/test; use `num_trials` out of a total possible
    number `max_trials` (this is to ensure that the same number of trials are
    used from each dataset when multiple are concatenated).

    The data is split into blocks that have gap trials between tr/val/test:
    train tr | gap tr | val tr | gap tr | test tr | gap tr

    Args:
        num_trials (int): number of trials to use in the split
        rng_seed (int):
        train_tr (int): number of train trials per block
        val_tr (int): number of validation trials per block
        test_tr (int): number of test trials per block
        gap_tr (int): number of gap trials between tr/val/test; there will be
            a total of 3 * `gap_tr` gap trials per block

    Returns:
        dict
    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = train_tr + gap_tr + val_tr + gap_tr + test_tr + gap_tr

    num_blocks = int(np.floor(num_trials / tr_per_block))
    leftover_trials = num_trials - tr_per_block * num_blocks
    if leftover_trials > 0:
        offset = np.random.randint(0, high=leftover_trials)
    else:
        offset = 0
    indxs_block = np.random.permutation(num_blocks)

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
        Read filenames

        Args:
            data_dir (str): root directory of data
            lab (str)
            expt (str)
            animal (str)
            session (str)
            signals (list of strs):
                'images'
            transforms (list of transforms): each entry corresponds to an
                entry in `signals`; for multiple transforms, chain
                together using pt transforms.Compose
            load_kwargs (list of dicts): each entry corresponds to loading
                parameters for an entry in `signals`
            device (str):
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
        if 'images' in signals:
            img_dir = os.path.join(self.data_dir, 'images.hdf5')
            with h5py.File(img_dir, 'r', libver='latest', swmr=True) as f:
                self.num_trials = len(f['images'])
                key_list = list(f['images'].keys())
                self.trial_len = f['images'][key_list[0]].shape[0]
        else:
            mat_contents = loadmat(os.path.join(self.data_dir, 'neural.mat'))
            self.num_trials = mat_contents['neural'].shape[0]
            self.trial_len = mat_contents['neural'].shape[1]

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
                self.paths[signal] = os.path.join(self.data_dir, 'images.hdf5')
            elif signal == 'masks':
                self.paths[signal] = os.path.join(self.data_dir, 'images.hdf5')
            elif signal == 'neural':
                self.paths[signal] = os.path.join(self.data_dir, 'neural.mat')
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
                            load_kwarg['model_dir'], 'val_ll')[0]
                        model_dir = os.path.join(
                            load_kwarg['model_dir'], model_version)
                    # find file with "latents" in name
                    self.paths[signal] = glob.glob(os.path.join(
                        model_dir, '*states*.pkl'))[0]
            else:
                raise ValueError('"%s" is an invalid signal type')

    def __len__(self):
        return self.num_trials

    def __getitem__(self, indx):
        """Return batch of data; if indx is None, return all data"""

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
                        for tr in range(self.num_trials):
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
                        for tr in range(self.num_trials):
                            temp_data.append(f[signal][
                                str('trial_%04i' % tr)][()].astype(dtype)[None, :])
                        sample[signal] = np.concatenate(temp_data, axis=0)
                    else:
                        sample[signal] = f[signal][
                            str('trial_%04i' % indx)][()].astype(dtype)

            elif signal == 'neural':
                dtype = 'float32'
                mat_contents = loadmat(self.paths[signal])
                if indx is None:
                    sample[signal] = mat_contents['neural']
                    # try:
                    #     self.reg_indxs = mat_contents['reg_indxs_consolidate']
                    # except KeyError:
                    #     try:
                    #         self.reg_indxs = mat_contents['reg_indxs']
                    #     except KeyError:
                    #         self.reg_indxs = None
                else:
                    sample[signal] = mat_contents['neural'][indx][None, :]

            elif signal == 'ae':
                dtype = 'float32'
                try:
                    with open(self.paths[signal], 'rb') as f:
                        latents_dict = pickle.load(f)
                    if indx is None:
                        sample[signal] = latents_dict['latents']
                    else:
                        sample[signal] = latents_dict['latents'][indx][None, :]
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
                        sample[signal] = latents_dict['predictions'][indx][None, :]
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
                        sample[signal] = latents_dict['states'][indx][None, :]
                    sample[signal] = sample[signal]
                except IOError:
                    raise NotImplementedError(
                        'Must create arhmm latents from model; currently not' +
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
        Read data

        Args:
            data_dir (str): root directory of data
            lab (str)
            expt (str)
            animal (str)
            session (str)
            signals (list of strs):
                'neural' | 'images' | 'ae' | 'arhmm' | 'ae_predictions' |
                'arhmm_predictions'
            transforms (list of transforms): each entry corresponds to an
                entry in `signals`; for multiple transforms, chain
                together using pt transforms.Compose
            load_kwargs (list of dicts): each entry corresponds to loading
                parameters for an entry in `signals`
            device (str):
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
        return self.num_trials

    def __getitem__(self, indx):
        sample = OrderedDict()
        for signal in self.signals:
            sample[signal] = self.data[signal][indx]
        sample['batch_indx'] = indx
        return sample


class ConcatSessionsGenerator(object):

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids, signals=None, transforms=None,
            load_kwargs=None, device='cuda', as_numpy=False, batch_load=True,
            rng_seed=0):
        """

        Args:
            data_dir:
            ids:
            signals (list):
            transforms (list):
            load_kwargs (list):
            device (str): location of model
                'cpu' | 'cuda'
            as_numpy (bool): `True` to return numpy array, `False` to return
                pytorch tensor
            batch_load (bool): `True` to load data in batches as model is
                training, otherwise all data is loaded at once and stored on
                `device`
            rng_seed (int):
        """

        self.ids = ids
        self.as_numpy = as_numpy

        # gather all datasets
        def get_dirs(path):
            return next(os.walk(path))[1]

        if batch_load:
            SingleSession = SingleSessionDatasetBatchedLoad
        else:
            SingleSession = SingleSessionDataset

        self.datasets = []
        self.datasets_info = []
        lab = ids['lab']
        if isinstance(ids['expt'], list):
            # get all experiments from one lab
            for expt in ids['expt']:
                animals = get_dirs(os.path.join(data_dir, lab, expt))
                for animal in animals:
                    sessions = get_dirs(
                        os.path.join(data_dir, lab, expt, animal))
                    for session in sessions:
                        self.datasets.append(SingleSession(
                            data_dir, lab=lab, expt=expt, animal=animal,
                            session=session, signals=signals,
                            transforms=transforms, load_kwargs=load_kwargs,
                            device=device))
                        self.datasets_info.append({
                            'lab': lab, 'expt': expt, 'animal': animal,
                            'session': session})
        elif isinstance(ids['animal'], list):
            # get all animals from one experiment
            expt = ids['expt']
            for animal in ids['animal']:
                sessions = get_dirs(
                    os.path.join(data_dir, lab, expt, animal))
                for session in sessions:
                    self.datasets.append(SingleSession(
                        data_dir, lab=lab, expt=expt, animal=animal,
                        session=session, signals=signals,
                        transforms=transforms, load_kwargs=load_kwargs,
                        device=device))
                    self.datasets_info.append({
                        'lab': lab, 'expt': expt, 'animal': animal,
                        'session': session})
        elif isinstance(ids['session'], list):
            # get all sessions from one animal
            expt = ids['expt']
            animal = ids['animal']
            for session in ids['session']:
                self.datasets.append(SingleSession(
                    data_dir, lab=lab, expt=expt, animal=animal,
                    session=session, signals=signals,
                    transforms=transforms, load_kwargs=load_kwargs,
                    device=device))
                self.datasets_info.append({
                    'lab': lab, 'expt': expt, 'animal': animal,
                    'session': session})
        else:
            self.datasets.append(SingleSession(
                data_dir, lab=ids['lab'], expt=ids['expt'],
                animal=ids['animal'], session=ids['session'],
                signals=signals, transforms=transforms,
                load_kwargs=load_kwargs, device=device))
            self.datasets_info.append({
                'lab': ids['lab'], 'expt': ids['expt'], 'animal': ids['animal'],
                'session': ids['session']})

        # collect info about datasets
        self.num_datasets = len(self.datasets)

        # get train/val/test batch indices for each dataset
        # TODO: move info into SingleSessionDataset objects?
        self.batch_indxs = [None] * self.num_datasets
        self.num_batches = [None] * self.num_datasets
        self.batch_ratios = [None] * self.num_datasets
        for i, dataset in enumerate(self.datasets):
            self.batch_indxs[i] = split_trials(len(dataset), rng_seed=rng_seed)
            self.num_batches[i] = {}
            for dtype in self._dtypes:
                self.num_batches[i][dtype] = len(self.batch_indxs[i][dtype])
                if dtype == 'train':
                    self.batch_ratios[i] = len(self.batch_indxs[i][dtype])
        self.batch_ratios = np.array(self.batch_ratios) / np.sum(self.batch_ratios)

        # find total number of batches per data type; this will be iterated
        # over in the training loop
        self.num_tot_batches = {}
        for dtype in self._dtypes:
            self.num_tot_batches[dtype] = np.sum([
                item[dtype] for item in self.num_batches])

        # create data loaders (will shuffle/batch/etc datasets)
        self.dataset_loaders = [None] * self.num_datasets
        for i, dataset in enumerate(self.datasets):
            self.dataset_loaders[i] = {}
            for dtype in self._dtypes:
                self.dataset_loaders[i][dtype] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=SubsetRandomSampler(self.batch_indxs[i][dtype]),
                    num_workers=0)
        
        # create all iterators (will iterate through data loaders)
        self.dataset_iters = [None] * self.num_datasets
        for i in range(self.num_datasets):
            self.dataset_iters[i] = {}
            for dtype in self._dtypes:
                self.dataset_iters[i][dtype] = iter(self.dataset_loaders[i][dtype])

    def reset_iterators(self, dtype):
        """
        Args:
            dtype (str): 'train' | 'val' | 'test' | 'all'
        """

        for i in range(self.num_datasets):
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
        is used before reseting the session iterator. Once a session runs out
        of trials it is skipped.
        """
        while True:
            # get next session
            dataset = np.random.choice(
                np.arange(self.num_datasets), p=self.batch_ratios)

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
