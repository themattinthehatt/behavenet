import os
import numpy as np
import glob
from collections import OrderedDict
from skimage import io as sio
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import h5py


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


def get_img_filenames(img_dir='', img_ext='jpg', pattern=None):
    if pattern is None:
        filenames = glob.glob(os.path.join(img_dir, '*.%s' % img_ext))
    else:
        filenames = glob.glob(pattern)
        img_dir = os.path.dirname(filenames[0])
    filenames_rel = [os.path.basename(x) for x in filenames]
    filenames_rel.sort()
    filenames = [os.path.join(img_dir, x) for x in filenames_rel]
    return filenames


def imread(file, **load_func_kwargs):
    return sio._io.imread(file, **load_func_kwargs).astype(np.float32)


class SingleSessionDataset(data.Dataset):
    """Dataset class for a single session"""

    def __init__(
            self, data_dir, lab='', expt='', animal='', session='',
            signals_list=None, transform_list=None, device='cpu',
            as_numpy=False, format='hdf5'):
        """
        Read filenames

        Args:
            data_dir (str): root directory of data
            lab (str)
            expt (str)
            animal (str)
            session (str)
            signals_list (list of strs):
                'neural' | 'images'
            transform_list (list of transforms): each entry corresponds to an
                entry in `signals_list`; for multiple transforms, chain
                together using pt transforms.Compose
            device (str):
                'cpu' | 'cuda'
            as_numpy (bool): `True` to return numpy array, `False` to return
                pytorch tensor
            format (str):
                'jpg' | 'hdf5'
        """

        # specify data
        self.lab = lab
        self.expt = expt
        self.animal = animal
        self.session = session
        self.data_dir = os.path.join(
            data_dir, self.lab, self.expt, self.animal, self.session)

        self.signals_list = signals_list
        self.transform_list = transform_list

        self.filenames = get_img_filenames(
            img_dir=os.path.join(self.data_dir, 'face'))
        if len(self.filenames) == 0:
            raise IOError('"%s" is not a valid data directory' % self.data_dir)
        self.trials = np.unique(np.array(
            [int(os.path.basename(t)[3:7]) for t in self.filenames]))

        self.format = format
        self.device = device
        self.as_numpy = as_numpy

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, indx):
        """Load images from filenames"""

        sample = OrderedDict()
        
        for i, signal in enumerate(self.signals_list):

            # index correct trial
            if signal == 'images':
                if self.format == 'jpg':
                    load_pattern = os.path.join(
                        self.data_dir, 'body', 'img%04i*.jpg' % indx)
                    sample[signal] = sio.ImageCollection(
                        get_img_filenames(pattern=load_pattern),
                        conserve_memory=False,
                        load_func=imread,
                        as_gray=True).concatenate()[:, None, :, :]
                elif self.format == 'hdf5':
                    f = h5py.File(os.path.join(
                        self.data_dir, 'images.hdf5'), 'r',
                        libver='latest', swmr=True)
                    sample[signal] = f['images'][
                        str('trial_%04i' % indx)][()].astype('float32') / 255.0

                # if self.lab == 'steinmetz':
                #     load_pattern = os.path.join(
                #         self.data_dir, 'face', 'img%04i*.jpg' % indx)
                #     sample[signal] = io.ImageCollection(
                #         get_img_filenames(pattern=load_pattern),
                #         conserve_memory=False,
                #         load_func=imread,
                #         as_gray=True).concatenate()[:, None, :, :]
                # elif self.lab == 'churchland':
                #     load_pattern_face = os.path.join(
                #         self.data_dir, 'face', 'img%04i*.jpg' % indx)
                #     load_pattern_body = os.path.join(
                #         self.data_dir, 'body', 'img%04i*.jpg' % indx)
                #     sample[signal] = np.concatenate([
                #         io.ImageCollection(
                #             get_img_filenames(pattern=load_pattern_face),
                #             conserve_memory=False,
                #             load_func=imread,
                #             as_gray=True).concatenate()[:, None, :, :],
                #         io.ImageCollection(
                #             get_img_filenames(pattern=load_pattern_body),
                #             conserve_memory=False,
                #             load_func=imread,
                #             as_gray=True).concatenate()[:, None, :, :]],
                #         axis=1)

            else:
                raise ValueError('"%s" is an invalid signal type' % signal)

            # apply transforms
            if self.transform_list[i]:
                sample[signal] = self.transform_list[i](sample[signal])
                
            # transform into tensor
            if not self.as_numpy:
                sample[signal] = torch.from_numpy(sample[signal]).to(self.device)
            sample['batch_indx'] = indx

        return sample


class ConcatSessionsGenerator(object):

    _dtypes = {'train', 'val', 'test'}

    def __init__(
            self, data_dir, ids, signals_list, transform_list, device='cuda',
            rng_seed=0, format='jpg', num_workers=0):
        """

        Args:
            data_dir:
            ids:
            signals_list:
            transform_list:
            device:
            rng_seed:
        """

        self.ids = ids

        # gather all datasets
        def get_dirs(path):
            return next(os.walk(path))[1]

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
                        self.datasets.append(SingleSessionDataset(
                            data_dir, lab=lab, expt=expt, animal=animal,
                            session=session, signals_list=signals_list,
                            transform_list=transform_list, device=device,
                            format=format))
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
                    self.datasets.append(SingleSessionDataset(
                        data_dir, lab=lab, expt=expt, animal=animal,
                        session=session, signals_list=signals_list,
                        transform_list=transform_list, device=device,
                        format=format))
                    self.datasets_info.append({
                        'lab': lab, 'expt': expt, 'animal': animal,
                        'session': session})
        elif isinstance(ids['session'], list):
            # get all sessions from one animal
            expt = ids['expt']
            animal = ids['animal']
            for session in ids['session']:
                self.datasets.append(SingleSessionDataset(
                    data_dir, lab=lab, expt=expt, animal=animal,
                    session=session, signals_list=signals_list,
                    transform_list=transform_list, device=device,
                    format=format))
                self.datasets_info.append({
                    'lab': lab, 'expt': expt, 'animal': animal,
                    'session': session})
        else:
            self.datasets.append(SingleSessionDataset(
                data_dir, lab=ids['lab'], expt=ids['expt'],
                animal=ids['animal'], session=ids['session'],
                signals_list=signals_list, transform_list=transform_list,
                device=device, format=format))
            self.datasets_info.append({
                'lab': ids['lab'], 'expt': ids['expt'], 'animal': ids['animal'],
                'session': ids['session']})

        # collect info about datasets
        self.num_datasets = len(self.datasets)

        # get train/val/test batch indices for each dataset
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
                if ids['lab'] == 'churchland' and format == 'jpg':
                    self.batch_indxs[i][dtype] = self.batch_indxs[i][dtype] + 1
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
                    num_workers=num_workers)
        
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

        return sample, dataset
