"""Tranform classes to process data.

Data generator objects can apply these transforms to batches of data upon loading.
"""

import numpy as np
from skimage import transform


class Compose(object):
    """Composes several transforms together.

    Adapted from pytorch source code:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    Example
    -------
    .. code-block::

        >> Compose([
        >>     behavenet.data.transforms.SelectRegion('mctx', mctx_idxs),
        >>     behavenet.data.transforms.Threshold(threshold=1.0, bin_size=25),
        >> ])

    Parameters
    ----------
    transforms : :obj:`list` of :obj:`transform`
        list of transforms to compose

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '{0}, '.format(t)
        format_string += '\b\b)'
        return format_string


class Transform(object):
    """Abstract base class for transforms."""

    def __call__(self, *args):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class GetMask(Transform):
    """Mask data using a static threshold."""

    def __init__(self, ll_thresh, depth_thresh):
        self.ll_thresh = ll_thresh
        self.depth_thresh = depth_thresh

    def __call__(self, signal1, signal2):
        # tranform the signal
        signal1 = signal1 < self.ll_thresh  # ll < ll thresh
        signal2 = signal2 > self.depth_thresh  # depth > depth thresh
        # mask = ll < ll thresh || depth > depth thresh
        signal = ((signal1.astype('int') + signal2.astype('int')) < 2).astype('int')
        return signal

    def __repr__(self):
        return str('GetMask(ll_thresh=%f, depth_thresh=%f)' % (self.ll_thresh, self.depth_thresh))


class ClipNormalize(Transform):
    """Clip upper level of signal and divide by clip value."""

    def __init__(self, clip_val=100):
        """

        Parameters
        ----------
        clip_val : :obj:`float`
            signal values above this will be set to this value, then divided by this value so that
            signal maximum is 1

        """
        assert clip_val != 0
        self.clip_val = clip_val

    def __call__(self, signal):
        """

        Parameters
        ----------
        signal : :obj:`np.ndarray`

        Returns
        -------
        :obj:`np.ndarray`

        """
        signal = np.minimum(signal, self.clip_val)
        signal = signal / self.clip_val
        return signal

    def __repr__(self):
        return str('ClipNormalize(clip_val=%f)' % self.clip_val)


class Resize(Transform):
    """Resize the sample images."""

    def __init__(self, size=(128, 128), order=1):
        """

        Parameters
        ----------
        size : :obj:`int` or :obj:`tuple`
            desired output size for each image; if type is :obj:`int`, the same value is used for
            both height and width
        order : :obj:`int`
            interpolation order

        """
        assert isinstance(size, (tuple, int))
        self.order = order
        if isinstance(size, tuple):
            self.x = size[0]
            self.y = size[1]
        else:
            self.x = self.y = size

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (trial, time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time, n_channels)

        """
        sh = sample.shape
        sample = transform.resize(sample, (sh[0], sh[1], self.x, self.y), order=self.order)
        return sample

    def __repr__(self):
        return str('Resize(size=(%i, %i))' % (self.x, self.y))


class Threshold(Transform):
    """Remove channels of neural activity whose mean value is below a threshold."""

    def __init__(self, threshold, bin_size):
        """

        Parameters
        ----------
        threshold : :obj:`float`
            threshold in Hz
        bin_size : :obj:`float`
            bin size of neural activity in ms

        """
        self.threshold = threshold
        self.bin_size = bin_size

    def __call__(self, sample):
        """Calculates firing rate over all time points and thresholds.

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (trial, time, n_channels)

         Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time, n_channels)

        """
        # get firing rates
        frs = np.squeeze(np.mean(sample, axis=(0, 1))) / (self.bin_size * 1e-3)
        fr_mask = frs > self.threshold
        # get rid of neurons below fr threshold
        sample = sample[:, :, fr_mask]
        return sample.astype(np.float)

    def __repr__(self):
        return str('Threshold(threshold=%f, bin_size=%f)' % (self.threshold, self.bin_size))


class ZScore(Transform):
    """z-score channel activity."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : :obj:`np.ndarray`
            input shape is (trial, time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time, n_channels)

        """
        sample -= np.mean(sample, axis=(0, 1))
        sample /= np.std(sample, axis=(0, 1))
        return sample

    def __repr__(self):
        return 'ZScore()'


class MakeOneHot(Transform):
    """Turn a categorical vector into a one-hot vector."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """Assumes that K classes are identified by the numbers 0:K-1.

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (trial, time)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time, K)

        """
        if len(sample.shape) == 2:  # weak test for if sample is already onehot
            n_trials, n_time = sample.shape
            n_classes = int(np.nanmax(sample))
            onehot = np.zeros((n_trials, n_time, n_classes + 1))
            for t in range(n_trials):
                if not any(np.isnan(sample[t])):
                    onehot[t, np.arange(n_time), sample[t].astype('int')] = 1
                else:
                    onehot[t] = np.nan
        else:
            onehot = sample
        return onehot

    def __repr__(self):
        return 'MakeOneHot()'


class BlockShuffle(Transform):
    """Shuffle blocks of contiguous discrete states within each trial."""

    def __init__(self, rng_seed):
        """

        Parameters
        ----------
        rng_seed : :obj:`int`
            to control random number generator

        """
        self.rng_seed = rng_seed

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : :obj:`np.ndarray`
            dense representation of shape (trial, time)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time)

        """

        np.random.seed(self.rng_seed)
        n_trials, n_time = sample.shape
        sample_shuff = np.zeros_like(sample)
        for t in range(n_trials):
            if not any(np.isnan(sample[t])):
                # mark first time point of state change with a nonzero number
                state_change = np.where(np.concatenate([[0], np.diff(sample[t])], axis=0) != 0)[0]
                # collect runs
                runs = []
                prev_beg = 0
                for curr_beg in state_change:
                    runs.append(np.arange(prev_beg, curr_beg))
                    prev_beg = curr_beg
                runs.append(np.arange(prev_beg, n_time))
                # shuffle runs
                rand_perm = np.random.permutation(len(runs))
                runs_shuff = [runs[idx] for idx in rand_perm]
                # index back into original labels with shuffled indices
                sample_shuff[t] = sample[t, np.concatenate(runs_shuff)]
            else:
                sample_shuff[t] = np.nan
        return sample_shuff

    def __repr__(self):
        return str('BlockShuffle(rng_seed=%i)' % self.rng_seed)


class SelectIdxs(Transform):
    """"Index-based subsampling of neural activity."""

    def __init__(self, idxs, sample_name=''):
        """

        Parameters
        ----------
        idxs : :obj:`array-like`
        sample_name : :obj:`str`, optional
            name of sample for printing

        """
        self.sample_name = sample_name
        self.idxs = idxs

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape of (trial, time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (trial, time, n_channels)

        """
        return sample[:, :, self.idxs]

    def __repr__(self):
        return str('SelectIndxs(idxs=idxs, sample_name=%s)' % self.sample_name)
