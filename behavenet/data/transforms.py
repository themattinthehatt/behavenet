"""Tranform classes to process data.

Data generator objects can apply these transforms to batches of data upon loading.
"""

import numpy as np
# from skimage import transform


class Compose(object):
    """Composes several transforms together.

    Adapted from pytorch source code:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    Example
    -------
    .. code-block:: python

        >> Compose([
        >>     behavenet.data.transforms.SelectIdxs(idxs),
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
            dense representation of shape (time)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time)

        """

        np.random.seed(self.rng_seed)
        n_time = len(sample)
        if not any(np.isnan(sample)):
            # mark first time point of state change with a nonzero number
            state_change = np.where(np.concatenate([[0], np.diff(sample)], axis=0) != 0)[0]
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
            sample_shuff = sample[np.concatenate(runs_shuff)]
        else:
            sample_shuff = np.full(n_time, fill_value=np.nan)
        return sample_shuff

    def __repr__(self):
        return str('BlockShuffle(rng_seed=%i)' % self.rng_seed)


class ClipNormalize(Transform):
    """Clip upper level of signal and divide by clip value."""

    def __init__(self, clip_val):
        """

        Parameters
        ----------
        clip_val : :obj:`float`
            signal values above this will be set to this value, then divided by this value so that
            signal maximum is 1

        """
        if clip_val <= 0:
            raise ValueError('clip value must be positive')
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


class MakeOneHot(Transform):
    """Turn a categorical vector into a one-hot vector."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """Assumes that K classes are identified by the numbers 0:K-1.

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (time)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, K)

        """
        if len(sample.shape) == 2:  # weak test for if sample is already onehot
            onehot = sample
        else:
            n_time = len(sample)
            n_classes = int(np.nanmax(sample))
            onehot = np.zeros((n_time, n_classes + 1))
            if not any(np.isnan(sample)):
                onehot[np.arange(n_time), sample.astype('int')] = 1
            else:
                onehot[:] = np.nan

        return onehot

    def __repr__(self):
        return 'MakeOneHot()'


class MakeOneHot2D(Transform):
    """Turn an array of continuous values into an array of one-hot 2D arrays.

    Assumes that there are an even number of values in the input array, and that the first half
    are x values and the second half are y values.

    For example, if y_pixels=128 and x_pixels=128 (inputs to constructor), and the input array is
    [64, 34, 56, 102], the output array is of shape (2, 128, 128) where all values are zero except:
    output[0, 56, 64] = 1
    output[1, 102, 34] = 1

    """

    def __init__(self, y_pixels, x_pixels):
        """

        Parameters
        ----------
        y_pixels : :obj:`int`
            y_pixels of output 2D array
        x_pixels : :obj:`int`
            x_pixels of output 2D array

        """
        self.y_pixels = y_pixels
        self.x_pixels = x_pixels

    def __call__(self, sample):
        """Assumes that x-values are first half, y-values are second half.

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (time, n_labels * 2)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, n_labels, y_pix, x_pix)

        """
        time, n_labels_ = sample.shape
        n_labels = int(n_labels_ / 2)
        labels_2d = np.zeros((time, n_labels, self.y_pixels, self.x_pixels))

        x_vals = sample[:, :n_labels]
        x_vals[np.isnan(x_vals)] = -1  # set nans to 0
        x_vals[x_vals > self.x_pixels - 1] = self.x_pixels - 1
        x_vals[x_vals < 0] = 0
        x_vals = np.round(x_vals).astype(np.int)

        y_vals = sample[:, n_labels:]
        y_vals[np.isnan(y_vals)] = -1  # set nans to 0
        y_vals[y_vals > self.y_pixels - 1] = self.y_pixels - 1
        y_vals[y_vals < 0] = 0
        y_vals = np.round(y_vals).astype(np.int)

        for n in range(n_labels):
            labels_2d[np.arange(time), n, y_vals[:, n], x_vals[:, n]] = 1
        return labels_2d

    def __repr__(self):
        return str('MakeOneHot2D(y_pixels=%i, x_pixels=%i)' % (self.y_pixels, self.x_pixels))


class MotionEnergy(Transform):
    """Compute motion energy across batch dimension."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : :obj:`np.ndarray`
            input shape is (time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, n_channels)

        """
        return np.vstack([np.zeros((1, sample.shape[1])), np.abs(np.diff(sample, axis=0))])

    def __repr__(self):
        return 'MotionEnergy()'


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
            input shape of (time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, n_channels)

        """
        return sample[:, self.idxs]

    def __repr__(self):
        return str('SelectIndxs(idxs=idxs, sample_name=%s)' % self.sample_name)


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
        if bin_size <= 0:
            raise ValueError('bin size must be positive')
        if threshold < 0:
            raise ValueError('threshold must be non-negative')

        self.threshold = threshold
        self.bin_size = bin_size

    def __call__(self, sample):
        """Calculates firing rate over all time points and thresholds.

        Parameters
        ----------
        sample: :obj:`np.ndarray`
            input shape is (time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, n_channels)

        """
        # get firing rates
        frs = np.squeeze(np.mean(sample, axis=0)) / (self.bin_size * 1e-3)
        fr_mask = frs > self.threshold
        # get rid of neurons below fr threshold
        sample = sample[:, fr_mask]
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
            input shape is (time, n_channels)

        Returns
        -------
        :obj:`np.ndarray`
            output shape is (time, n_channels)

        """
        sample -= np.mean(sample, axis=0)
        sample /= np.std(sample, axis=0)
        return sample

    def __repr__(self):
        return 'ZScore()'


# class Resize(Transform):
#     """Resize the sample images."""
#
#     def __init__(self, size=(128, 128), order=1):
#         """
#
#         Parameters
#         ----------
#         size : :obj:`int` or :obj:`tuple`
#             desired output size for each image; if type is :obj:`int`, the same value is used for
#             both height and width
#         order : :obj:`int`
#             interpolation order
#
#         """
#         assert isinstance(size, (tuple, int))
#         self.order = order
#         if isinstance(size, tuple):
#             self.x = size[0]
#             self.y = size[1]
#         else:
#             self.x = self.y = size
#
#     def __call__(self, sample):
#         """
#
#         Parameters
#         ----------
#         sample: :obj:`np.ndarray`
#             input shape is (trial, time, n_channels)
#
#         Returns
#         -------
#         :obj:`np.ndarray`
#             output shape is (trial, time, n_channels)
#
#         """
#         sh = sample.shape
#         sample = transform.resize(sample, (sh[0], sh[1], self.y, self.x), order=self.order)
#         return sample
#
#     def __repr__(self):
#         return str('Resize(size=(%i, %i))' % (self.y, self.x))
