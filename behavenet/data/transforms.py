import numpy as np
import copy
from skimage import transform


class Compose(object):
    """
    Composes several transforms together. Adapted from pytorch source code:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    Super hacky way of keeping track of specific neural populations

    Args:
        transforms (list of `Transform` objects): list of transforms to compose

    Example:
        >> Compose([
        >>     transforms.Subsample('mctx'),
        >>     transforms.Threshold(threshold=1.0, bin_size=25),
        >> ])
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
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class GetMask(object):

    # TODO: update GetMask for DLC likelihoods
    def __init__(self, ll_thresh, depth_thresh):
        self.ll_thresh = ll_thresh
        self.depth_thresh = depth_thresh

    def __call__(self, signal1, signal2):
        # tranform the signal
        signal1 = signal1 < self.ll_thresh  # ll < ll thresh
        signal2 = signal2 > self.depth_thresh  # depth > depth thresh
        # mask = ll < ll thresh || depth > depth thresh
        signal = ((signal1.astype('int')+signal2.astype('int'))<2).astype('int') #((signal1+signal2)<2).astype('int')
        return signal


class ClipNormalize(object):

    def __init__(self, clip_val=100):
        self.clip_val = clip_val

    def __call__(self, signal):
        signal = np.minimum(signal, self.clip_val)
        signal = signal / self.clip_val
        return signal


class Resize(object):
    """Resize the sample images"""

    def __init__(self, size=(128, 128), order=1):
        """
        Args:
            size (int or tuple): desired output size for each image
            order (int): interpolation order
        """
        assert isinstance(size, (tuple, int))
        self.order = order
        if isinstance(size, tuple):
            self.x = size[0]
            self.y = size[1]
        else:
            self.x = self.y = size

    def __call__(self, sample):
        """Assumes sample is of size (batch, channels, height, width)"""

        sh = sample.shape

        sample = transform.resize(
            sample, (sh[0], sh[1], self.x, self.y), order=self.order)

        return sample


class Threshold(object):

    def __init__(self, threshold, bin_size):
        """

        Args:
            threshold (float): Hz
            bin_size (float): ms
        """
        self.threshold = threshold
        self.bin_size = bin_size

    def __call__(self, sample):
        """
        Assumes sample is of size (trial x batch/time x predictors)

        Calculates firing rate over all trials/time points
        """

        # get firing rates
        frs = np.squeeze(np.mean(sample, axis=(0, 1))) / (self.bin_size * 1e-3)
        fr_mask = frs > self.threshold

        # get rid of neurons below fr threshold
        sample = sample[:, :, fr_mask]

        # # !! PROBLEM HERE !!
        # # get rid of indices if they are below threshold
        # reg_indxs = copy.copy(region_indxs)
        # keep_indxs = np.where(fr_mask)[0]
        # for region in reg_indxs.keys():
        #     # get overlap between region indices and keep indices
        #     reg_keep_indxs = np.intersect1d(keep_indxs, reg_indxs[region])
        #     reg_indxs[region] = reg_keep_indxs
        #
        # # get rid of regions if they have no neurons
        # keys = reg_indxs.keys()
        # keys_to_remove = []
        # for k in keys:
        #     if len(reg_indxs[k]) == 0:
        #         keys_to_remove.append(k)
        #
        # for k in keys_to_remove:
        #     reg_indxs.pop(k)

        return sample.astype(np.float)  #, reg_indxs


class ZScore(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        """Assumes sample is of size (trial x batch/time x predictors)"""
        sample -= np.mean(sample, axis=(0, 1))
        sample /= np.std(sample, axis=(0, 1))
        return sample


class MakeOneHot(object):
    """Turn a categorical vector into a one-hot vector"""

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Assumes sample is of size (trial x batch/time)
        Also assumes that K classes are identified by the numbers 0:K-1
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


class BlockShuffle(object):
    """Shuffle blocks of contiguous discrete states within each trial"""

    def __init__(self, rng_seed):
        self.rng_seed = rng_seed

    def __call__(self, sample):
        """Assumes sample is a dense rep of size (trial x batch/time)"""

        np.random.seed(self.rng_seed)

        # # collapse from one-hot to dense representation
        # state = np.where(self.data)[1]

        n_trials, n_time = sample.shape

        sample_shuff = np.zeros_like(sample)

        for t in range(n_trials):

            if not any(np.isnan(sample[t])):

                # mark first time point of state change with a nonzero number
                state_change = np.where(
                    np.concatenate([[0], np.diff(sample[t])], axis=0) != 0)[0]

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


class SelectRegion(object):
    """"Region-based subsampling"""

    def __init__(self, region):
        self.region = region

    def __call__(self, sample):
        raise NotImplementedError
