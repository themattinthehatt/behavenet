import torch
from torch.utils import data
# import h5py
from collections import OrderedDict
from torchvision import transforms
import numpy as np
from skimage import transform


class GetMask(object):

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
        """Assumes image stack is of form (batch, channels, height, width)"""

        sh = sample.shape

        sample = transform.resize(
            sample, (sh[0], sh[1], self.x, self.y), order=self.order)

        return sample


class Threshold(object):

    def __init__(self, threshold, binsize):
        """
        Args:
            threshold:
            binsize:
        """
        # TODO: implement Threshold transform
        raise NotImplementedError
