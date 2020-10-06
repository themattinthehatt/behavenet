"""Hierarchical encoding/decoding models implemented in PyTorch."""

import numpy as np
from sklearn.metrics import r2_score, accuracy_score
import torch
from torch import nn
import behavenet.fitting.losses as losses
from behavenet.models.base import BaseModule
from behavenet.models.decoders import Decoder


class HierarchicalDecoder(Decoder):
    """General wrapper class for hierarchical encoding/decoding models."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - model_type (:obj:`str`): 'mlp' | 'mlp-mv' | 'lstm'
            - input_size (:obj:`int`)
            - output_size (:obj:`int`)
            - n_hid_layers (:obj:`int`)
            - n_hid_units (:obj:`int`)
            - n_lags (:obj:`int`): number of lags in input data to use for temporal convolution
            - noise_dist (:obj:`str`): 'gaussian' | 'gaussian-full' | 'poisson' | 'categorical'
            - activation (:obj:`str`): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'

        """
        super().__init__(hparams)

    def build_model(self):
        """Construct the model using hparams."""

        # TODO
        # if self.hparams['model_type'] == 'mlp' or self.hparams['model_type'] == 'mlp-mv':
        #     self.model = MLP(self.hparams)
        # elif self.hparams['model_type'] == 'lstm':
        #     self.model = LSTM(self.hparams)
        # else:
        #     raise ValueError('"%s" is not a valid model type' % self.hparams['model_type'])

    def forward(self, x):
        """Process input data."""
        return self.model(x)

    def loss(self, data, accumulate_grad=True, chunk_size=200, **kwargs):
        # TODO
        pass


class HierarchicalMLP(BaseModule):
    """Feedforward neural network model."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        # TODO
        pass

    def build_model(self):
        """Construct the model."""
        # TODO
        pass

    def forward(self, x):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            shape of (time, neurons)

        Returns
        -------
        :obj:`tuple`
            - x (:obj:`torch.Tensor`): mean prediction of model
            - y (:obj:`torch.Tensor`): precision matrix prediction of model (when using 'mlp-mv')

        """
        # TODO
        pass
