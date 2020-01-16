"""Encoding/decoding models implemented in PyTorch."""

import torch
from torch import nn


class Decoder(nn.Module):
    """General wrapper class for encoding/decoding models."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - model_type (:obj:`str`): 'ff' | 'ff-mv' | 'lstm'
            - input_size (:obj:`int`)
            - output_size (:obj:`int`)
            - n_hid_layers (:obj:`int`)
            - n_hid_units (:obj:`int`)
            - n_lags (:obj:`int`): number of lags in input data to use for temporal convolution
            - noise_dist (:obj:`str`): 'gaussian' | 'gaussian-full' | 'poisson' | 'categorical'
            - activation (:obj:`str`): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'

        """
        super().__init__()
        self.hparams = hparams
        if hparams['model_type'] == 'ff' or hparams['model_type'] == 'ff-mv':
            self.model = NN(hparams)
        elif hparams['model_type'] == 'lstm':
            self.model = LSTM(hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % hparams['model_type'])

    def __str__(self):
        """Pretty print model architecture."""
        return self.model.__str__()

    def forward(self, x):
        """Process input data."""
        return self.model(x)


class NN(nn.Module):
    """Feedforward neural network model."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\nNN architecture\n'
        format_str += '---------------\n'
        for i, module in enumerate(self.decoder):
            format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model."""

        self.decoder = nn.ModuleList()

        global_layer_num = 0

        in_size = self.hparams['input_size']

        # first layer is 1d conv for incorporating past/future neural activity
        if self.hparams['n_hid_layers'] == 0:
            out_size = self.hparams['output_size']
        else:
            out_size = self.hparams['n_hid_units']


        layer = nn.Conv1d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        self.decoder.add_module(name, layer)
        self.final_layer = name

        # add activation
        if self.hparams['n_hid_layers'] == 0:
            if self.hparams['noise_dist'] == 'gaussian':
                activation = None
            elif self.hparams['noise_dist'] == 'gaussian-full':
                activation = None
            elif self.hparams['noise_dist'] == 'poisson':
                activation = nn.Softplus()
            elif self.hparams['noise_dist'] == 'categorical':
                activation = None
            else:
                raise ValueError('"%s" is an invalid noise dist' % self.hparams['noise_dist'])
        else:
            if self.hparams['activation'] == 'linear':
                activation = None
            elif self.hparams['activation'] == 'relu':
                activation = nn.ReLU()
            elif self.hparams['activation'] == 'lrelu':
                activation = nn.LeakyReLU(0.05)
            elif self.hparams['activation'] == 'sigmoid':
                activation = nn.Sigmoid()
            elif self.hparams['activation'] == 'tanh':
                activation = nn.Tanh()
            else:
                raise ValueError(
                    '"%s" is an invalid activation function' % self.hparams['activation'])

        if activation:
            name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
            self.decoder.add_module(name, activation)

        # add layer for data-dependent precision matrix if required
        if self.hparams['n_hid_layers'] == 0 and self.hparams['noise_dist'] == 'gaussian-full':
            # build sqrt of precision matrix
            self.precision_sqrt = nn.Linear(in_features=in_size, out_features=out_size ** 2)
        else:
            self.precision_sqrt = None

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # loop over hidden layers (0 layers <-> linear regression)
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                out_size = self.hparams['output_size']
            else:
                out_size = self.hparams['n_hid_units']

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.decoder.add_module(name, layer)
            self.final_layer = name

            # add activation
            if i_layer == self.hparams['n_hid_layers'] - 1:
                if self.hparams['noise_dist'] == 'gaussian':
                    activation = None
                elif self.hparams['noise_dist'] == 'gaussian-full':
                    activation = None
                elif self.hparams['noise_dist'] == 'poisson':
                    activation = nn.Softplus()
                elif self.hparams['noise_dist'] == 'categorical':
                    activation = None
                else:
                    raise ValueError('"%s" is an invalid noise dist' % self.hparams['noise_dist'])
            else:
                if self.hparams['activation'] == 'linear':
                    activation = None
                elif self.hparams['activation'] == 'relu':
                    activation = nn.ReLU()
                elif self.hparams['activation'] == 'lrelu':
                    activation = nn.LeakyReLU(0.05)
                elif self.hparams['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif self.hparams['activation'] == 'tanh':
                    activation = nn.Tanh()
                else:
                    raise ValueError(
                        '"%s" is an invalid activation function' % self.hparams['activation'])

            if activation:
                self.decoder.add_module(
                    '%s_%02i' % (self.hparams['activation'], global_layer_num), activation)

            # add layer for data-dependent precision matrix if required
            if i_layer == self.hparams['n_hid_layers'] - 1 \
                    and self.hparams['noise_dist'] == 'gaussian-full':
                # build sqrt of precision matrix
                self.precision_sqrt = nn.Linear(in_features=in_size, out_features=out_size ** 2)
            else:
                self.precision_sqrt = None

            # update layer info
            global_layer_num += 1
            in_size = out_size

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
            - y (:obj:`torch.Tensor`): precision matrix prediction of model (when using 'ff-mv')

        """
        y = None
        for name, layer in self.decoder.named_children():

            # get data-dependent precision matrix if required
            if name == self.final_layer and self.hparams['noise_dist'] == 'gaussian-full':
                y = self.precision_sqrt(x)
                y = y.reshape(-1, self.hparams['output_size'], self.hparams['output_size'])
                y = torch.bmm(y, y.transpose(1, 2))

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer(x.transpose(1, 0).unsqueeze(0)).squeeze().transpose(1, 0)
            else:
                x = layer(x)

        return x, y

    def freeze(self):
        """Prevent updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class LSTM(nn.Module):
    """LSTM neural network model.

    Note
    ----
    Not currently implemented

    """

    def __init__(self, hparams):
        super().__init__()
        raise NotImplementedError
