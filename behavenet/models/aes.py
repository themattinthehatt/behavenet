"""Autoencoder models implemented in PyTorch."""

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn
import torch.nn.functional as functional
import behavenet.fitting.losses as losses
from behavenet.models.base import BaseModule, BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = [
    'ConvAEEncoder', 'ConvAEDecoder', 'LinearAEEncoder', 'LinearAEDecoder', 'AE', 'ConditionalAE',
    'AEMSP', 'load_pretrained_ae']


class ConvAEEncoder(BaseModule):
    """Convolutional encoder."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - 'model_class' (:obj:`str`): 'ae' | 'vae'
            - 'n_ae_latents' (:obj:`int`)
            - 'fit_sess_io_layers; (:obj:`bool`): fit session-specific input/output layers
            - 'ae_encoding_x_dim' (:obj:`list`)
            - 'ae_encoding_y_dim' (:obj:`list`)
            - 'ae_encoding_n_channels' (:obj:`list`)
            - 'ae_encoding_kernel_size' (:obj:`list`)
            - 'ae_encoding_stride_size' (:obj:`list`)
            - 'ae_encoding_x_padding' (:obj:`list`)
            - 'ae_encoding_y_padding' (:obj:`list`)
            - 'ae_encoding_layer_type' (:obj:`list`)

        """
        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.build_model()

    def __str__(self):
        """Pretty print encoder architecture."""
        format_str = 'Encoder architecture:\n'
        i = 0
        for module in self.encoder:
            format_str += str('    {:02d}: {}\n'.format(i, module))
            i += 1
        # final ff layer
        format_str += str('    {:02d}: {}\n'.format(i, self.FF))
        return format_str

    def build_model(self):
        """Construct the encoder."""

        self.encoder = nn.ModuleList()
        # Loop over layers (each conv/batch norm/max pool/relu chunk counts as
        # one layer for global_layer_num)
        global_layer_num = 0
        for i_layer in range(0, len(self.hparams['ae_encoding_n_channels'])):

            # only add if conv layer (checks within this for max pool layer)
            if self.hparams['ae_encoding_layer_type'][i_layer] == 'conv':

                # convolution layer
                args = self._get_conv2d_args(i_layer, global_layer_num)
                if self.hparams.get('fit_sess_io_layers', False) and i_layer == 0:
                    module = nn.ModuleList([
                        nn.Conv2d(
                            in_channels=args['in_channels'],
                            out_channels=args['out_channels'],
                            kernel_size=args['kernel_size'],
                            stride=args['stride'],
                            padding=args['padding'])
                        for _ in range(self.hparams['n_datasets'])])
                    self.encoder.add_module(
                        str('conv%i_sess_io_layers' % global_layer_num), module)
                else:
                    module = nn.Conv2d(
                        in_channels=args['in_channels'],
                        out_channels=args['out_channels'],
                        kernel_size=args['kernel_size'],
                        stride=args['stride'],
                        padding=args['padding'])
                    self.encoder.add_module(
                        str('conv%i' % global_layer_num), module)

                # batch norm layer
                if self.hparams['ae_batch_norm']:
                    module = nn.BatchNorm2d(
                        self.hparams['ae_encoding_n_channels'][i_layer],
                        momentum=self.hparams.get('ae_batch_norm_momentum', 0.1),
                        track_running_stats=self.hparams.get('track_running_stats', True))
                    self.encoder.add_module(
                        str('batchnorm%i' % global_layer_num), module)

                # max pool layer
                if i_layer < (len(self.hparams['ae_encoding_n_channels'])-1) \
                        and (self.hparams['ae_encoding_layer_type'][i_layer+1] == 'maxpool'):
                    args = self._get_maxpool2d_args(i_layer)
                    module = nn.MaxPool2d(
                        kernel_size=args['kernel_size'],
                        stride=args['stride'],
                        padding=args['padding'],
                        return_indices=args['return_indices'],
                        ceil_mode=args['ceil_mode'])
                    self.encoder.add_module(
                        str('maxpool%i' % global_layer_num), module)

                # leaky ReLU
                self.encoder.add_module(
                    str('relu%i' % global_layer_num), nn.LeakyReLU(0.05))
                global_layer_num += 1

        # final ff layer to latents
        last_conv_size = self.hparams['ae_encoding_n_channels'][-1] \
            * self.hparams['ae_encoding_y_dim'][-1] \
            * self.hparams['ae_encoding_x_dim'][-1]
        self.FF = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])

        # If VAE model, have additional ff layer to latent variances
        if self.hparams.get('variational', False):
            self.logvar = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])

    def _get_conv2d_args(self, layer, global_layer):

        if layer == 0:
            if self.hparams['model_class'] == 'cond-ae' and \
                    self.hparams.get('conditional_encoder', False):
                # labels will be appended to input if using conditional autoencoder with
                # conditional encoder flag
                n_labels = int(self.hparams['n_labels'] / 2)  # 'n_labels' key includes x/y coords
            else:
                n_labels = 0
            in_channels = self.hparams['ae_input_dim'][0] + n_labels
        else:
            in_channels = self.hparams['ae_encoding_n_channels'][layer - 1]

        out_channels = self.hparams['ae_encoding_n_channels'][layer]
        kernel_size = self.hparams['ae_encoding_kernel_size'][layer]
        stride = self.hparams['ae_encoding_stride_size'][layer]

        x_pad_0 = self.hparams['ae_encoding_x_padding'][layer][0]
        x_pad_1 = self.hparams['ae_encoding_x_padding'][layer][1]
        y_pad_0 = self.hparams['ae_encoding_y_padding'][layer][0]
        y_pad_1 = self.hparams['ae_encoding_y_padding'][layer][1]
        if (x_pad_0 == x_pad_1) and (y_pad_0 == y_pad_1):
            # if symmetric padding
            padding = (y_pad_0, x_pad_0)
        else:
            module = nn.ZeroPad2d((x_pad_0, x_pad_1, y_pad_0, y_pad_1))
            self.encoder.add_module(str('zero_pad%i' % global_layer), module)
            padding = 0

        args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding}
        return args

    def _get_maxpool2d_args(self, layer):
        args = {
            'kernel_size': int(self.hparams['ae_encoding_kernel_size'][layer + 1]),
            'stride': int(self.hparams['ae_encoding_stride_size'][layer + 1]),
            'padding': (
                self.hparams['ae_encoding_y_padding'][layer + 1][0],
                self.hparams['ae_encoding_x_padding'][layer + 1][0]),
            'return_indices': True}
        if self.hparams['ae_padding_type'] == 'valid':
            # no ceil mode in valid mode
            args['ceil_mode'] = False
        else:
            # using ceil mode instead of zero padding
            args['ceil_mode'] = True
        return args

    def forward(self, x, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - encoder output (:obj:`torch.Tensor`): shape (n_latents)
            - pool_idx (:obj:`list`): max pooling indices for each layer
            - output_size (:obj:`list`): output size for each layer

        """
        # loop over layers, have to collect pool_idx and output sizes if using max pooling to use
        # in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                target_output_size.append(x.size())
                x, idx = layer(x)
                pool_idx.append(idx)
            elif isinstance(layer, nn.ModuleList):
                x = layer[dataset](x)
            else:
                x = layer(x)

        # reshape for ff layer
        x = x.view(x.size(0), -1)
        if self.hparams.get('variational', False):
            return self.FF(x), self.logvar(x), pool_idx, target_output_size
        else:
            return self.FF(x), pool_idx, target_output_size


class ConvAEDecoder(BaseModule):
    """Convolutional decoder."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - 'model_class' (:obj:`str`): 'ae' | 'vae'
            - 'n_ae_latents' (:obj:`int`)
            - 'fit_sess_io_layers; (:obj:`bool`): fit session-specific input/output layers
            - 'ae_decoding_x_dim' (:obj:`list`)
            - 'ae_decoding_y_dim' (:obj:`list`)
            - 'ae_decoding_n_channels' (:obj:`list`)
            - 'ae_decoding_kernel_size' (:obj:`list`)
            - 'ae_decoding_stride_size' (:obj:`list`)
            - 'ae_decoding_x_padding' (:obj:`list`)
            - 'ae_decoding_y_padding' (:obj:`list`)
            - 'ae_decoding_layer_type' (:obj:`list`)
            - 'ae_decoding_starting_dim' (:obj:`list`)
            - 'ae_decoding_last_FF_layer' (:obj:`bool`)

        """
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print decoder architecture."""
        format_str = 'Decoder architecture:\n'
        # initial ff layer
        format_str += str('    {:02d}: {}\n'.format(0, self.FF))
        for i, module in enumerate(self.decoder):
            format_str += str('    {:02d}: {}\n'.format(i + 1, module))
        return format_str

    def build_model(self):
        """Construct the decoder."""

        # First ff layer (from latents to size of last encoding layer)
        first_conv_size = self.hparams['ae_decoding_starting_dim'][0] \
            * self.hparams['ae_decoding_starting_dim'][1] \
            * self.hparams['ae_decoding_starting_dim'][2]
        self.FF = nn.Linear(self.hparams['hidden_layer_size'], first_conv_size)

        self.decoder = nn.ModuleList()

        # Loop over layers (each unpool/convtranspose/batch norm/relu chunk
        # counts as one layer for global_layer_num)
        global_layer_num = 0
        self.conv_t_pads = {}

        for i_layer in range(0, len(self.hparams['ae_decoding_n_channels'])):

            # only add if conv transpose layer
            if self.hparams['ae_decoding_layer_type'][i_layer] == 'convtranspose':

                # unpooling layer
                if i_layer > 0 and \
                        (self.hparams['ae_decoding_layer_type'][i_layer-1] == 'unpool'):
                    module = nn.MaxUnpool2d(
                        kernel_size=(
                            int(self.hparams['ae_decoding_kernel_size'][i_layer-1]),
                            int(self.hparams['ae_decoding_kernel_size'][i_layer-1])),
                        stride=(
                            int(self.hparams['ae_decoding_stride_size'][i_layer-1]),
                            int(self.hparams['ae_decoding_stride_size'][i_layer-1])),
                        padding=(
                            self.hparams['ae_decoding_y_padding'][i_layer-1][0],
                            self.hparams['ae_decoding_x_padding'][i_layer-1][0]))
                    self.decoder.add_module(
                        str('maxunpool%i' % global_layer_num), module)

                # conv transpose layer
                args = self._get_convtranspose2d_args(i_layer, global_layer_num)
                if self.hparams.get('fit_sess_io_layers', False) \
                        and i_layer == (len(self.hparams['ae_decoding_n_channels']) - 1) \
                        and not self.hparams['ae_decoding_last_FF_layer']:
                    module = nn.ModuleList([
                        nn.ConvTranspose2d(
                            in_channels=args['in_channels'],
                            out_channels=args['out_channels'],
                            kernel_size=args['kernel_size'],
                            stride=args['stride'],
                            padding=args['padding'],
                            output_padding=args['output_padding'])
                        for _ in range(self.hparams['n_datasets'])])
                    self.decoder.add_module(
                        str('convtranspose%i_sess_io_layers' % global_layer_num), module)
                    self.conv_t_pads[str('convtranspose%i_sess_io_layers' % global_layer_num)] = \
                        self.conv_t_pads[str('convtranspose%i' % global_layer_num)]
                else:
                    module = nn.ConvTranspose2d(
                        in_channels=args['in_channels'],
                        out_channels=args['out_channels'],
                        kernel_size=args['kernel_size'],
                        stride=args['stride'],
                        padding=args['padding'],
                        output_padding=args['output_padding'])
                    self.decoder.add_module(
                        str('convtranspose%i' % global_layer_num), module)

                # batch norm + relu or sigmoid if last layer
                if i_layer == (len(self.hparams['ae_decoding_n_channels'])-1) \
                        and not self.hparams['ae_decoding_last_FF_layer']:
                    # last layer: no batch norm/sigmoid nonlin
                    self.decoder.add_module(
                        str('sigmoid%i' % global_layer_num), nn.Sigmoid())
                else:
                    if self.hparams['ae_batch_norm']:
                        module = nn.BatchNorm2d(
                            self.hparams['ae_decoding_n_channels'][i_layer],
                            momentum=self.hparams.get('ae_batch_norm_momentum', 0.1),
                            track_running_stats=self.hparams.get('track_running_stats', True))
                        self.decoder.add_module(
                            str('batchnorm%i' % global_layer_num), module)

                    self.decoder.add_module(
                        str('relu%i' % global_layer_num), nn.LeakyReLU(0.05))
                global_layer_num += 1

        # optional final ff layer (rarely used)
        if self.hparams['ae_decoding_last_FF_layer']:
            if self.hparams.get('fit_sess_io_layers', False):
                raise NotImplementedError
            # have last layer be feedforward if this is 1
            module = nn.Linear(
                self.hparams['ae_decoding_x_dim'][-1]
                * self.hparams['ae_decoding_y_dim'][-1]
                * self.hparams['ae_decoding_n_channels'][-1],
                self.hparams['ae_input_dim'][0]
                * self.hparams['ae_input_dim'][1]
                * self.hparams['ae_input_dim'][2])
            self.decoder.add_module(
                str('last_ff%i' % global_layer_num), module)
            self.decoder.add_module(
                str('sigmoid%i' % global_layer_num), nn.Sigmoid())

    def _get_convtranspose2d_args(self, layer, global_layer):

        # input channels
        if layer == 0:
            in_channels = self.hparams['ae_decoding_starting_dim'][0]
        else:
            in_channels = self.hparams['ae_decoding_n_channels'][layer - 1]

        out_channels = self.hparams['ae_decoding_n_channels'][layer]
        kernel_size = (
            self.hparams['ae_decoding_kernel_size'][layer],
            self.hparams['ae_decoding_kernel_size'][layer])
        stride = (
            self.hparams['ae_decoding_stride_size'][layer],
            self.hparams['ae_decoding_stride_size'][layer])

        # input/output padding
        x_pad_0 = self.hparams['ae_decoding_x_padding'][layer][0]
        x_pad_1 = self.hparams['ae_decoding_x_padding'][layer][1]
        y_pad_0 = self.hparams['ae_decoding_y_padding'][layer][0]
        y_pad_1 = self.hparams['ae_decoding_y_padding'][layer][1]
        if self.hparams['ae_padding_type'] == 'valid':
            # calculate output padding to get back original input shape
            if layer > 0:
                input_y = self.hparams['ae_decoding_y_dim'][layer - 1]
            else:
                input_y = self.hparams['ae_decoding_starting_dim'][1]
            y_output_padding = \
                self.hparams['ae_decoding_y_dim'][layer] \
                - ((input_y - 1) * self.hparams['ae_decoding_stride_size'][layer]
                    + self.hparams['ae_decoding_kernel_size'][layer])

            if layer > 0:
                input_x = self.hparams['ae_decoding_x_dim'][layer - 1]
            else:
                input_x = self.hparams['ae_decoding_starting_dim'][2]
            x_output_padding = \
                self.hparams['ae_decoding_x_dim'][layer] \
                - ((input_x - 1) * self.hparams['ae_decoding_stride_size'][layer]
                    + self.hparams['ae_decoding_kernel_size'][layer])

            input_padding = (y_pad_0, x_pad_0)
            output_padding = (y_output_padding, x_output_padding)

            self.conv_t_pads[str('convtranspose%i' % global_layer)] = None

        elif self.hparams['ae_padding_type'] == 'same':
            if (x_pad_0 == x_pad_1) and (y_pad_0 == y_pad_1):
                input_padding = (y_pad_0, x_pad_0)
                output_padding = 0
                self.conv_t_pads[str('convtranspose%i' % global_layer)] = None
            else:
                # If uneven padding originally, don't pad here and do
                # it in forward()
                input_padding = 0
                output_padding = 0
                self.conv_t_pads[str('convtranspose%i' % global_layer)] = [
                    x_pad_0, x_pad_1, y_pad_0, y_pad_1]
        else:
            raise ValueError(
                '"%s" is not a valid padding type' % self.hparams['ae_padding_type'])

        args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': input_padding,
            'output_padding': output_padding}
        return args

    def forward(self, x, pool_idx=None, target_output_size=None, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        pool_idx : :obj:`list`
            max pooling indices from encoder for unpooling
        target_output_size : :obj:`list`
            layer-specific output sizes from encoder for unpooling
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`torch.Tensor`
            shape (n_input_channels, y_pix, x_pix)

        """
        # First ff layer/resize to be convolutional input
        x = self.FF(x)
        x = x.view(
            x.size(0),
            self.hparams['ae_decoding_starting_dim'][0],
            self.hparams['ae_decoding_starting_dim'][1],
            self.hparams['ae_decoding_starting_dim'][2])

        for name, layer in self.decoder.named_children():
            if isinstance(layer, nn.MaxUnpool2d):
                idx = pool_idx.pop(-1)
                outsize = target_output_size.pop(-1)
                x = layer(x, idx, outsize)
            elif isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                if self.conv_t_pads[name] is not None:
                    # asymmetric padding for convtranspose layer if necessary
                    # (-i does cropping!)
                    x = functional.pad(x, [-i for i in self.conv_t_pads[name]])
            elif isinstance(layer, nn.ModuleList):
                x = layer[dataset](x)
                if self.conv_t_pads[name] is not None:
                    # asymmetric padding for convtranspose layer if necessary
                    # (-i does cropping!)
                    x = functional.pad(x, [-i for i in self.conv_t_pads[name]])
            elif isinstance(layer, nn.Linear):
                x = x.view(x.shape[0], -1)
                x = layer(x)
                x = x.view(
                    -1,
                    self.hparams['ae_input_dim'][0],
                    self.hparams['ae_input_dim'][1],
                    self.hparams['ae_input_dim'][2])
            else:
                x = layer(x)

        return x


class LinearAEEncoder(BaseModule):
    """Linear encoder."""

    def __init__(self, n_latents, input_size):
        """

        Parameters
        ----------
        n_latents : :obj:`int`
            number of latents in encoder output
        input_size : :obj:`array-like`
            shape of encoder input as (n_channels, y_pix, x_pix)

        """
        super().__init__()
        self.n_latents = n_latents
        self.input_size = input_size
        self.encoder = None
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print the encoder architecture."""
        format_str = 'Encoder architecture:\n'
        format_str += str('    {}\n'.format(self.encoder))
        return format_str

    def build_model(self):
        """Construct the encoder."""
        self.encoder = nn.Linear(
            out_features=self.n_latents,
            in_features=np.prod(self.input_size),
            bias=True)

    def forward(self, x, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - encoder output (:obj:`torch.Tensor`): shape (n_latents)
            - :obj:`NoneType`: to match convolutional encoder outputs
            - :obj:`NoneType`: to match convolutional encoder outputs

        """
        x = x.view(x.size(0), -1)
        return self.encoder(x), None, None


class LinearAEDecoder(BaseModule):
    """Linear decoder."""

    def __init__(self, n_latents, output_size, encoder=None):
        """

        Parameters
        ----------
        n_latents : :obj:`int`
            number of latents in decoder input
        output_size : :obj:`array-like`
            shape of decoder output as (n_channels, y_pix, x_pix)
        encoder : :obj:`nn.Module` object or :obj:`NoneType`
            if :obj:`nn.Module` object, use the transpose weights for the decoder plus an
            independent bias; otherwise fit a separate set of parameters

        """
        super().__init__()
        self.n_latents = n_latents
        self.output_size = output_size
        self.encoder = encoder
        self.decoder = None
        self.build_model()

    def __str__(self):
        """Pretty print the decoder architecture."""
        format_str = 'Decoder architecture:\n'
        if self.bias is not None:
            format_str += str('    Encoder weights transposed (plus independent bias)\n')
        else:
            format_str += str('    {}\n'.format(self.decoder))
        return format_str

    def build_model(self):
        """Construct the decoder."""
        if self.encoder is None:
            self.decoder = nn.Linear(
                out_features=np.prod(self.output_size),
                in_features=self.n_latents,
                bias=True)
        else:
            self.bias = nn.Parameter(
                torch.zeros(int(np.prod(self.output_size))), requires_grad=True)

    def forward(self, x, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`torch.Tensor`
            shape (n_input_channels, y_pix, x_pix)

        """
        if self.encoder is None:
            x = self.decoder(x)
        else:
            x = functional.linear(x, self.encoder.encoder.weight.t()) + self.bias
        # reshape
        x = x.view(x.size(0), *self.output_size)
        return x


class AE(BaseModel):
    """Base autoencoder class.

    This class can construct both linear and convolutional autoencoders. The linear autoencoder
    utilizes a single hidden layer, dense feedforward layers (i.e. not convolutional), and the
    encoding and decoding weights are tied to more closely resemble PCA/SVD. The convolutional
    autoencoder architecture is defined by various keys in the dict that serves as the constructor
    input. See the :mod:`behavenet.models.ae_model_architecture_generator` module to see examples
    for how this is done.
    """

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - 'model_type' (:obj:`int`): 'conv' | 'linear'
            - 'model_class' (:obj:`str`): 'ae'
            - 'y_pixels' (:obj:`int`)
            - 'x_pixels' (:obj:`int`)
            - 'n_input_channels' (:obj:`int`)
            - 'n_ae_latents' (:obj:`int`)
            - 'fit_sess_io_layers; (:obj:`bool`): fit session-specific input/output layers
            - 'ae_encoding_x_dim' (:obj:`list`)
            - 'ae_encoding_y_dim' (:obj:`list`)
            - 'ae_encoding_n_channels' (:obj:`list`)
            - 'ae_encoding_kernel_size' (:obj:`list`)
            - 'ae_encoding_stride_size' (:obj:`list`)
            - 'ae_encoding_x_padding' (:obj:`list`)
            - 'ae_encoding_y_padding' (:obj:`list`)
            - 'ae_encoding_layer_type' (:obj:`list`)
            - 'ae_decoding_x_dim' (:obj:`list`)
            - 'ae_decoding_y_dim' (:obj:`list`)
            - 'ae_decoding_n_channels' (:obj:`list`)
            - 'ae_decoding_kernel_size' (:obj:`list`)
            - 'ae_decoding_stride_size' (:obj:`list`)
            - 'ae_decoding_x_padding' (:obj:`list`)
            - 'ae_decoding_y_padding' (:obj:`list`)
            - 'ae_decoding_layer_type' (:obj:`list`)
            - 'ae_decoding_starting_dim' (:obj:`list`)
            - 'ae_decoding_last_FF_layer' (:obj:`bool`)

        """
        super().__init__()
        self.hparams = hparams
        self.model_type = self.hparams['model_type']
        self.img_size = (
            self.hparams['n_input_channels'],
            self.hparams['y_pixels'],
            self.hparams['x_pixels'])
        self.encoding = None
        self.decoding = None
        self.build_model()

    def __str__(self):
        """Pretty print the model architecture."""
        format_str = '\nAutoencoder architecture\n'
        format_str += '------------------------\n'
        format_str += self.encoding.__str__()
        format_str += self.decoding.__str__()
        format_str += '\n'
        return format_str

    def build_model(self):
        """Construct the model using hparams."""
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents']
        if self.model_type == 'conv':
            self.encoding = ConvAEEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            if self.hparams.get('fit_sess_io_layers', False):
                raise NotImplementedError
            n_latents = self.hparams['n_ae_latents']
            self.encoding = LinearAEEncoder(n_latents, self.img_size)
            self.decoding = LinearAEDecoder(n_latents, self.img_size, self.encoding)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

    def forward(self, x, dataset=None, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - y (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - x (:obj:`torch.Tensor`): hidden representation of shape (n_frames, n_latents)

        """
        if self.model_type == 'conv':
            x, pool_idx, outsize = self.encoding(x, dataset=dataset)
            y = self.decoding(x, pool_idx, outsize, dataset=dataset)
        elif self.model_type == 'linear':
            x, _, _ = self.encoding(x)
            y = self.decoding(x)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)
        return y, x

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate MSE loss for autoencoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers
        accumulate_grad : :obj:`bool`, optional
            accumulate gradient for training step
        chunk_size : :obj:`int`, optional
            batch is split into chunks of this size to keep memory requirements low

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): mse loss

        """

        x = data['images'][0]
        m = data['masks'][0] if 'masks' in data else None

        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        loss_val = 0
        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            x_hat, _ = self.forward(x_in, dataset=dataset)

            loss = losses.mse(x_in, x_hat, m_in)

            if accumulate_grad:
                loss.backward()

            # get loss value (weighted by batch size)
            loss_val += loss.item() * (idx_end - idx_beg)

        loss_val /= batch_size

        return {'loss': loss_val}


class ConditionalAE(AE):
    """Conditional autoencoder class.

    This class constructs conditional convolutional autoencoders. At the latent layer an additional
    set of variables, saved under the 'labels' key in the hdf5 data file, are concatenated with the
    latents before being reshaped into a 2D array for decoding.
    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain :obj:`n_labels` and
            :obj:`conditional_encoder`

        """
        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        super().__init__(hparams)

    def build_model(self):
        """Construct the model using hparams.

        The ConditionalAE is initialized when :obj:`model_class='cond-ae`, and currently only
        supports :obj:`model_type='conv` (i.e. no linear)
        """
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents'] + self.hparams['n_labels']
        self.encoding = ConvAEEncoder(self.hparams)
        self.decoding = ConvAEDecoder(self.hparams)

    def forward(self, x, dataset=None, labels=None, labels_2d=None, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data of shape (batch, n_channels, y_pix, x_pix)
        dataset : :obj:`int`
            used with session-specific io layers
        labels : :obj:`torch.Tensor` object
            continuous labels corresponding to input data, of shape (batch, n_labels)
        labels_2d: :obj:`torch.Tensor` object
            one-hot labels corresponding to input data, of shape (batch, n_labels, y_pix, x_pix);
            for a given frame, each channel corresponds to a label and is all zeros with a single
            value of one in the proper x/y position

        Returns
        -------
        :obj:`tuple`
            - y (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - x (:obj:`torch.Tensor`): hidden representation of shape (n_frames, n_latents)

        """
        if self.hparams['conditional_encoder']:
            # append label information to input
            x = torch.cat((x, labels_2d), dim=1)
        x, pool_idx, outsize = self.encoding(x, dataset=dataset)
        z = torch.cat((x, labels), dim=1)
        y = self.decoding(z, pool_idx, outsize, dataset=dataset)
        return y, x

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate MSE loss for autoencoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers
        accumulate_grad : :obj:`bool`, optional
            accumulate gradient for training step
        chunk_size : :obj:`int`, optional
            batch is split into chunks of this size to keep memory requirements low

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): mse loss

        """

        x = data['images'][0]
        y = data['labels'][0]
        m = data['masks'][0] if 'masks' in data else None
        if self.hparams['conditional_encoder']:
            # continuous labels transformed into 2d one-hot array as input to encoder
            y_2d = data['labels_sc'][0]
        else:
            y_2d = None

        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        loss_val = 0
        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            y_in = y[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            y_2d_in = y_2d[idx_beg:idx_end] if y_2d is not None else None
            x_hat, _ = self.forward(x_in, labels=y_in, labels_2d=y_2d_in, dataset=dataset)

            loss = losses.mse(x_in, x_hat, m_in)

            if accumulate_grad:
                loss.backward()

            # get loss value (weighted by batch size)
            loss_val += loss.item() * (idx_end - idx_beg)

        loss_val /= batch_size

        return {'loss': loss_val}


class AEMSP(AE):
    """Autoencoder class with matrix subspace projection for disentangling the latent space.

    This class constructs an autoencoder whose latent space is forced to learn a subspace that
    reconstructs a set of supervised labels; this subspace should be orthogonal to another subspace
    that does not contain information about the labels. These labels are saved under the 'labels'
    key in the hdf5 data file. For more information see:
    Li et al 2019, Latent Space Factorisation and Manipulation via Matrix Subspace Projection
    https://arxiv.org/pdf/1907.12385.pdf

    Note: the data in the hdf5 group `labels` should be mean/median centered, as no bias is learned
    in the transformation from the original latent space to the predicted labels.
    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain :obj:`n_labels` and
            :obj:`msp.alpha`

        """
        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        if hparams['n_ae_latents'] < hparams['n_labels']:
            raise ValueError('AEMSP model must contain at least as many latents as labels')

        self.n_latents = hparams['n_ae_latents']
        self.n_labels = hparams['n_labels']

        # linear projection from latents to labels
        self.projection = None

        # (inverse) linear projection from transformed latent space to original latent space; this
        # is used when manipulating latent/label space
        self.U = None

        super().__init__(hparams)

    def build_model(self):
        """Construct the model using hparams.

        The AEMSP is initialized when :obj:`model_class='cond-ae-msp`, and currently only supports
        :obj:`model_type='conv` (i.e. no linear)
        """
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents']
        self.encoding = ConvAEEncoder(self.hparams)
        self.decoding = ConvAEDecoder(self.hparams)
        self.projection = nn.Linear(self.n_latents, self.n_labels, bias=False)
        # construct U here so that it is in model state dict, but will be overwritten later
        with torch.no_grad():
            self.U = nn.Linear(self.n_latents, self.n_latents, bias=False)

    def forward(self, x, dataset=None, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data of shape (batch, n_channels, y_pix, x_pix)
        dataset : :obj:`int`, optional
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - x_hat (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - z (:obj:`torch.Tensor`): hidden representation of shape (n_frames, n_latents)
            - y (:obj:`torch.Tensor`): predicted labels of shape (n_frames, n_labels)

        """
        z, pool_idx, outsize = self.encoding(x, dataset=dataset)
        y = self.projection(z)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        return x_hat, z, y

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate MSE loss for autoencoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers
        accumulate_grad : :obj:`bool`, optional
            accumulate gradient for training step
        chunk_size : :obj:`int`, optional
            batch is split into chunks of this size to keep memory requirements low

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): total loss
            - 'loss_mse' (:obj:`float`): pixel mse loss
            - 'loss_msp' (:obj:`float`): combined msp loss
            - 'labels_r2' (:obj:`float`): variance-weighted $R^2$ of reconstructed labels

        """

        x = data['images'][0]
        y = data['labels'][0]
        m = data['masks'][0] if 'masks' in data else None

        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        loss_val = 0
        loss_mse_val = 0
        loss_msp_val = 0
        y_hat_all = []
        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            y_in = y[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            x_hat, z, y_hat = self.forward(x_in, dataset=dataset)

            # mse loss
            loss_mse = losses.mse(x_in, x_hat, m_in)

            # msp loss
            loss_msp = losses.mse(y_in, y_hat) + \
                losses.mse(z, torch.matmul(y_hat, self.projection.weight))
            # ^NOTE: transpose on projection weights implicitly performed due to layer def

            # combine
            loss = loss_mse + self.hparams['msp.alpha'] * loss_msp

            if accumulate_grad:
                loss.backward()

            # get loss value (weighted by batch size)
            loss_val += loss.item() * (idx_end - idx_beg)
            loss_mse_val += loss_mse.item() * (idx_end - idx_beg)
            loss_msp_val += loss_msp.item() * (idx_end - idx_beg)

            y_hat_all.append(y_hat.cpu().detach().numpy())

        loss_val /= batch_size
        loss_mse_val /= batch_size
        loss_msp_val /= batch_size

        # use variance-weighted r2s to ignore small-variance latents
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        r2 = r2_score(y.cpu().detach().numpy(), y_hat_all, multioutput='variance_weighted')

        loss_dict = {
            'loss': loss_val, 'loss_mse': loss_mse_val, 'loss_msp': loss_msp_val, 'labels_r2': r2}

        return loss_dict

    def save(self, filepath):
        """Save model parameters."""
        self.create_orthogonal_matrix()
        super().save(filepath)

    def create_orthogonal_matrix(self):
        """Use the learned projection matrix to construct a full rank orthogonal matrix."""
        from scipy.linalg import null_space

        # find nullspace of linear projection layer
        M = self.projection.weight.data.detach().cpu().numpy()  # M shape: [n_labels, n_latents]
        N = null_space(M)  # N shape: [n_latents, n_latents - n_labels]
        U = np.concatenate([M, N.T], axis=0)

        # create new torch tensor with full matrix
        # self.U = nn.Linear(self.n_latents, self.n_latents, bias=False)
        with torch.no_grad():
            self.U.weight = nn.Parameter(torch.from_numpy(U).float(), requires_grad=False)
        self.U.to(self.hparams['device'])

    def get_transformed_latents(self, inputs, dataset=None, as_numpy=True):
        """Return latents after they have been transformed using the orthogonal matrix U.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor` object
            - image tensor of shape (batch, n_channels, y_pix, x_pix)
            - latents tensor of shape (batch, n_ae_latents)
        dataset : :obj:`int`, optional
            used with session-specific io layers
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in transformed latent space

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)

        # check to see if inputs are images or latents
        if len(inputs.shape) == 2:
            input_type = 'latents'
        else:
            input_type = 'images'

        # get latents in original space
        if input_type == 'images':
            latents_og, pool_idx, outsize = self.encoding(inputs, dataset=dataset)
        else:
            latents_og = inputs

        # transform with complete orthogonal matrix U
        latents_tr = self.U(latents_og)
        if as_numpy:
            return latents_tr.cpu().detach().numpy()
        else:
            return latents_tr

    def get_inverse_transformed_latents(self, latents, as_numpy=True):
        """Take latents in transformed space to original space to push through decoder.

        Parameters
        ----------
        latents : :obj:`torch.Tensor` object
            shape (batch, n_ae_latents)
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in original latent space

        """
        if not isinstance(latents, torch.Tensor):
            latents = torch.Tensor(latents)
        latents_og = torch.matmul(latents, self.U.weight)
        if as_numpy:
            return latents_og.cpu().detach().numpy()
        else:
            return latents_og

    def sample(self, x=None, dataset=None, latents=None, labels=None, labels_2d=None):
        """Generate output given an input x and arbitrary labels and/or latents.

        How output image is generated:

        * if latents is not None and labels is not None, these are concatenated, tranformed to the
          original latent space, and pushed through the decoder
        * if latents is not None and labels is None, the input x is pushed through the encoder to
          produce the latents, these are transformed with the projection layer, and the resulting
          latents (n_latents - n_labels dimensions) are replaced with the user-defined latents.
          This vector (labels + latents) is then transformed back into the original latent space,
          and pushed through the decoder.
        * if latents is None and labels is not None, the input x is pushed through the encoder to
          produce the latents, these are transformed with the projection layer, and the resulting
          labels (n_labels dimensions) are replaced with the user-defined labels. This vector
          (latents + labels) is then transformed back into the original latent space, and pushed
          through the decoder.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object, optional
            input data of shape (batch, n_channels, y_pix, x_pix)
        dataset : :obj:`int`, optional
            used with session-specific io layers
        latents  : :obj:`np.ndarray` object, optional
            transformed latents of shape (batch, n_latents - n_labels)
        labels : :obj:`np.ndarray` object, optional
            continuous labels corresponding to input data, of shape (batch, n_labels)
        labels_2d : :obj:`torch.Tensor` object, optional
            one-hot labels corresponding to input data, of shape (batch, n_labels, y_pix, x_pix);
            for a given frame, each channel corresponds to a label and is all zeros with a single
            value of one in the proper x/y position

        Returns
        -------
        :obj:`torch.Tensor`
            output of shape (n_frames, n_channels, y_pix, x_pix)

        """

        if latents is None or labels is None:
            # push input image through encoder
            latents_tr = self.get_transformed_latents(x, dataset, labels_2d)
        else:
            # user-defined labels AND latents
            if latents is not None:
                batch_size = latents.shape[0]
            else:
                batch_size = labels.shape[0]
            # initialize outputs
            latents_tr = np.full(shape=(batch_size, self.n_latents), fill_value=np.nan)

        if labels is not None:
            # replace labels produced by input with user-defined labels
            latents_tr[:, :self.n_labels] = labels

        if latents is not None:
            # replace latents produced by input with user-defined latents
            latents_tr[:, self.n_labels:] = latents

        latents_tr_tensor = torch.from_numpy(latents_tr).float()

        # invert (tranform) back to original latent space space
        latents_tensor = torch.matmul(latents_tr_tensor, self.U.weight)
        # ^NOTE: transpose on projection weights implicitly performed due to layer def

        # push through decoder
        x_hat = self.decoding(latents_tensor, None, None, dataset=dataset)

        return x_hat


def load_pretrained_ae(model, hparams):
    """Load pretrained weights into already constructed AE model.

    Parameters
    ----------
    model : :obj:`behavenet.models.aes` object
        autoencoder-based model; AE, ConditionalAE, AEMSP
    hparams : :obj:`dict`
        needs to contain keys `model_type` and `pretrained_weights_path`

    Returns
    -------
    :obj:`behavenet.models.aes` object
        input model with updated weights

    """

    if hparams['model_type'] == 'conv' \
            and hparams.get('pretrained_weights_path', False) \
            and hparams['pretrained_weights_path'] is not None \
            and hparams['pretrained_weights_path'] != '':

        print('Loading pretrained weights')
        loaded_model_dict = torch.load(hparams['pretrained_weights_path'])

        if loaded_model_dict['encoding.FF.weight'].shape == model.encoding.FF.weight.shape:
            model.load_state_dict(loaded_model_dict, strict=False)
        else:
            print('PRETRAINED MODEL HAS DIFFERENT SPATIAL DIMENSIONS OR N LATENTS: ' +
                  'NOT LOADING FF PARAMETERS')
            del loaded_model_dict['encoding.FF.weight']
            del loaded_model_dict['encoding.FF.bias']
            del loaded_model_dict['decoding.FF.weight']
            del loaded_model_dict['decoding.FF.bias']

            # TODO: get rid of other latent-related parameters
            if model.hparams['model_class'] == 'vae':
                pass
            elif model.hparams['model_class'] == 'beta_tcvae':
                pass
            elif model.hparams['model_class'] == 'sss_vae':
                pass

            model.load_state_dict(loaded_model_dict, strict=False)

    elif hparams['model_type'] == 'linear' \
            and hparams.get('pretrained_weights_path', False) \
            and hparams['pretrained_weights_path'] is not None \
            and hparams['pretrained_weights_path'] != '':
        raise NotImplementedError('Loading pretrained weights with linear AE')

    else:
        print('Initializing with random weights')

    return model
