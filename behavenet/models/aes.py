"""Autoencoder models implemented in PyTorch."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


class ConvAEEncoder(nn.Module):
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
        super(ConvAEEncoder, self).__init__()
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
                        momentum=self.hparams['ae_batch_norm_momentum'])
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
        if self.hparams['model_class'] == 'ae' or self.hparams['model_class'] == 'cond-ae':
            pass
        elif self.hparams['model_class'] == 'vae':
            raise NotImplementedError
            # self.logvar = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])
            # self.softplus = nn.Softplus()
        else:
            raise ValueError('Not valid model type')

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
        # Loop over layers, have to collect pool_idx and output sizes if using
        # max pooling to use in unpooling
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

        # Reshape for ff layer
        x = x.view(x.size(0), -1)

        if self.hparams['model_class'] == 'ae' or self.hparams['model_class'] == 'cond-ae':
            return self.FF(x), pool_idx, target_output_size
        elif self.hparams['model_class'] == 'vae':
            return NotImplementedError
        else:
            raise ValueError('"%s" is not a valid model class' % self.hparams['model_class'])

    def freeze(self):
        """Prevent updates to encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ConvAEDecoder(nn.Module):
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
        super(ConvAEDecoder, self).__init__()
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
                            momentum=self.hparams['ae_batch_norm_momentum'])
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

        if self.hparams['model_class'] == 'ae' or self.hparams['model_class'] == 'cond-ae':
            pass
        elif self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'labels-images':
            pass
        else:
            raise ValueError('Not valid model type')

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

    def forward(self, x, pool_idx, target_output_size, dataset=None):
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
                x = x.view(x.shape[0],-1)
                x = layer(x)
                x = x.view(
                    -1,
                    self.hparams['ae_input_dim'][0],
                    self.hparams['ae_input_dim'][1],
                    self.hparams['ae_input_dim'][2])
            else:
                x = layer(x)

        if self.hparams['model_class'] == 'ae' or self.hparams['model_class'] == 'cond-ae':
            return x
        elif self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'labels-images':
            return x
        else:
            raise ValueError('"%s" is not a valid model class' % self.hparams['model_class'])

    def freeze(self):
        """Prevent updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class LinearAEEncoder(nn.Module):
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

    def freeze(self):
        """Prevent updates to encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class LinearAEDecoder(nn.Module):
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

    def freeze(self):
        """Prevent updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Force updates to decoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class AE(nn.Module):
    """Main autoencoder class.

    This class can construct both linear and convolutional autoencoders. The linear autoencoder
    utilizes a single hidden layer, dense feedforward layers (i.e. not convolutional), and the
    encoding and decoding weights are tied to more closely resemble PCA/SVD. The convolutional
    autoencoder architecture is defined by various keys in the dict that serves as the constructor
    input. See the :mod:`behavenet.fitting.ae_model_architecture` module to see examples for how
    this is done.
    """

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - 'model_type' (:obj:`int`): 'conv' | 'linear'
            - 'model_class' (:obj:`str`): 'ae' | 'vae'
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
        super(AE, self).__init__()
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


class ConditionalAE(AE):
    """Conditional autoencoder class.

    This class constructs conditional convolutional autoencoders. At the latent layer an additional
    set of variables, saved under the 'labels' key in the hdf5 data file, are concatenated with the
    latents before being reshaped into a 2D array for decoding. Note that standard implementations
    of conditional convolutional autoencoders also include the labels as input to the encoder,
    which does not occur here.
    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain :obj:`n_labels`

        """
        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        super(ConditionalAE, self).__init__(hparams)

    def build_model(self):
        """Construct the model using hparams.

        The ConditionalAE is initialized when :obj:`model_class='cond-ae`, and currently only
        supports :obj:`model_type='conv` (i.e. no linear)
        """
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents'] + self.hparams['n_labels']
        self.encoding = ConvAEEncoder(self.hparams)
        self.decoding = ConvAEDecoder(self.hparams)

    def forward(self, x, dataset=None, labels=None, labels_2d=None):
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
        if labels_2d is not None:
            # append label information to input
            x = torch.cat((x, labels_2d), dim=1)
        x, pool_idx, outsize = self.encoding(x, dataset=dataset)
        z = torch.cat((x, labels), dim=1)
        y = self.decoding(z, pool_idx, outsize, dataset=dataset)
        return y, x


class CustomDataParallel(nn.DataParallel):
    """Wrapper class for multi-gpu training.

    from https://github.com/pytorch/tutorials/issues/836
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
