import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# TODO: __repr__ methods on models for printing


class ConvAEEncoder(nn.Module):

    def __init__(self, hparams):

        super(ConvAEEncoder, self).__init__()

        self.hparams = hparams
        self.encoder = None
        self.build_model()

    def build_model(self):

        self.encoder = nn.ModuleList()

        # Loop over layers (each conv/batch norm/max pool/relu chunk counts as
        # one layer for global_layer_num)
        global_layer_num = 0
        for i_layer in range(0, len(self.hparams['ae_encoding_n_channels'])):

            # only add if conv layer (checks within this for max pool layer)
            if self.hparams['ae_encoding_layer_type'][i_layer] == 'conv':

                # convolution layer
                args = self.get_conv2d_args(i_layer, global_layer_num)
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
                    args = self.get_maxpool2d_args(i_layer)
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

        # final FF layer to latents
        last_conv_size = self.hparams['ae_encoding_n_channels'][-1] \
                         * self.hparams['ae_encoding_y_dim'][-1] \
                         * self.hparams['ae_encoding_x_dim'][-1]
        self.FF = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])

        # If VAE model, have additional FF layer to latent variances
        if self.hparams['model_class'] == 'vae':
            raise NotImplementedError
            # self.logvar = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])
            # self.softplus = nn.Softplus()
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')

    def get_conv2d_args(self, layer, global_layer):

        if layer == 0:
            in_channels = self.hparams['ae_input_dim'][0]
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

    def get_maxpool2d_args(self, layer):
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
        # x should be batch size x n channels x xdim x ydim

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

        # Reshape for FF layer
        x = x.view(x.size(0), -1)

        if self.hparams['model_class'] == 'ae':
            return self.FF(x), pool_idx, target_output_size
        elif self.hparams['model_class'] == 'vae':
            return NotImplementedError
        else:
            raise ValueError(
                '"%s" is not a valid model class' % self.hparams['model_class'])

    def freeze(self):
        # easily freeze the AE encoder parameters
        for param in self.parameters():
            param.requires_grad = False


class ConvAEDecoder(nn.Module):

    def __init__(self, hparams):

        super(ConvAEDecoder, self).__init__()

        self.hparams = hparams
        self.decoder = None
        self.build_model()

    def build_model(self):

        # First FF layer (from latents to size of last encoding layer)
        first_conv_size = self.hparams['ae_decoding_starting_dim'][0] \
                          * self.hparams['ae_decoding_starting_dim'][1] \
                          *self.hparams['ae_decoding_starting_dim'][2]
        self.FF = nn.Linear(self.hparams['n_ae_latents'], first_conv_size)

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
                args = self.get_convtranspose2d_args(i_layer, global_layer_num)
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

        # optional final FF layer (rarely used)
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
                str('last_FF%i' % global_layer_num), module)
            self.decoder.add_module(
                str('sigmoid%i' % global_layer_num), nn.Sigmoid())

        if self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')

    def get_convtranspose2d_args(self, layer, global_layer):

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

        # First FF layer/resize to be convolutional input
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
                    x = F.pad(x, [-i for i in self.conv_t_pads[name]])
            elif isinstance(layer, nn.ModuleList):
                x = layer[dataset](x)
                if self.conv_t_pads[name] is not None:
                    # asymmetric padding for convtranspose layer if necessary
                    # (-i does cropping!)
                    x = F.pad(x, [-i for i in self.conv_t_pads[name]])
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

        if self.hparams['model_class'] == 'ae':
            return x
        elif self.hparams['model_class'] == 'vae':
            raise ValueError('Not Implemented Error')
        else:
            raise ValueError('Not Implemented Error')

    def freeze(self):
        # easily freeze the AE decoder parameters
        for param in self.parameters():
            param.requires_grad = False


class LinearAEEncoder(nn.Module):

    def __init__(self, n_latents, input_size):
        """

        Args:
            n_latents (int):
            input_size (list or tuple): n_channels x y_pix x x_pix
        """
        super().__init__()

        self.n_latents = n_latents
        self.input_size = input_size
        self.build_model()

    def build_model(self):
        self.encoder = nn.Linear(
            out_features=self.n_latents,
            in_features=np.prod(self.input_size),
            bias=True)

    def forward(self, x, dataset=None):
        # reshape
        x = x.view(x.size(0), -1)
        return self.encoder(x), None, None

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LinearAEDecoder(nn.Module):

    def __init__(self, n_latents, output_size, encoder=None):
        """

        Args:
            n_latents (int):
            output_size (list or tuple): n_channels x y_pix x x_pix
            encoder (nn.Module object): for linking encoder/decoder weights
        """
        super().__init__()
        self.n_latents = n_latents
        self.output_size = output_size
        self.encoder = encoder
        self.build_model()

    def build_model(self):

        if self.encoder is None:
            self.decoder = nn.Linear(
                out_features=np.prod(self.output_size),
                in_features=self.n_latents,
                bias=True)
        else:
            self.bias = nn.Parameter(
                torch.zeros(int(np.prod(self.output_size))), requires_grad=True)

    def forward(self, x, dataset=None):
        # push through
        if self.encoder is None:
            x = self.decoder(x)
        else:
            x = F.linear(x, self.encoder.encoder.weight.t()) + self.bias
        # reshape
        x = x.view(x.size(0), *self.output_size)

        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class AE(nn.Module):

    def __init__(self, hparams):

        super(AE, self).__init__()
        self.hparams = hparams
        self.model_type = self.hparams['model_type']
        self.img_size = (
                self.hparams['n_input_channels'],
                self.hparams['y_pixels'],
                self.hparams['x_pixels'])
        self.build_model()

    def build_model(self):

        if self.model_type == 'conv':
            self.encoding = ConvAEEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            if self.hparams.get('fit_sess_io_layers', False):
                raise NotImplementedError
            n_latents = self.hparams['n_ae_latents']
            self.encoding = LinearAEEncoder(n_latents, self.img_size)
            self.decoding = LinearAEDecoder(
                n_latents, self.img_size, self.encoding)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

    def forward(self, x, dataset=None):

        if self.model_type == 'conv':
            x, pool_idx, outsize = self.encoding(x, dataset=dataset)
            y = self.decoding(x, pool_idx, outsize, dataset=dataset)
        elif self.model_type == 'linear':
            x, _, _ = self.encoding(x)
            y = self.decoding(x)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

        return y, x
