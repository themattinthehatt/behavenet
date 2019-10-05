import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import behavenet.core as core
# from pyslds.models import HMMSLDS
# from pyslds.states import HMMSLDSStatesEigen
# from pybasicbayes.distributions import Gaussian, Regression
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


class ARHMM(nn.Module):

    def __init__(self, hparams):
        super(ARHMM, self).__init__()
        self.hparams = hparams

        assert self.hparams['dynamics'] in ("gaussian", "diagonal_gaussian", "studentst")
        self.dynamics = self.hparams['dynamics'].lower()

        self.build_model()

    def build_model(self):
        hp = self.hparams
        dynamics = self.dynamics

        # Dynamics parameters
        self.As = nn.Parameter(torch.zeros((hp['n_discrete_states'], hp['latent_dim_size_h']*hp['nlags'], hp['latent_dim_size_h'])))
        self.bs = nn.Parameter(torch.zeros((hp['n_discrete_states'], hp['latent_dim_size_h'])))

        if dynamics.lower() == "gaussian":
            self.sqrt_Qs = nn.Parameter(
                 torch.eye(hp['latent_dim_size_h']).unsqueeze(0).repeat((hp['n_discrete_states'], 1, 1)))
        elif dynamics.lower() == "diagonal_gaussian":
            self.inv_softplus_Qs = nn.Parameter(torch.ones((hp['n_discrete_states'], hp['latent_dim_size_h'])))
        elif dynamics.lower() == "studentst":
            self.inv_softplus_nus = nn.Parameter(torch.ones((hp['n_discrete_states'], hp['latent_dim_size_h'])))
        else:
            raise Exception("Bad dynamics model: {}".format(dynamics))

        # Transition parameters
        self.stat_log_transition_proba = \
                nn.Parameter(torch.log(
                hp['transition_init'] * torch.eye(hp['n_discrete_states']) + (1-hp['transition_init']) / hp['n_discrete_states'] * torch.ones((hp['n_discrete_states'], hp['n_discrete_states']))))

    def initialize(self,method="lr", *args, **kwargs):
        init_methods = dict(lr=self._initialize_with_lr)
        if method not in init_methods:
            raise Exception("Invalid initialization method: {}".format(method))
        return init_methods[method](*args, **kwargs)

    def _initialize_with_lr(self, data_gen, L2_reg=0.01):
        self.As.data, self.bs.data, self.inv_softplus_Qs.data = core.initialize_with_lr(self, self.hparams, data_gen, L2_reg=L2_reg)

    def log_pi0(self, *args):
        return core.uniform_initial_distn(self).to(self.hparams['device'])

    def log_prior(self,*args):
        return core.dirichlet_prior(self)

    def log_transition_proba(self, data, *args):
        batch_size = data.shape[0]
        return core.stationary_log_transition_proba(self, batch_size)

    def log_dynamics_proba(self, data, *args):
        if self.dynamics == "gaussian":
            return core.gaussian_ar_log_proba(self,data)
        elif self.dynamics == "diagonal_gaussian":
            return core.diagonal_gaussian_ar_log_proba(self,data)
        elif self.dynamics == "studentst":
            return core.studentst_ar_log_proba(self,data)
        else:
            raise Exception("Invalid dynamics: {}".format(self.dynamics))

    def get_low_d(self,signal):
        return signal


class InputDrivenARHMM(ARHMM):

    def __init__(self, hparams):
        super(InputDrivenARHMM, self).__init__(hparams)

    def build_model(self):
        super(InputDrivenARHMM,self).build_model()
        if self.hparams['decoding_model_class']=='time_lagged_linear':
            self.transition_matrix_bias = TimeLaggedLinear(self.hparams,self.hparams['n_discrete_states'])
            self.emission_bias = TimeLaggedLinear(self.hparams,self.hparams['latent_dim_size_h'])

    def log_transition_proba(self, data, inputs):
        return core.input_driven_log_transition_proba(self, inputs)

    def log_dynamics_proba(self, data, inputs, *args):
        if self.dynamics == "gaussian":
            return core.gaussian_ar_log_proba(self,data, inputs)
        else:
            raise Exception("Invalid input driven dynamics: {}".format(self.dynamics))


class TimeLaggedLinear(nn.Module):

    def __init__(self, hparams, output_size):
        super(TimeLaggedLinear, self).__init__()
        self.hparams = hparams
        self.output_size = output_size

        self.build_model()

    def build_model(self):

        self.linear = nn.Conv1d(self.hparams['n_neurons'],self.output_size,self.hparams['neural_lags'],padding=int((self.hparams['neural_lags']-1)/2))

    def forward(self, x):
        # x should be timesteps x neurons

        # have to reconfigure to 1 x neurons x timesteps
        x = x.unsqueeze(0).transpose(1,2)

        x = self.linear(x)

        x = x.transpose(2,1).squeeze(0)
        return x


class Decoder(nn.Module):
    """General wrapper class for decoding models"""

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        if hparams['model_type'] == 'ff' \
                or hparams['model_type'] == 'ff-mv' \
                or hparams['model_type'] == 'linear' \
                or hparams['model_type'] == 'linear-mv':
            self.model = NN(hparams)
        elif hparams['model_type'] == 'lstm':
            self.model = LSTM(hparams)
        else:
            raise ValueError(
                '"%s" is not a valid model type' % hparams['model_type'])

    def forward(self, x):
        return self.model(x)


class NN(nn.Module):

    def __init__(self, hparams):

        super().__init__()
        self.hparams = hparams
        self.build_model()

    def build_model(self):

        self.decoder = nn.ModuleList()

        global_layer_num = 0

        in_size = self.hparams['input_size']

        # first layer is 1d conv for incorporating past/future neural activity
        if self.hparams['n_hid_layers'] == 0:
            out_size = self.hparams['output_size']
        elif self.hparams['n_hid_layers'] == 1:
            out_size = self.hparams['n_final_units']
        else:
            out_size = self.hparams['n_int_units']

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
                raise ValueError(
                    '"%s" is an invalid noise dist' % self.hparams['noise_dist'])
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
                    '"%s" is an invalid activation function' %
                    self.hparams['activation'])

        if activation:
            name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
            self.decoder.add_module(name, activation)

        # add layer for data-dependent precision matrix if required
        if self.hparams['n_hid_layers'] == 0 \
                and self.hparams['noise_dist'] == 'gaussian-full':
            # build sqrt of precision matrix
            self.precision_sqrt = nn.Linear(
                in_features=in_size,
                out_features=out_size ** 2)
        else:
            self.precision_sqrt = None

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # loop over hidden layers (0 layers <-> linear regression)
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                out_size = self.hparams['output_size']
            elif i_layer == self.hparams['n_hid_layers'] - 2:
                out_size = self.hparams['n_final_units']
            else:
                out_size = self.hparams['n_int_units']

            # add layer
            layer = nn.Linear(
                in_features=in_size,
                out_features=out_size)
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
                    raise ValueError(
                        '"%s" is an invalid noise dist' % self.hparams['noise_dist'])
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
                        '"%s" is an invalid activation function' %
                        self.hparams['activation'])

            if activation:
                self.decoder.add_module(
                    '%s_%02i' % (self.hparams['activation'], global_layer_num),
                    activation)

            # add layer for data-dependent precision matrix if required
            if i_layer == self.hparams['n_hid_layers'] - 1 \
                    and self.hparams['noise_dist'] == 'gaussian-full':
                # build sqrt of precision matrix
                self.precision_sqrt = nn.Linear(
                    in_features=in_size,
                    out_features=out_size ** 2)
            else:
                self.precision_sqrt = None

            # update layer info
            global_layer_num += 1
            in_size = out_size

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): time x neurons

        Returns:

        """
        # print('Model input size is {}'.format(x.shape))
        # print()
        y = None
        for name, layer in self.decoder.named_children():

            # get data-dependent precision matrix if required
            if name == self.final_layer \
                    and self.hparams['noise_dist'] == 'gaussian-full':
                y = self.precision_sqrt(x)
                y = y.reshape(
                    -1, self.hparams['output_size'],
                    self.hparams['output_size'])
                y = torch.bmm(y, y.transpose(1, 2))

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer(x.transpose(1, 0).unsqueeze(0)).squeeze().transpose(1, 0)
            else:
                x = layer(x)

            # print('Layer {}'.format(name))
            # print('\toutput size: {}'.format(x.shape))
            # for param in layer.parameters():
            #     print('\tparam shape is {}'.format(param.size()))
            # print()

        return x, y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LSTM(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        raise NotImplementedError


# class NeuralNetDecoderLaggedSLDSStates(HMMSLDSStatesEigen):
#     """
#     Override the SLDS potentials with those from a neural net.
#
#     The "states" are really a concatenation of states over n lags.
#
#     x'_t = [x'_t, x'_{t-1}, ..., x'_{t-nlags+1}]   in R^{D*n_lags}
#
#     The dynamics are given by
#
#     x'_{t+1} = A' x'_t
#
#     where
#
#     A' = [A_1,  A_2,  ...,  A_{n_lags-1},  A_{n_lags}]
#          [ I ,   0 ,  ...,   0          ,   0        ]
#          [ 0 ,   I ,  ...,   0          ,   0        ]
#          ...
#          [ 0 ,   0 ,  ...,   I          ,   0        ]
#
#     is a block dynamics matrix in R^{D*n_lags x D*n_lags}
#
#     That means the emission potentials also need to be
#     the same shape.  They only apply to the first D entries in x' though.
#     """
#     def __init__(self, model, x_preds=None, x_covs=None, z_logprobs=None,
#                  x_scale=1.0, z_scale=1.0, **kwargs):
#         """
#         :param x_preds:     TxD array of decoded continuous states from neural net.
#         :param x_covs:      DxD or TxDxD array of decoded covariance of continuous states from neural net.
#         :param z_logprobs:  TxK array of log probabilities of discrete states from neural net.
#         :param x_scale:     positive scalar value indicating how much to weight the continuous predictions.
#         :param z_scale:     positive scalar value indicating how much to weight the discrete predictions.
#
#         Other kwargs include data,
#         """
#         super(NeuralNetDecoderLaggedSLDSStates, self).__init__(model, **kwargs)
#
#         # Precompute the info form potentials from the decoder
#         assert np.isscalar(x_scale)
#         T, D = x_preds.shape
#         assert T == self.T
#         assert D == self.D_latent / self.n_lags
#         assert x_covs.shape == (T, D, D) or x_covs.shape == (D, D)
#
#         # Convert predictions into information form potentials
#         self.J_node = np.zeros((T, self.D_latent, self.D_latent))
#         self.J_node[:, :D, :D] = x_scale * np.linalg.inv(x_covs)
#         self.h_node = np.zeros((T, self.D_latent))
#         self.h_node[:, :D] = np.matmul(self.J_node[:, :D, :D], x_preds[:, :, None])[:, :, 0]
#         self.log_Z_node = np.zeros(self.T)
#
#         # Precompute the discrete state potentials from the decoder
#         assert np.isscalar(z_scale)
#         T, K = z_logprobs.shape
#         assert T == self.T
#         assert K == self.num_states
#         self.decoder_aBl = z_scale * z_logprobs
#
#     @property
#     def n_lags(self):
#         return self.model.n_lags
#
#     # Override the emissions parameters with the J and h given by the decoder
#     # J = Sigma^{-1} and h = Sigma^{-1} mu.  These are the natural parameters
#     # of the multivariate normal distribution
#     @property
#     def info_emission_params(self):
#         return self.J_node, self.h_node, self.log_Z_node
#
#     # Override the discrete potentials too
#     @property
#     def aBl(self):
#         if self._aBl is None:
#             self._aBl = np.zeros((self.T, self.num_states))
#             ids, dds = self.init_dynamics_distns, self.dynamics_distns
#
#             for idx, (d1, d2) in enumerate(zip(ids, dds)):
#                 # Initial state distribution
#                 self._aBl[0, idx] = d1.log_likelihood(self.gaussian_states[0])
#
#                 # Dynamics
#                 xs = np.hstack((self.gaussian_states[:-1], self.inputs[:-1]))
#                 self._aBl[:-1, idx] = d2.log_likelihood((xs, self.gaussian_states[1:]))
#
#             # Add the decoder potential
#             self._aBl += self.decoder_aBl
#
#             # Handle NaN's
#             self._aBl[np.isnan(self._aBl).any(1)] = 0.
#
#         return self._aBl
#
#
# class NeuralNetDecoderLaggedSLDS(HMMSLDS):
#     _states_class = NeuralNetDecoderLaggedSLDSStates
#
#     def __init__(self, arhmm):
#         """
#         Build the decoder from an SSM autoregressive hidden Marko model
#         """
#         K = arhmm.K  # number of discrete states
#         D_latent = D = arhmm.D   # dimensionality of continuous latents
#         D_input = 1          # input dimensionality is 1 for the affine term
#         D_obs = 1            # dummy value
#
#         # NOTE: The latent states will be D_latent * L dimensional
#         self.n_lags = L = arhmm.observations.lags       # number of lags used by the ARHMM
#
#         # Initialize the initial state distribution of the continuous states
#         init_dynamics_distns = \
#             [Gaussian(nu_0=D_latent * L +3,
#                       sigma_0=3.*np.eye(D_latent * L),
#                       mu_0=np.zeros(D_latent * L),
#                       kappa_0=0.01)
#              for _ in range(K)]
#
#         for id, mu, sigma in zip(init_dynamics_distns,
#                                  arhmm.observations.mu_init,
#                                  arhmm.observations.Sigmas_init):
#             id.sigma *= 0
#             for l in range(L):
#                 id.mu[l*D:(l+1)*D] = mu
#                 id.sigma[l*D:(l+1)*D, l*D:(l+1)*D] = sigma
#
#         # Initialize dynamics distributions
#         dynamics_distns = [Regression(
#             nu_0=D_latent * L + 1,
#             S_0=D_latent * np.eye(D_latent * L),
#             M_0=np.hstack((.99 * np.eye(D_latent * L), np.zeros((D_latent * L, D_input)))),
#             K_0=D_latent * np.eye(D_latent * L + D_input))
#             for _ in range(K)]
#
#         # Combine (A, b) into a single matrix
#         Abs = [np.column_stack((A, b)) for A, b in zip(arhmm.observations.As, arhmm.observations.bs)]
#         for dd, Ab, sigma in zip(dynamics_distns, Abs, arhmm.observations.Sigmas):
#             assert Ab.shape == (D_latent, D_latent * L + 1)
#
#             # Convert Ab into a "full" dynamics matrix that propagates past states
#             A_full = np.zeros((D_latent * L, D_latent * L + 1))
#             sigma_full = np.zeros((D_latent * L, D_latent * L))
#
#             # First row are the AR dynamics matrices and bias
#             A_full[:D_latent, :] = Ab
#             sigma_full[:D_latent, :D_latent] = sigma
#
#             # Following rows propagate lagged states
#             for l in range(1, L):
#                 A_full[l*D_latent:(l+1)*D_latent, (l-1)*D_latent:l*D_latent] = np.eye(D_latent)
#                 sigma_full[l*D_latent:(l+1)*D_latent, l*D_latent:(l+1)*D_latent] = 1e-8 * np.eye(D_latent)
#
#             dd.A = A_full
#             dd.sigma = sigma_full
#
#         # Initialize the transitions
#         trans_matrix = np.exp(arhmm.transitions.log_Ps)
#         assert np.allclose(trans_matrix.sum(1), 1)
#
#         # Initialize the initial state distribution
#         pi_0 = np.exp(arhmm.init_state_distn.log_pi0)
#
#         # Call the super constructor with these distributions
#         super(NeuralNetDecoderLaggedSLDS, self).__init__(
#             init_dynamics_distns=init_dynamics_distns,
#             dynamics_distns=dynamics_distns,
#             emission_distns=None,
#             pi_0=pi_0, init_state_concentration=1,
#             trans_matrix=trans_matrix, alpha=1)
#
#
#     def add_data(self, x_preds, x_covs, z_logprobs, x_scale=1, z_scale=1):
#         # Make dummy data for these predictions
#         T = x_preds.shape[0]
#         dummy_data = np.zeros((T, 1))
#         dummy_inputs = np.ones((T, 1))
#
#         # Construct a states object for these predictions
#         self.states_list.append(
#             NeuralNetDecoderLaggedSLDSStates(
#                 model=self, data=dummy_data, inputs=dummy_inputs,
#                 x_preds=x_preds, x_covs=x_covs, z_logprobs=z_logprobs,
#                 x_scale=x_scale, z_scale=z_scale,
#                 stateseq=None, fixed_stateseq=False))
#
#         return self.states_list[-1]
# class LinearVAEEncoder(nn.Module):

#     def __init__(self, latent_dim_size_h, pixel_size):

#         super(LinearVAEEncoder, self).__init__()

#         self.latent_dim_size_h = latent_dim_size_h
#         self.pixel_size=pixel_size
#         self.__build_model()

#     def __build_model(self):

#         self.prior_mu = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
#         self.prior_logvar = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
#       # self.h_var = nn.Parameter(1e-1*torch.ones(100,10),requires_grad=True)
#         self.softplus = nn.Softplus()
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.prior_mu(x), self.softplus(self.prior_logvar(x))

#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False


# class LinearVAEDecoder(nn.Module):

#     def __init__(self, latent_dim_size_h, pixel_size, y_var_value, y_var_parameter, encoding):

#         super(LinearVAEDecoder, self).__init__()
#         self.latent_dim_size_h = latent_dim_size_h
#         self.y_var_value = y_var_value
#         self.encoding = encoding
#         self.pixel_size = pixel_size
#         self.y_var_parameter = y_var_parameter
#         self.__build_model()

#     def __build_model(self):

#         self.bias = nn.Parameter(torch.zeros(self.pixel_size*self.pixel_size),requires_grad=True)
#         if self.y_var_parameter:
#             inv_softplus_var = np.log(np.exp(self.y_var_value)-1)
#             self.y_var = nn.Parameter(inv_softplus_var*torch.ones(self.pixel_size,self.pixel_size),requires_grad=True)
#         else:
#             self.y_var = nn.Parameter(self.y_var_value*torch.ones(1),requires_grad=False)

#     def forward(self, x):

#         y_mu =  F.linear(x, self.encoding.prior_mu.weight.t()) + self.bias
#         y_mu = y_mu.view(y_mu.size(0), 1, self.pixel_size,self.pixel_size)

#         if self.y_var_parameter:
#             y_var = F.softplus(self.y_var).unsqueeze(0).unsqueeze(0).expand(y_mu.shape[0],-1,-1,-1)
#         else:
#             y_var = self.y_var
#         return y_mu, y_var

# class ConvVAEEncoder(nn.Module):

#     def __init__(self, latent_dim_size_h, bn):

#         super(ConvVAEEncoder, self).__init__()

#         self.latent_dim_size_h = latent_dim_size_h
#         self.bn = bn
#         self.__build_model()

#     def __build_model(self):
#         # TO DO: make flexible

#         if self.bn:
#             self.encoder = nn.Sequential(
#               nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
#                         stride=2, padding=1, bias=False),
#               nn.BatchNorm2d(32),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(64),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(256),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(512),
#               nn.LeakyReLU(0.05, inplace=True)
#             )
#         else:
#             self.encoder = nn.Sequential(
#               nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
#                         stride=2, padding=1, bias=False),
#              # nn.BatchNorm2d(32),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#              # nn.BatchNorm2d(64),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#              # nn.BatchNorm2d(256),
#               nn.LeakyReLU(0.05, inplace=True),
#               nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
#                         stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(512),
#               nn.LeakyReLU(0.05, inplace=True)
#             )

#         self.out_img = (512, 5, 5)
#         self.prior_mu = nn.Linear(512*5*5, self.latent_dim_size_h)
#         #self.h_var = nn.Parameter(1e-6*torch.ones(100,10),requires_grad=False)
#         self.prior_logvar = nn.Linear(512*5*5, self.latent_dim_size_h)
#         self.softplus = nn.Softplus()
#     def forward(self, x):
#         if x.dim() == 3:
#           x = x.view(x.size(0), 1, x.size(1), x.size(2))
#         h = self.encoder(x)
#         h = h.view(h.size(0), -1)
#         return self.prior_mu(h), self.softplus(self.prior_logvar(h))

#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False


# class ConvVAEDecoder(nn.Module):

#     def __init__(self, latent_dim_size_h, pixel_size, y_var_value, y_var_parameter, bn):

#         super(ConvVAEDecoder, self).__init__()
#         self.latent_dim_size_h = latent_dim_size_h
#         self.y_var_value = y_var_value
#         self.y_var_parameter = y_var_parameter
#         self.bn = bn
#         self.pixel_size = pixel_size
#         self.__build_model()

#     def __build_model(self):

#          # TO DO: make flexible
#         self.out_img = (512, 5, 5)

#         self.linear_decode = nn.Linear(self.latent_dim_size_h, 512*5*5)
#         if self.bn:
#             self.decoder = nn.Sequential(
#               nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(256),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(128),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(64),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(32),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#               nn.BatchNorm2d(16),
#               nn.ReLU(inplace=True),
#               nn.MaxPool2d(kernel_size=2, stride=2),
#               nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,
#                         padding=1),
#               nn.Sigmoid()
#             )
#         else:
#             self.decoder = nn.Sequential(
#               nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(256),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(128),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(64),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(32),
#               nn.ReLU(inplace=True),
#               nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,
#                                  stride=2, padding=1, bias=True),
#             #  nn.BatchNorm2d(16),
#               nn.ReLU(inplace=True),
#               nn.MaxPool2d(kernel_size=2, stride=2),
#               nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,
#                         padding=1),
#               nn.Sigmoid()
#             )

#         if self.y_var_parameter:
#             inv_softplus_var = np.log(np.exp(self.y_var_value)-1)
#             self.y_var = nn.Parameter(inv_softplus_var*torch.ones(self.pixel_size,self.pixel_size),requires_grad=True)
#         else:
#             self.y_var = nn.Parameter(self.y_var_value*torch.ones(1),requires_grad=False)

#     def forward(self, x):

#         y = self.linear_decode(x)
#         y = y.view(y.size(0), *self.out_img)

#         y_mu = self.decoder(y)
#         if self.y_var_parameter:
#             y_var = F.softplus(self.y_var).unsqueeze(0).unsqueeze(0).expand(y_mu.shape[0],-1,-1,-1)
#         else:
#             y_var = self.y_var

#         return y_mu, y_var

#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False


# class LinearVAEEncoder(nn.Module):

#     def __init__(self, latent_dim_size_h, pixel_size):

#         super(LinearVAEEncoder, self).__init__()

#         self.latent_dim_size_h = latent_dim_size_h
#         self.pixel_size=pixel_size
#         self.__build_model()

#     def __build_model(self):

#         self.prior_mu = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
#         self.prior_logvar = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
#       # self.h_var = nn.Parameter(1e-1*torch.ones(100,10),requires_grad=True)
#         self.softplus = nn.Softplus()
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.prior_mu(x), self.softplus(self.prior_logvar(x))

#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False


# class LinearVAEDecoder(nn.Module):

#     def __init__(self, latent_dim_size_h, pixel_size, y_var_value, y_var_parameter, encoding):

#         super(LinearVAEDecoder, self).__init__()
#         self.latent_dim_size_h = latent_dim_size_h
#         self.y_var_value = y_var_value
#         self.encoding = encoding
#         self.pixel_size = pixel_size
#         self.y_var_parameter = y_var_parameter
#         self.__build_model()

#     def __build_model(self):

#         self.bias = nn.Parameter(torch.zeros(self.pixel_size*self.pixel_size),requires_grad=True)
#         if self.y_var_parameter:
#             inv_softplus_var = np.log(np.exp(self.y_var_value)-1)
#             self.y_var = nn.Parameter(inv_softplus_var*torch.ones(self.pixel_size,self.pixel_size),requires_grad=True)
#         else:
#             self.y_var = nn.Parameter(self.y_var_value*torch.ones(1),requires_grad=False)

#     def forward(self, x):

#         y_mu =  F.linear(x, self.encoding.prior_mu.weight.t()) + self.bias
#         y_mu = y_mu.view(y_mu.size(0), 1, self.pixel_size,self.pixel_size)

#         if self.y_var_parameter:
#             y_var = F.softplus(self.y_var).unsqueeze(0).unsqueeze(0).expand(y_mu.shape[0],-1,-1,-1)
#         else:
#             y_var = self.y_var
#         return y_mu, y_var


# class VAE(nn.Module):

#     def __init__(self, hparams):

#         super(VAE, self).__init__()
#         self.hparams = hparams

#         self.__build_model()

#     def __build_model(self):

#         if self.hparams.vae_type=='conv':
#             self.encoding = ConvVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.bn)
#             self.decoding = ConvVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter, self.hparams.bn)
#         elif self.hparams.vae_type=='linear':
#             self.encoding = LinearVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size)
#             self.decoding = LinearVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter, self.encoding)

#     def reparameterize(self, mu, var, random_draw):
#        if random_draw:
#           std = torch.pow(var,0.5)
#           eps = torch.randn_like(std)
#           return eps.mul(std).add_(mu)
#        else:
#           return mu

#     def forward(self, x, random_draw=1):

#         h_mu, h_var = self.encoding(x)
#         x  = self.reparameterize(h_mu,h_var,random_draw)
#         y_mu, y_var = self.decoding(x)

#         return y_mu, y_var, h_mu, h_var


# class SLDS(nn.Module):

#     """
#     This will look a lot like an ARHMM but it has a decoder for mapping
#     continuous latent states to observations.
#     """

#     def __init__(self, hparams, dynamics="gaussian", emissions="gaussian"):
#         super(SLDS, self).__init__()
#         self.hparams = hparams

#         assert dynamics.lower() in ("gaussian", "studentst")
#         self.dynamics = dynamics.lower()

#         assert emissions.lower() in ("gaussian",)
#         self.emissions = emissions.lower()

#         self.__build_model()

#     def __build_model(self):
#         hp = self.hparams
#         dynamics = self.dynamics

#         # Dynamics parameters
#         self.As = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h*hp.nlags, hp.latent_dim_size_h)))
#         self.bs = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h)))
#         self.inv_softplus_Qs = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))

#         if dynamics.lower() == "studentst":
#             self.inv_softplus_nus = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))

#         # Transition parameters
#         self.stat_log_transition_proba = \
#                 nn.Parameter(torch.log(
#                 hp.transition_init * torch.eye(hp.n_discrete_states) +
#                 (1-hp.transition_init) / hp.n_discrete_states * torch.ones((hp.n_discrete_states, hp.n_discrete_states))))

#         if self.hparams.low_d_type == 'vae':
#             hp = pd.read_csv(self.hparams.init_vae_model_path+'meta_tags.csv')
#             hp = dict(zip(hp['key'], hp['value']))
#             vae_hparams = objectview(hp)

#             vae_model = VAE(vae_hparams)
#             vae_model2 = VAE(vae_hparams)

#             vae_model.load_state_dict(torch.load(self.hparams.init_vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
#             VAE_decoder_model = vae_model.decoding
#             VAE_decoder_model.to(self.hparams.device)
#             self.VAE_decoder_model = VAE_decoder_model
#             #self.VAE_decoder_model.encoding.prior_mu.bias=None
#             #self.VAE_decoder_model.encoding.prior_logvar.weight=None
#             #self.VAE_decoder_model.encoding.prior_logvar.bias=None

#             vae_model2.load_state_dict(torch.load(self.hparams.init_vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
#             VAE_encoder_model = vae_model2.encoding
#             VAE_encoder_model.freeze()
#             #VAE_encoder_model.training=False
#             VAE_encoder_model.to(self.hparams.device)
#             self.VAE_encoder_model = VAE_encoder_model

#     def decode(self, states):
#         """
#         Pass the continuous latent state through the decoder network
#         get the mean of the observations.

#         @param states: a T (time) x H (latent dim)
#         """
#         y_mu, y_var = self.VAE_decoder_model(states)
#         return y_mu, y_var

#     # The remainder of the methods look like those of the ARHMM,
#     # but now we also have an emission probability of the data given
#     # the continuous latent states.
#     def initialize(self,method="lr", *args, **kwargs):
#         init_methods = dict(lr=self._initialize_with_lr)
#         if method not in init_methods:
#             raise Exception("Invalid initialization method: {}".format(method))
#         return init_methods[method](*args, **kwargs)

#     def _initialize_with_lr(self, data_gen, L2_reg=0.01):
#         self.As.data, self.bs.data, self.inv_softplus_Qs.data = core.initialize_with_lr(self, self.hparams,data_gen, L2_reg=L2_reg)

#     def get_low_d(self,signal):
#         if self.hparams.low_d_type == 'vae':
#             signal,_= self.VAE_encoder_model(signal)
#         elif self.hparams.low_d_type == 'pca':
#             pass
#         else:
#             raise NotImplementedError
#         return signal

#     def log_pi0(self, *args):
#         return core.uniform_initial_distn(self).to(self.hparams.device)

#     def log_prior(self,*args):
#         return core.dirichlet_prior(self)

#     def log_transition_proba(self, *args):
#         return core.stationary_log_transition_proba(self)

#     def log_dynamics_proba(self, data, *args):
#         if self.dynamics == "gaussian":
#             return core.gaussian_ar_log_proba(self, data)
#         elif self.dynamics == "studentst":
#             return core.studentst_ar_log_proba(self, data)
#         else:
#             raise Exception("Invalid dynamics: {}".format(self.dynamics))

#     def log_emission_proba(self, data, states):
#         """
#         Compute the likelihood of the data given the continuous states.
#         """
#         if self.emissions == "gaussian":
#             return core.gaussian_emissions_diagonal_variance(self, data, states)
#         else:
#             raise Exception("Invalid emissions: {}".format(self.emissions))
