import torch
from torch import nn, optim
import pandas as pd
import torch.nn.functional as F
import numpy as np
from ast import literal_eval
import behavenet.core as core


class ConvAEEncoder(nn.Module):
    
    def __init__(self, hparams):

        super(ConvAEEncoder, self).__init__()
      
        self.hparams = hparams
        self.build_model()

    def build_model(self):
        
        self.encoder = nn.ModuleList()

        # Loop over layers (each conv/batch norm/max pool/relu chunk counts as one layer for global_layer_num)
        global_layer_num=0
        for i_layer in range(0,len(self.hparams['ae_encoding_n_channels'])):

            if self.hparams['ae_encoding_layer_type'][i_layer]=='conv': # only add if conv layer (checks within this for next max pool layer)

                ## Convolution layer 
                in_channels = self.hparams['ae_input_dim'][0] if i_layer==0 else self.hparams['ae_encoding_n_channels'][i_layer-1]
                if self.hparams['ae_encoding_x_padding'][i_layer][0] == self.hparams['ae_encoding_x_padding'][i_layer][1] and self.hparams['ae_encoding_y_padding'][i_layer][0] == self.hparams['ae_encoding_y_padding'][i_layer][1]: # if symmetric padding
                    self.encoder.add_module('conv'+str(global_layer_num),nn.Conv2d(in_channels=in_channels,out_channels=self.hparams['ae_encoding_n_channels'][i_layer],kernel_size=self.hparams['ae_encoding_kernel_size'][i_layer],stride=self.hparams['ae_encoding_stride_size'][i_layer],padding=(self.hparams['ae_encoding_y_padding'][i_layer][0],self.hparams['ae_encoding_x_padding'][i_layer][0])))
                else:
                    self.encoder.add_module('zero_pad'+str(global_layer_num),nn.ZeroPad2d((self.hparams['ae_encoding_x_padding'][i_layer][0] ,self.hparams['ae_encoding_x_padding'][i_layer][1] ,self.hparams['ae_encoding_y_padding'][i_layer][0] ,self.hparams['ae_encoding_y_padding'][i_layer][1] )))
                    self.encoder.add_module('conv'+str(global_layer_num),nn.Conv2d(in_channels=in_channels,out_channels=self.hparams['ae_encoding_n_channels'][i_layer],kernel_size=self.hparams['ae_encoding_kernel_size'][i_layer],stride=self.hparams['ae_encoding_stride_size'][i_layer],padding=0))

                ## Batch norm layer
                if self.hparams['ae_batch_norm']:
                    self.encoder.add_module('batch norm'+str(global_layer_num),nn.BatchNorm2d(self.hparams['ae_encoding_n_channels'][i_layer],momentum=self.hparams['ae_batch_norm_momentum']))

                ## Max pool layer
                if i_layer<(len(self.hparams['ae_encoding_n_channels'])-1) and self.hparams['ae_encoding_layer_type'][i_layer+1]=='maxpool':
                    if self.hparams['ae_padding_type']=='valid':
                        self.encoder.add_module('maxpool'+str(global_layer_num),nn.MaxPool2d(kernel_size=int(self.hparams['ae_encoding_kernel_size'][i_layer+1]),stride=int(self.hparams['ae_encoding_stride_size'][i_layer+1]),padding=(self.hparams['ae_encoding_y_padding'][i_layer+1][0],self.hparams['ae_encoding_x_padding'][i_layer+1][0]),return_indices=True,ceil_mode=False)) # no ceil mode in valid mode
                    else:
                        self.encoder.add_module('maxpool'+str(global_layer_num),nn.MaxPool2d(kernel_size=int(self.hparams['ae_encoding_kernel_size'][i_layer+1]),stride=int(self.hparams['ae_encoding_stride_size'][i_layer+1]),padding=(self.hparams['ae_encoding_y_padding'][i_layer+1][0],self.hparams['ae_encoding_x_padding'][i_layer+1][0]),return_indices=True,ceil_mode=True)) # using ceil mode instead of zero padding

                ## Leaky ReLU
                self.encoder.add_module('relu'+str(global_layer_num),nn.LeakyReLU(0.05))
                global_layer_num+=1

        # Final FF layer to latents
        last_conv_size = self.hparams['ae_encoding_n_channels'][-1]*self.hparams['ae_encoding_y_dim'][-1]*self.hparams['ae_encoding_x_dim'][-1]
        self.FF = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])
    
        ## If VAE model, have additional FF layer to latent variances
        if self.hparams['model_class'] == 'vae':
            self.logvar = nn.Linear(last_conv_size, self.hparams['n_ae_latents'])
            self.softplus = nn.Softplus()
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')
            
    def forward(self, x):
        # x should be batch size x n channels x xdim x ydim

        # Loop over layers, have to collect pool_idx and output sizes if using max pooling to use in unpooling
        pool_idx=[]
        target_output_size=[]
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                target_output_size.append(x.size())
                x, idx = layer(x) 
                pool_idx.append(idx)
            else:
                x = layer(x)

        # Reshape for FF layer
        x = x.view(x.size(0), -1)
        

        if self.hparams['model_class'] == 'ae':
            return self.FF(x), pool_idx, target_output_size
        elif self.hparams['model_class'] == 'vae':
            return NotImplementedError
        else:
            raise ValueError(self.hparams['model_class'] + ' not valid model type')
            
    def freeze(self):
        # easily freeze the AE encoder parameters
        for param in self.parameters():
            param.requires_grad = False
    
    
class ConvAEDecoder(nn.Module):
    
    def __init__(self, hparams):

        super(ConvAEDecoder, self).__init__()
      
        self.hparams=hparams
        self.build_model()

    def build_model(self):

        # First FF layer (from latents to size of last encoding layer)
        first_conv_size = self.hparams['ae_decoding_starting_dim'][0]*self.hparams['ae_decoding_starting_dim'][1]*self.hparams['ae_decoding_starting_dim'][2]
        self.FF = nn.Linear(self.hparams['n_ae_latents'], first_conv_size)
        
        self.decoder = nn.ModuleList()

        # Loop over layers (each unpool/convtranspose/batch norm/relu chunk counts as one layer for global_layer_num)
        global_layer_num=0
        self.conv_t_pads = {}

        for i_layer in range(0,len(self.hparams['ae_decoding_n_channels'])):

            if self.hparams['ae_decoding_layer_type'][i_layer]=='convtranspose': # only add if conv transpose layer 
                
                ## Unpooling layer
                if i_layer>0 and self.hparams['ae_decoding_layer_type'][i_layer-1]=='unpool':
                    self.decoder.add_module('maxunpool'+str(global_layer_num),nn.MaxUnpool2d(kernel_size=(int(self.hparams['ae_decoding_kernel_size'][i_layer-1]),int(self.hparams['ae_decoding_kernel_size'][i_layer-1])),stride = (int(self.hparams['ae_decoding_stride_size'][i_layer-1]),int(self.hparams['ae_decoding_stride_size'][i_layer-1])),padding=(self.hparams['ae_decoding_y_padding'][i_layer-1][0],self.hparams['ae_decoding_x_padding'][i_layer-1][0])))

                ## ConvTranspose layer
                in_channels = self.hparams['ae_decoding_starting_dim'][0] if i_layer==0 else self.hparams['ae_decoding_n_channels'][i_layer-1]
                if self.hparams['ae_padding_type']=='valid':

                    # Calculate necessary output padding to get back original input shape
                    input_y = self.hparams['ae_decoding_y_dim'][i_layer-1] if i_layer > 0 else self.hparams['ae_decoding_starting_dim'][1]
                    y_output_padding = self.hparams['ae_decoding_y_dim'][i_layer]-((input_y-1)*self.hparams['ae_decoding_stride_size'][i_layer]+self.hparams['ae_decoding_kernel_size'][i_layer])
                    
                    input_x = self.hparams['ae_decoding_x_dim'][i_layer-1] if i_layer > 0 else self.hparams['ae_decoding_starting_dim'][2]
                    x_output_padding = self.hparams['ae_decoding_x_dim'][i_layer]-((input_x-1)*self.hparams['ae_decoding_stride_size'][i_layer]+self.hparams['ae_decoding_kernel_size'][i_layer])

                    self.decoder.add_module('convtranspose'+str(global_layer_num),nn.ConvTranspose2d(in_channels=in_channels,out_channels=self.hparams['ae_decoding_n_channels'][i_layer],kernel_size=(self.hparams['ae_decoding_kernel_size'][i_layer],self.hparams['ae_decoding_kernel_size'][i_layer]),stride=(self.hparams['ae_decoding_stride_size'][i_layer],self.hparams['ae_decoding_stride_size'][i_layer]),padding=(self.hparams['ae_decoding_y_padding'][i_layer][0],self.hparams['ae_decoding_x_padding'][i_layer][0]),output_padding=(y_output_padding,x_output_padding)))
                    self.conv_t_pads['convtranspose'+str(global_layer_num)] = None
                
                elif self.hparams['ae_padding_type']=='same':
                    if self.hparams['ae_decoding_x_padding'][i_layer][0] == self.hparams['ae_decoding_x_padding'][i_layer][1] and self.hparams['ae_decoding_y_padding'][i_layer][0] == self.hparams['ae_decoding_y_padding'][i_layer][1]:
                        self.decoder.add_module('convtranspose'+str(global_layer_num),nn.ConvTranspose2d(in_channels=in_channels,out_channels=self.hparams['ae_decoding_n_channels'][i_layer],kernel_size=(self.hparams['ae_decoding_kernel_size'][i_layer],self.hparams['ae_decoding_kernel_size'][i_layer]),stride=(self.hparams['ae_decoding_stride_size'][i_layer],self.hparams['ae_decoding_stride_size'][i_layer]),padding=(self.hparams['ae_decoding_y_padding'][i_layer][0],self.hparams['ae_decoding_x_padding'][i_layer][0])))
                        self.conv_t_pads['convtranspose'+str(global_layer_num)] = None
                    else:
                        # If uneven padding originally, don't pad here and do it in forward()
                        self.decoder.add_module('convtranspose'+str(global_layer_num),nn.ConvTranspose2d(in_channels=in_channels,out_channels=self.hparams['ae_decoding_n_channels'][i_layer],kernel_size=(self.hparams['ae_decoding_kernel_size'][i_layer],self.hparams['ae_decoding_kernel_size'][i_layer]),stride=(self.hparams['ae_decoding_stride_size'][i_layer],self.hparams['ae_decoding_stride_size'][i_layer])))
                        self.conv_t_pads['convtranspose'+str(global_layer_num)] = [self.hparams['ae_decoding_x_padding'][i_layer][0] ,self.hparams['ae_decoding_x_padding'][i_layer][1] ,self.hparams['ae_decoding_y_padding'][i_layer][0] ,self.hparams['ae_decoding_y_padding'][i_layer][1]]


                ## BatchNorm + Relu or Sigmoid if last layer
                if i_layer == (len(self.hparams['ae_decoding_n_channels'])-1) and not self.hparams['ae_decoding_last_FF_layer']: # last layer: no batch norm/sigmoi nonlin
                    self.decoder.add_module('sigmoid'+str(global_layer_num),nn.Sigmoid())
                else:
                    if self.hparams['ae_batch_norm']:
                        self.decoder.add_module('batch norm'+str(global_layer_num),nn.BatchNorm2d(self.hparams['ae_decoding_n_channels'][i_layer],momentum=self.hparams['ae_batch_norm_momentum']))

                    self.decoder.add_module('relu'+str(global_layer_num),nn.LeakyReLU(0.05))
                global_layer_num+=1
         
        ## Optional final FF layer (rarely used)
        if self.hparams['ae_decoding_last_FF_layer']: # have last layer be feedforward if this is 1
            self.decoder.add_module('last_FF'+str(global_layer_num)+'',nn.Linear(self.hparams['ae_decoding_x_dim'][-1]*self.hparams['ae_decoding_y_dim'][-1]*self.hparams['ae_decoding_n_channels'][-1],self.hparams['ae_input_dim'][0]*self.hparams['ae_input_dim'][1]*self.hparams['ae_input_dim'][2]))
            self.decoder.add_module('sigmoid'+str(global_layer_num),nn.Sigmoid())

        if self.hparams['model_class'] == 'vae':
            raise NotImplementedError
        elif self.hparams['model_class'] == 'ae':
            pass
        else:
            raise ValueError('Not valid model type')
             
    def forward(self, x, pool_idx, target_output_size):

        # First FF layer/resize to be convolutional input
        x = self.FF(x)
        x = x.view(x.size(0),self.hparams['ae_decoding_starting_dim'][0], self.hparams['ae_decoding_starting_dim'][1], self.hparams['ae_decoding_starting_dim'][2])

        for name, layer in self.decoder.named_children():
            if isinstance(layer, nn.MaxUnpool2d):
                idx = pool_idx.pop(-1)
                outsize = target_output_size.pop(-1)
                x = layer(x,idx,outsize) 
            elif isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                if self.conv_t_pads[name]is not None:
                    x = F.pad(x,[-i for i in self.conv_t_pads[name]]) # asymmetric padding for convtranspose layer if necessary (-i does cropping!)
            elif isinstance(layer, nn.Linear):
                x = x.view(x.shape[0],-1)
                x = layer(x)
                x = x.view(-1,self.hparams['ae_input_dim'][0],self.hparams['ae_input_dim'][1],self.hparams['ae_input_dim'][2])
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
            input_size (list or tuple): y_pix x x_pix
        """
        super().__init__()

        self.n_latents = n_latents
        self.input_size = input_size
        self._build_model()

    def _build_model(self):
        self.output = nn.Linear(
            in_features=np.prod(self.input_size),
            out_features=self.n_latents,
            bias=True)

    def forward(self, x):
        # reshape
        x = x.view(x.size(0), -1)
        return self.output(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LinearAEDecoder(nn.Module):

    def __init__(self, n_latents, output_size):

        super().__init__()
        self.n_latents = n_latents
        self.output_size = output_size
        self._build_model()

    def _build_model(self):

        self.output = nn.Linear(
            in_features=self.n_latents,
            out_features=np.prod(self.output_size),
            bias=True)

    def forward(self, x):
        # push through
        x = self.output(x)
        # reshape
        x = x.view(x.size(0), *self.output_size)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class AE(nn.Module):

    def __init__(self, hparams):

        super(AE, self).__init__()
        self.hparams = hparams
        self.model_type = self.hparams['model_type']
        self.build_model()

    def build_model(self):

        if self.model_type == 'conv':
            self.encoding = ConvAEEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            n_latents = self.hparams['n_ae_latents']
            img_size = (self.hparams['y_pixels'], self.hparams['x_pixels'])
            self.encoding = LinearAEEncoder(n_latents, img_size)
            self.decoding = LinearAEDecoder(n_latents, img_size)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

    def forward(self, x):

        if self.model_type == 'conv':
            x, pool_idx, outsize = self.encoding(x)
            y = self.decoding(x, pool_idx, outsize)
        elif self.model_type == 'linear':
            x = self.encoding(x)
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

        if hparams['model_type'] == 'ff' or hparams['model_type'] == 'linear':
            self.model = NN(hparams)
        elif hparams['model_type'] == 'lstm':
            self.model = LSTM(hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % hparams['model_type'])

    def forward(self, x):
        return self.model(x)


class NN(nn.Module):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.__build_model()

    def __build_model(self):

        self.decoder = nn.ModuleList()

        in_size = self.hparams['input_size']

        # loop over hidden layers (0 layers <-> linear regression)
        global_layer_num = 0
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                out_size = self.hparams['n_final_units']
            else:
                out_size = self.hparams['n_int_units']

            # add layer
            if i_layer == 0:
                # first layer is 1d conv for incorporating past/future neural
                # activity
                layer = nn.Conv1d(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
                    padding=self.hparams['n_lags'])  # same output
                name = str('conv1d_layer_%02i' % global_layer_num)
            else:
                layer = nn.Linear(
                    in_features=in_size,
                    out_features=out_size)
                name = str('dense_layer_%02i' % global_layer_num)
            self.decoder.add_module(name, layer)

            # add activation
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

            # update layer info
            global_layer_num += 1
            in_size = out_size

        # final layer
        layer = nn.Linear(
            in_features=in_size,
            out_features=self.hparams['output_size'])
        self.decoder.add_module(
            'dense_layer_%02i' % global_layer_num, layer)

        if self.hparams['noise_dist'] == 'gaussian':
            activation = None
        elif self.hparams['noise_dist'] == 'poisson':
            activation = nn.Softplus()
        elif self.hparams['noise_dist'] == 'categorical':
            activation = None
        else:
            raise ValueError(
                '"%s" is an invalid noise dist' % self.hparams['noise_dist'])

        if activation:
            self.decoder.add_module(
                '%s_%02i' % (self.hparams['activation'], global_layer_num),
                activation)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): time x neurons

        Returns:

        """
        # print('Model input size is {}'.format(x.shape))
        # print()
        for name, layer in self.decoder.named_children():
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

        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LSTM(nn.Module):

    def __init__(self, hparams):
        raise NotImplementedError


class ConvVAEEncoder(nn.Module):
    raise NotImplementedError

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


class ConvVAEDecoder(nn.Module):
    raise NotImplementedError

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


class LinearVAEEncoder(nn.Module):
    raise NotImplementedError

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


class LinearVAEDecoder(nn.Module):
    raise NotImplementedError

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


class VAE(nn.Module):
    raise NotImplementedError

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


class SLDS(nn.Module):
    raise NotImplementedError

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
