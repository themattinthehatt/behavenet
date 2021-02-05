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
        self.hparams = hparams
        self.model = None
        self.build_model()
        # choose loss based on noise distribution of the model
        if self.hparams['noise_dist'] == 'gaussian':
            self._loss = nn.MSELoss()
        elif self.hparams['noise_dist'] == 'gaussian-full':
            from behavenet.fitting.losses import GaussianNegLogProb
            self._loss = GaussianNegLogProb()  # model holds precision mat
        elif self.hparams['noise_dist'] == 'poisson':
            self._loss = nn.PoissonNLLLoss(log_input=False)
        elif self.hparams['noise_dist'] == 'categorical':
            self._loss = nn.CrossEntropyLoss()
        else:
            raise ValueError('"%s" is not a valid noise dist' % self.model['noise_dist'])

        def __str__(self):
            """Pretty print model architecture."""
            return self.model.__str__()

    def build_model(self):
        """Construct the model using hparams."""

        # TODO
        if self.hparams['model_type'] == 'mlp' or self.hparams['model_type'] == 'mlp-mv':
            self.model = HierarchicalMLP(self.hparams)
        elif self.hparams['model_type'] == 'lstm':
            self.model = HierarchicalLSTM(self.hparams)
        else:
            raise ValueError('"%s" is not a valid model type' % self.hparams['model_type'])

    def forward(self, x,dataset):
        """Process input data."""
        return self.model(x,dataset)

    def loss(self, data,dataset, accumulate_grad=True, chunk_size=200, **kwargs):
        # TODO
        """Calculate negative log-likelihood loss for supervised models.

         The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
         requirements low; gradients are accumulated across all chunks before a gradient step is
         taken.

         Parameters
         ----------
         data : :obj:`dict`
             signals are of shape (1, time, n_channels)
         accumulate_grad : :obj:`bool`, optional
             accumulate gradient for training step
         chunk_size : :obj:`int`, optional
             batch is split into chunks of this size to keep memory requirements low

         Returns
         -------
         :obj:`dict`
             - 'loss' (:obj:`float`): total loss (negative log-like under specified noise dist)
             - 'r2' (:obj:`float`): variance-weighted $R^2$ when noise dist is Gaussian
             - 'fc' (:obj:`float`): fraction correct when noise dist is Categorical

         """
        # self.dataset = dataset # it is passed as a kwarg, not sure how else to access this and pass it into forward()
        predictors = data[self.hparams['input_signal']][0]
        targets = data[self.hparams['output_signal']][0]

        max_lags = self.hparams['n_max_lags']

        batch_size = targets.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        outputs_all = []
        loss_val = 0
        for chunk in range(n_chunks):

            # take chunks of size chunk_size, plus overlap due to max_lags
            idx_beg = np.max([chunk * chunk_size - max_lags, 0])
            idx_end = np.min([(chunk + 1) * chunk_size + max_lags, batch_size])

            outputs, precision = self.forward(predictors[idx_beg:idx_end],dataset)

            # define loss on allowed window of data
            if self.hparams['noise_dist'] == 'gaussian-full':
                loss = self._loss(
                    outputs[max_lags:-max_lags],
                    targets[idx_beg:idx_end][max_lags:-max_lags],
                    precision[max_lags:-max_lags])
            else:
                loss = self._loss(
                    outputs[max_lags:-max_lags],
                    targets[idx_beg:idx_end][max_lags:-max_lags])

            if accumulate_grad:
                loss.backward()

            # get loss value (weighted by batch size)
            loss_val += loss.item() * outputs[max_lags:-max_lags].shape[0]

            outputs_all.append(outputs[max_lags:-max_lags].cpu().detach().numpy())

        loss_val /= batch_size
        outputs_all = np.concatenate(outputs_all, axis=0)

        if self.hparams['noise_dist'] == 'gaussian' or \
                self.hparams['noise_dist'] == 'gaussian-full':
            # use variance-weighted r2s to ignore small-variance latents
            r2 = r2_score(
                targets[max_lags:-max_lags].cpu().detach().numpy(),
                outputs_all,
                multioutput='variance_weighted')
            fc = 0
        elif self.hparams['noise_dist'] == 'poisson':
            raise NotImplementedError
        elif self.hparams['noise_dist'] == 'categorical':
            r2 = 0
            fc = accuracy_score(
                targets[max_lags:-max_lags].cpu().detach().numpy(),
                np.argmax(outputs_all, axis=1))
        else:
            raise ValueError('"%s" is not a valid noise_dist' % self.hparams['noise_dist'])

        return {'loss': loss_val, 'r2': r2, 'fc': fc}


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
        self.decoder = nn.ModuleList()

        global_layer_num = 0
        # Ask if the input size field of the hparams should be populated according to the multiple datasets that are
        # present in the datagenerator.datasets somewhere else? Because at the moment hparams just has one single inp dim
        out_size = self.hparams['n_hid_units']# fix it to the input size of the global backbone network
        # for i,i_layer in enumerate(range(len(sess_ids))):
        #     in_size = self.hparams['input_size'][i]
        #
        #     # first layer is 1d conv for incorporating past/future neural activity
        #     # Separate 1d conv for each dataset
        #     layer = nn.Conv1D(in_channels=in_size, out_channels=out_size,
        #                       kernel_size=self.hparams['n_lags']*2+1, #window around t
        #                       padding=self.hparams['n_lags'])# same output
        #     name = str('conv1d_layer_%02i'% global_layer_num)
        #     self.decoder.add_module(name,layer)
        #     self.final_layer = name

        layer = nn.ModuleList([
            nn.Conv1d(in_channels=in_size, out_channels=out_size,
                      kernel_size=self.hparams['n_lags']*2+1,
                      padding=self.hparams['n_lags'])
            for in_size in self.hparams['input_size']
        ])

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
                raise ValueError('"%s" is an invalid noise dist'% self.hparams['noise_dist'])

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

        # add layer for data dependent precision matrix if requires
        if self.hparams['n_hid_layers'] == 0 and self.hparams['noise_dist'] == 'gaussian-full':
            # build sqrt of precision matrix
            self.precision_sqrt = nn.Linear(in_features=in_size, out_features=out_size**2)
        else:
            self.precision_sqrt = None

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # loop over hidden layers
        for i_layer in range(self.hparams['n_hid_layers']):

            if i_layer == self.hparams['n_hid_layers'] - 1:
                out_size = self.hparams['output_size']
            else:
                out_size = self.hparams['n_hid_units']

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i'%global_layer_num)
            self.decoder.add_module(name,layer)
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

        in_size_list = self.hparams['input_size']

        #
        # if self.hparams['n_hid_layers'] == 0:
        #     out_size

    def forward(self, x,dataset):
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
        # sess_id = [s for s in self.hparams['input_size']]
        y = None
        for name, layer in self.decoder.named_children():

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer[dataset](x.transpose(1,0).unsqueeze(0)).squeeze().transpose(1,0)
                # x = layer(x.transpose(1,0).unsqueeze(0)).squeeze().transpose(1,0)
            else:
                x = layer(x)

        return x, y
        # pass

class HierarchicalLSTM(BaseModule):
    """Feedforward neural network model."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = None
        self.build_model()
        self.hidden_cell = (torch.zeros(hparams["stack"], hparams["batch"], hparams["hidden_layer_size"]),
                            torch.zeros(hparams["stack"], hparams["batch"], hparams["hidden_layer_size"]))

    def __str__(self):
        """Pretty print model architecture."""
        # TODO
        pass

    def build_model(self):
        """Construct the model."""
        # TODO
        self.decoder = nn.ModuleList()

        global_layer_num = 0

        out_size = self.hparams['n_hid_units']# fix it to the input size of the global backbone network

        in_size_1  = self.hparams['input_size'][0]
        in_size_2 = self.hparams['input_size'][1]


        layer = nn.ModuleList(
            [
                nn.Linear(in_size_1, self.hparams['lstm_in_size'])
            ])
        name = str('InputMLP_layer_%02i' % global_layer_num)
        self.decoder.add_module(name, layer)

        # # Add activation
        # global_layer_num += 1
        # name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
        # activation = nn.ReLU()
        # self.decoder.add_module(name, activation)

        # Add a second head of linear and activations
        global_layer_num += 1
        layer = nn.ModuleList(
            [
                nn.Linear(in_size_2, self.hparams['lstm_in_size'])
            ])
        name = str('InputMLP_layer_%02i' % global_layer_num)
        self.decoder.add_module(name, layer)

        # # Add activation
        # global_layer_num += 1
        # name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
        # activation = nn.ReLU()
        # self.decoder.add_module(name, activation)

        # update layer info # add lstm layer
        global_layer_num += 1
        layer = nn.LSTM(input_size=self.hparams["lstm_in_size"], hidden_size=self.hparams["hidden_layer_size"], num_layers=self.hparams["stack"])
        name = str('lstm_layer_%02i'%global_layer_num)
        self.decoder.add_module(name,layer)

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # add linear layer
        layer = nn.Linear(in_features=self.hparams["hidden_layer_size"],out_features=self.hparams["output_size"])
        name = str('dense_layer_%02i'%global_layer_num)
        self.decoder.add_module(name,layer)
        self.final_layer = name




    def forward(self, x,dataset):
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
        # sess_id = [s for s in self.hparams['input_size']]



        y = None
        for name, layer in self.decoder.named_children():

            if name == 'InputMLP_layer_00' and dataset==0:
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer[0](x.unsqueeze(0)).squeeze().transpose(1,0)

            # if name=='relu_01' and dataset==0:
            #     x = layer(x)

            if name == 'InputMLP_layer_01' and dataset==1:
                # input is batch x in_channels x time
                # output is batch x out_channels x time
                x = layer[0](x.unsqueeze(0)).squeeze().transpose(1,0)

            # if name=='relu_03' and dataset==1:
            #     x = layer(x)

            if name == 'lstm_layer_02':
                x = x.reshape(189,1,-1)
                x, _ = layer(x,self.hidden_cell)

            elif name == 'dense_layer_03':
                x = layer(x)

        return x.reshape(189,10), y



