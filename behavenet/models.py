import torch
import datta.core as core
from torch import nn, optim
import pandas as pd
import torch.nn.functional as F
import numpy as np
from ast import literal_eval

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        for k in list(d.keys()):
            try:
                self.__dict__[k] = literal_eval(self.__dict__[k])
            except:
                pass

class ConvVAEEncoder(nn.Module):

    def __init__(self, latent_dim_size_h, bn):

        super(ConvVAEEncoder, self).__init__()

        self.latent_dim_size_h = latent_dim_size_h
        self.bn = bn
        self.__build_model()

    def __build_model(self):
        # TO DO: make flexible

        if self.bn:
            self.encoder = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
                        stride=2, padding=1, bias=False),
              nn.BatchNorm2d(32),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                        stride=2, padding=1, bias=True),
              nn.BatchNorm2d(64),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4,
                        stride=2, padding=1, bias=True),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                        stride=2, padding=1, bias=True),
              nn.BatchNorm2d(512),
              nn.LeakyReLU(0.05, inplace=True)
            )
        else:
            self.encoder = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
                        stride=2, padding=1, bias=False),
             # nn.BatchNorm2d(32),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                        stride=2, padding=1, bias=True),
             # nn.BatchNorm2d(64),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4,
                        stride=2, padding=1, bias=True),
             # nn.BatchNorm2d(256),
              nn.LeakyReLU(0.05, inplace=True),
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                        stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(512),
              nn.LeakyReLU(0.05, inplace=True)
            )
            
        self.out_img = (512, 5, 5)
        self.prior_mu = nn.Linear(512*5*5, self.latent_dim_size_h)
        #self.h_var = nn.Parameter(1e-6*torch.ones(100,10),requires_grad=False)
        self.prior_logvar = nn.Linear(512*5*5, self.latent_dim_size_h)
        self.softplus = nn.Softplus()
    def forward(self, x):
        if x.dim() == 3:
          x = x.view(x.size(0), 1, x.size(1), x.size(2))
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.prior_mu(h), self.softplus(self.prior_logvar(h))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class ConvVAEDecoder(nn.Module):

    def __init__(self, latent_dim_size_h, pixel_size, y_var_value, y_var_parameter, bn):

        super(ConvVAEDecoder, self).__init__()
        self.latent_dim_size_h = latent_dim_size_h
        self.y_var_value = y_var_value
        self.y_var_parameter = y_var_parameter
        self.bn = bn
        self.pixel_size = pixel_size
        self.__build_model()

    def __build_model(self):

         # TO DO: make flexible
        self.out_img = (512, 5, 5)

        self.linear_decode = nn.Linear(self.latent_dim_size_h, 512*5*5)
        if self.bn:
            self.decoder = nn.Sequential(
              nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
                                 stride=2, padding=1, bias=True),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
                                 stride=2, padding=1, bias=True),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                 stride=2, padding=1, bias=True),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                                 stride=2, padding=1, bias=True),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,
                                 stride=2, padding=1, bias=True),
              nn.BatchNorm2d(16),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,
                        padding=1),
              nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
              nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
                                 stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
                                 stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                 stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                                 stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,
                                 stride=2, padding=1, bias=True),
            #  nn.BatchNorm2d(16),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1,
                        padding=1),
              nn.Sigmoid()
            )    

        if self.y_var_parameter:
            inv_softplus_var = np.log(np.exp(self.y_var_value)-1)
            self.y_var = nn.Parameter(inv_softplus_var*torch.ones(self.pixel_size,self.pixel_size),requires_grad=True)
        else:
            self.y_var = nn.Parameter(self.y_var_value*torch.ones(1),requires_grad=False)

    def forward(self, x):

        y = self.linear_decode(x)
        y = y.view(y.size(0), *self.out_img)

        y_mu = self.decoder(y)
        if self.y_var_parameter:
            y_var = F.softplus(self.y_var).unsqueeze(0).unsqueeze(0).expand(y_mu.shape[0],-1,-1,-1)
        else:
            y_var = self.y_var

        return y_mu, y_var

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LinearVAEEncoder(nn.Module):

    def __init__(self, latent_dim_size_h, pixel_size):

        super(LinearVAEEncoder, self).__init__()

        self.latent_dim_size_h = latent_dim_size_h
        self.pixel_size=pixel_size
        self.__build_model()

    def __build_model(self):
      
        self.prior_mu = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
        self.prior_logvar = nn.Linear(self.pixel_size*self.pixel_size, self.latent_dim_size_h,bias=True)
      # self.h_var = nn.Parameter(1e-1*torch.ones(100,10),requires_grad=True)
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.prior_mu(x), self.softplus(self.prior_logvar(x))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class LinearVAEDecoder(nn.Module):

    def __init__(self, latent_dim_size_h, pixel_size, y_var_value, y_var_parameter, encoding):

        super(LinearVAEDecoder, self).__init__()
        self.latent_dim_size_h = latent_dim_size_h
        self.y_var_value = y_var_value
        self.encoding = encoding
        self.pixel_size = pixel_size
        self.y_var_parameter = y_var_parameter
        self.__build_model()

    def __build_model(self):

        self.bias = nn.Parameter(torch.zeros(self.pixel_size*self.pixel_size),requires_grad=True)
        if self.y_var_parameter:
            inv_softplus_var = np.log(np.exp(self.y_var_value)-1)
            self.y_var = nn.Parameter(inv_softplus_var*torch.ones(self.pixel_size,self.pixel_size),requires_grad=True)
        else:
            self.y_var = nn.Parameter(self.y_var_value*torch.ones(1),requires_grad=False)

    def forward(self, x):

        y_mu =  F.linear(x, self.encoding.prior_mu.weight.t()) + self.bias 
        y_mu = y_mu.view(y_mu.size(0), 1, self.pixel_size,self.pixel_size)

        if self.y_var_parameter:
            y_var = F.softplus(self.y_var).unsqueeze(0).unsqueeze(0).expand(y_mu.shape[0],-1,-1,-1)
        else:
            y_var = self.y_var
        return y_mu, y_var

class VAE(nn.Module):

    def __init__(self, hparams):

        super(VAE, self).__init__()
        self.hparams = hparams

        self.__build_model()

    def __build_model(self):

        if self.hparams.vae_type=='conv':
            self.encoding = ConvVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.bn)
            self.decoding = ConvVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter, self.hparams.bn)
        elif self.hparams.vae_type=='linear':
            self.encoding = LinearVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size)
            self.decoding = LinearVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter, self.encoding)

    def reparameterize(self, mu, var, random_draw):
       if random_draw:
          std = torch.pow(var,0.5) 
          eps = torch.randn_like(std)
          return eps.mul(std).add_(mu)
       else:
          return mu

    def forward(self, x, random_draw=1):

        h_mu, h_var = self.encoding(x)
        x  = self.reparameterize(h_mu,h_var,random_draw)
        y_mu, y_var = self.decoding(x)

        return y_mu, y_var, h_mu, h_var

class AE(nn.Module):

    def __init__(self, hparams):

        super(AE, self).__init__()
        self.hparams = hparams

        self.__build_model()

    def __build_model(self):

        if self.hparams.ae_type=='conv':
            self.encoding = ConvVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.bn)
            self.decoding = ConvVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter, self.hparams.bn)
        elif self.hparams.ae_type=='linear':
            self.encoding = LinearVAEEncoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size)
            self.decoding = LinearVAEDecoder(self.hparams.latent_dim_size_h, self.hparams.pixel_size, self.hparams.y_var_value, self.hparams.y_var_parameter,  self.encoding)

    def forward(self, x):

        h_mu, h_var = self.encoding(x)
        y_mu, y_var = self.decoding(h_mu)

        return y_mu, y_var, h_mu

class ARHMM(nn.Module):
    def __init__(self, hparams, dynamics="gaussian"):
        super(ARHMM, self).__init__()
        self.hparams = hparams

        assert dynamics in ("gaussian", "studentst")
        self.dynamics = dynamics.lower()

        self.__build_model()

    def __build_model(self):
        hp = self.hparams
        dynamics = self.dynamics
        
        # Dynamics parameters
        self.As = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h*hp.nlags, hp.latent_dim_size_h)))
        self.bs = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h)))
        self.inv_softplus_Qs = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))

        if dynamics.lower() == "studentst":
            self.inv_softplus_nus = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))
        
        # Transition parameters
        self.stat_log_transition_proba = \
                nn.Parameter(torch.log(
                hp.transition_init * torch.eye(hp.n_discrete_states) + (1-hp.transition_init) / hp.n_discrete_states * torch.ones((hp.n_discrete_states, hp.n_discrete_states))))

        if self.hparams.low_d_type == 'vae':
            hp = pd.read_csv(self.hparams.vae_model_path+'meta_tags.csv')
            hp = dict(zip(hp['key'], hp['value']))
            vae_hparams = objectview(hp)

            vae_model = VAE(vae_hparams)

            vae_model.load_state_dict(torch.load(self.hparams.vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
            VAE_encoder_model = vae_model.encoding
            VAE_encoder_model.freeze()
            VAE_encoder_model.training=False
            VAE_encoder_model.to(self.hparams.device)
            self.VAE_encoder_model = VAE_encoder_model

    def initialize(self,method="lr", *args, **kwargs):
        init_methods = dict(lr=self._initialize_with_lr)
        if method not in init_methods:
            raise Exception("Invalid initialization method: {}".format(method))
        return init_methods[method](*args, **kwargs)
        
    def _initialize_with_lr(self, data_gen, L2_reg=0.01):
        self.As.data, self.bs.data, self.inv_softplus_Qs.data = core.initialize_with_lr(self, self.hparams,data_gen, L2_reg=L2_reg)
        
    def log_pi0(self, *args):
        return core.uniform_initial_distn(self).to(self.hparams.device)

    def log_prior(self,*args):
        return core.dirichlet_prior(self)

    def log_transition_proba(self, *args):
        return core.stationary_log_transition_proba(self)

    def log_dynamics_proba(self, data, *args):
        if self.dynamics == "gaussian":
            return core.gaussian_ar_log_proba(self,data)
        elif self.dynamics == "studentst":
            return core.studentst_ar_log_proba(self,data)
        else:
            raise Exception("Invalid dynamics: {}".format(self.dynamics))
    def get_low_d(self,signal):  
        if self.hparams.low_d_type == 'vae':
            self.VAE_encoder_model.training=False
            signal,_= self.VAE_encoder_model(signal)
            if self.hparams.whiten_vae:
                mean_h = np.load('normalization_values/vae_mean.npy')
                whiten_h = np.load('normalization_values/vae_whitening_matrix.npy')
                apply_whitening = lambda x:  np.linalg.solve(whiten_h, (x-mean_h).T).T 
                signal = apply_whitening(signal[:,:10].cpu().detach().numpy())
                signal = torch.tensor(signal).to(self.hparams.device).float()
        elif self.hparams.low_d_type == 'pca':
            signal = signal[:,:10]
        else:
            raise NotImplementedError
        return signal

class SLDS(nn.Module):
    """
    This will look a lot like an ARHMM but it has a decoder for mapping 
    continuous latent states to observations.
    """

    def __init__(self, hparams, dynamics="gaussian", emissions="gaussian"):
        super(SLDS, self).__init__()
        self.hparams = hparams

        assert dynamics.lower() in ("gaussian", "studentst")
        self.dynamics = dynamics.lower()

        assert emissions.lower() in ("gaussian",)
        self.emissions = emissions.lower()

        self.__build_model()

    def __build_model(self):
        hp = self.hparams
        dynamics = self.dynamics
        
        # Dynamics parameters
        self.As = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h*hp.nlags, hp.latent_dim_size_h)))
        self.bs = nn.Parameter(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h)))
        self.inv_softplus_Qs = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))

        if dynamics.lower() == "studentst":
            self.inv_softplus_nus = nn.Parameter(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))
        
        # Transition parameters
        self.stat_log_transition_proba = \
                nn.Parameter(torch.log(
                hp.transition_init * torch.eye(hp.n_discrete_states) + 
                (1-hp.transition_init) / hp.n_discrete_states * torch.ones((hp.n_discrete_states, hp.n_discrete_states))))

        if self.hparams.low_d_type == 'vae':
            hp = pd.read_csv(self.hparams.init_vae_model_path+'meta_tags.csv')
            hp = dict(zip(hp['key'], hp['value']))
            vae_hparams = objectview(hp)

            vae_model = VAE(vae_hparams)
            vae_model2 = VAE(vae_hparams)

            vae_model.load_state_dict(torch.load(self.hparams.init_vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
            VAE_decoder_model = vae_model.decoding
            VAE_decoder_model.to(self.hparams.device)
            self.VAE_decoder_model = VAE_decoder_model
            #self.VAE_decoder_model.encoding.prior_mu.bias=None
            #self.VAE_decoder_model.encoding.prior_logvar.weight=None
            #self.VAE_decoder_model.encoding.prior_logvar.bias=None

            vae_model2.load_state_dict(torch.load(self.hparams.init_vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
            VAE_encoder_model = vae_model2.encoding
            VAE_encoder_model.freeze()
            #VAE_encoder_model.training=False
            VAE_encoder_model.to(self.hparams.device)
            self.VAE_encoder_model = VAE_encoder_model

    def decode(self, states):
        """
        Pass the continuous latent state through the decoder network 
        get the mean of the observations.

        @param states: a T (time) x H (latent dim)
        """
        y_mu, y_var = self.VAE_decoder_model(states)
        return y_mu, y_var

    # The remainder of the methods look like those of the ARHMM,
    # but now we also have an emission probability of the data given 
    # the continuous latent states.
    def initialize(self,method="lr", *args, **kwargs):
        init_methods = dict(lr=self._initialize_with_lr)
        if method not in init_methods:
            raise Exception("Invalid initialization method: {}".format(method))
        return init_methods[method](*args, **kwargs)
        
    def _initialize_with_lr(self, data_gen, L2_reg=0.01):
        self.As.data, self.bs.data, self.inv_softplus_Qs.data = core.initialize_with_lr(self, self.hparams,data_gen, L2_reg=L2_reg)
        
    def get_low_d(self,signal):  
        if self.hparams.low_d_type == 'vae':
            signal,_= self.VAE_encoder_model(signal)
        elif self.hparams.low_d_type == 'pca':
            pass
        else:
            raise NotImplementedError
        return signal

    def log_pi0(self, *args):
        return core.uniform_initial_distn(self).to(self.hparams.device)

    def log_prior(self,*args):
        return core.dirichlet_prior(self)

    def log_transition_proba(self, *args):
        return core.stationary_log_transition_proba(self)

    def log_dynamics_proba(self, data, *args):
        if self.dynamics == "gaussian":
            return core.gaussian_ar_log_proba(self, data)
        elif self.dynamics == "studentst":
            return core.studentst_ar_log_proba(self, data)
        else:
            raise Exception("Invalid dynamics: {}".format(self.dynamics))

    def log_emission_proba(self, data, states):
        """
        Compute the likelihood of the data given the continuous states.
        """
        if self.emissions == "gaussian":
            return core.gaussian_emissions_diagonal_variance(self, data, states)
        else:
            raise Exception("Invalid emissions: {}".format(self.emissions))
