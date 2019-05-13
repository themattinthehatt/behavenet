import torch
import datta.core as core
from torch import nn, optim
from models import VAE
import math
import pandas as pd
from ast import literal_eval


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        for k in list(d.keys()):
            try:
                self.__dict__[k] = literal_eval(self.__dict__[k])
            except:
                pass

class VanillaRecognitionNetwork(nn.Module):
    def __init__(self, hparams):
        super(VanillaRecognitionNetwork, self).__init__()
        self.hparams = hparams
        self.__build_model()

    def __build_model(self):
        hp = self.hparams

        # TODO: Make a recognition network that takes in data (T x P)
        # and outputs a mean (T x H) and marginal variance (T x H) of 
        # a Gaussian posterior for the continuous latent states. 
        hp = pd.read_csv(self.hparams.init_vae_model_path+'meta_tags.csv')
        hp = dict(zip(hp['key'], hp['value']))
        vae_hparams = objectview(hp)

        vae_model = VAE(vae_hparams)

        vae_model.load_state_dict(torch.load(self.hparams.init_vae_model_path+'best_val_model.pt', map_location=lambda storage, loc: storage))
        VAE_encoder_model = vae_model.encoding
        #VAE_encoder_model.freeze()
        #VAE_encoder_model.training=False
        VAE_encoder_model.to(self.hparams.device)
        self.VAE_encoder_model = VAE_encoder_model

    def recognize(self, data):
        mu, var = self.VAE_encoder_model(data)
        return mu, var

    def sample(self, data):
        # Sample num_samples from the posterior with the recognized mean and variance
        mean, variance = self.recognize(data)

        return mean + torch.sqrt(variance) * torch.randn_like(mean)

    def log_proba(self, data, states):
        """
        @param data:   T (time) x 1 x P x P (pixels)
        @param states: T (time) x H (continuous state dim)
        
        @return: log variational density, log q(states | data)

        TODO: We could allow samples to be a 3-tensor with a third dimension of samples
        """
        means, variances = self.recognize(data)                                    # T x H
        lls = -0.5 * torch.sum((states - means)**2 / variances)                  # scalar
        lls += -0.5 * torch.sum(math.log(2 * math.pi) + torch.log(variances))     # scalar
        return lls
        