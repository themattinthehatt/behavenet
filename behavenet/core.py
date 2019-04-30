"""
Functions to compute log_pi0, log_Ps, lls
"""
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import math


# Helpers
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


# Log likelihoods
def expected_log_likelihood(expectations, log_pi0, log_Ps, lls):
    Ez, Ezzp1, normalizer = expectations
    ell = torch.sum(Ez[0] * log_pi0)
    ell += torch.sum(Ezzp1 * log_Ps)
    ell += torch.sum(Ez * lls)
    return ell

# Dynamics models
def gaussian_ar_log_proba(model, data):
    hparams = model.hparams
    # Compute the mean, A_{z_t} x_{t-1} + b_{z_t}  for t > nlags
    means = torch.transpose(
        torch.matmul(torch.cat(([data[hparams['nlags']-i:hparams['batch_size']-i] for i in range(hparams['nlags'],0,-1)]), dim=1), model.As),1,0)+ model.bs

    covs = torch.matmul(model.sqrt_Qs, model.sqrt_Qs.transpose(1, 2))
    lls = MultivariateNormal(means, covs).log_prob(data[hparams['nlags']:].unsqueeze(1))
    assert lls.shape == (data.shape[0] - hparams['nlags'], hparams['n_discrete_states'])

    lls = torch.cat((torch.zeros(hparams['nlags'], hparams['n_discrete_states']), lls), 0)

    return lls


def diagonal_gaussian_ar_log_proba(model, data):
    hparams = model.hparams

    # Compute the mean, A_{z_t} x_{t-1} + b_{z_t}  for t > nlags
    means = torch.transpose(
        torch.matmul(torch.cat(([data[hparams['nlags']-i:hparams['batch_size']-i]
                                 for i in range(hparams['nlags'],0,-1)]), dim=1), model.As),1,0)+ model.bs

    # Concatenate the mean for the first t = 1...nlags
    # means = torch.cat((model.bs.view(1, hparams.n_discrete_states, hparams.latent_dim_size_h).repeat(hparams.nlags,1,1) , means), 0)

    lls = -0.5 * torch.sum((data[hparams['nlags']:].unsqueeze(1) - means)**2 / F.softplus(model.inv_softplus_Qs), dim=2)
    lls += -0.5 * torch.sum(math.log(2 * math.pi) + torch.log(F.softplus(model.inv_softplus_Qs)), dim=1)
    lls = torch.cat((torch.zeros(hparams['nlags'], hparams['n_discrete_states']), lls), 0)
    return lls

# def studentst_ar_log_proba(model):
#    D = self.D
#    mus = self._compute_mus(data, input, mask, tag)
#    sigmas = self._compute_sigmas(data, input, mask, tag)
#    nus = np.exp(self.inv_nus)
#
#    resid = data[:, None, :] - mus
#    z = resid / sigmas
#    return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
#        gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
#        -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=-1)


# Transition models
def stationary_log_transition_proba(model):
    hparams = model.hparams
    normalized_Ps = model.stat_log_transition_proba - log_sum_exp(model.stat_log_transition_proba, dim=-1, keepdim=True)
    return normalized_Ps.unsqueeze(0).repeat(hparams['batch_size']-1,1,1)


def dirichlet_prior(model):
    hparams = model.hparams
    log_Ps = model.stat_log_transition_proba-log_sum_exp(model.stat_log_transition_proba,dim=-1,keepdim=True)

    lp = 0
    for i_state in range(hparams['n_discrete_states']):
        concentration = hparams['alpha']*torch.ones(hparams['n_discrete_states']).to(hparams['device']) / hparams['n_discrete_states']
        concentration[i_state] += hparams['kappa']
        lp += ((log_Ps[i_state] * (concentration - 1.0)).sum(-1) +
            torch.lgamma(concentration.sum(-1)) -
            torch.lgamma(concentration).sum(-1))
    return lp


# Emission models for the SLDS
def gaussian_emissions_diagonal_variance(model, data, states):
    """
    Gaussian likelihood on the observed data.  Assume that the
    model has a diagonal observation variance, sigmasq.

    @param model:  an SLDS object
    @param data:   a T (time) x P (pixels) tensor
    @param states: a T (time) x H (latent dim) tensor

    # TODO: We could allow states to be a 3-tensor whose last dim is posterior samples
    """
    hparams = model.hparams
    means, variances = model.decode(states)

    lls = torch.sum((-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(variances)-0.5 * (data - means)**2 / variances),dim=1)

    return lls

def uniform_initial_distn(model):
    hparams = model.hparams
    return -math.log(hparams['n_discrete_states']) * torch.ones(hparams['n_discrete_states'])


# Initialization helpers
def initialize_with_lr(model, hp, data_gen, L2_reg=0.01):
    nb_tng_batches = data_gen.n_max_train_batches
    As = torch.tensor(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h*hp.nlags, hp.latent_dim_size_h)))
    bs = torch.tensor(torch.zeros((hp.n_discrete_states, hp.latent_dim_size_h)))
    inv_softplus_Qs = torch.tensor(torch.ones((hp.n_discrete_states, hp.latent_dim_size_h)))

    # Split the data into n_discrete_state chunks, fit linear regression to each chunk separately
    data_split = np.floor(nb_tng_batches/hp.n_discrete_states)

    i_discrete_state = 0
    start_collecting=1

    for batch_nb in range(nb_tng_batches):

        # Get this batch of data
        data = data_gen.next_train_batch()
        if hp.low_d_type == 'vae':
            this_data = data['depth'].unsqueeze(2)
        elif hp.low_d_type == 'pca':
            this_data = data['pca_score']

        for i_session in range(this_data.shape[0]):

            low_d = model.get_low_d(this_data[i_session])
            X = torch.cat(([low_d[hp.nlags-1-i:low_d.shape[0]-1-i] for i in range(hp.nlags,-1,-1)]),dim=1)
            X = F.pad(X,(1,0),value=1)
            Y = low_d[hp.nlags:]

            # Collect X/Y/XTX/XTY
            if start_collecting: # start of a new chunk
                all_X = X
                all_Y = Y
                XTX = torch.matmul(X.transpose(1,0),X)
                XTY = torch.matmul(X.transpose(1,0),Y)
                start_collecting=0
            else:
                all_X = torch.cat((all_X,X),0)
                all_Y = torch.cat((all_Y,Y),0)
                XTX += torch.matmul(X.transpose(1,0),X)
                XTY += torch.matmul(X.transpose(1,0),Y)

        if i_discrete_state < hp.n_discrete_states:
            if np.mod(batch_nb+1,data_split)==0:

                # Calculate weights for this chunk
                reg_XTX = XTX+L2_reg*torch.eye(X.shape[1]).to(hp.device)
                XTX_inv = torch.inverse(reg_XTX)
                W = torch.matmul(XTX_inv,XTY)

                As.data[i_discrete_state] = W[1:,:].data
                bs.data[i_discrete_state] = W[0,:].data

                # Reconstruct to get residuals/covariances
                Y_hat = torch.matmul(all_X,W)
                residuals = Y_hat-all_Y
                Qs = torch.var(residuals,0).data
                inv_softplus_Qs.data[i_discrete_state] = torch.log(torch.exp(Qs)-1)

                # Reset
                start_collecting=1
                i_discrete_state +=1

    if i_discrete_state < hp.n_discrete_states-1:
        raise Exception('ERROR WITH INITIALIZATION')

    return As, bs, inv_softplus_Qs


