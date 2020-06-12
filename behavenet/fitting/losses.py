"""Custom losses for PyTorch models."""

import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal

LN2PI = np.log(2 * np.pi)


class GaussianNegLogProb(_Loss):
    """Minimize negative Gaussian log probability with learned covariance matrix.

    For now the covariance matrix is not data-dependent
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        if reduction != 'mean':
            raise NotImplementedError
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, precision):
        output_dim = target.shape[1]
        dist = MultivariateNormal(
            loc=input,
            covariance_matrix=1e-3 * torch.eye(output_dim) + precision)
        return torch.mean(-dist.log_prob(target))


def mse(y_pred, y_true, masks=None):
    """Compute mean square error (MSE) loss with masks.

    Parameters
    ----------
    y_pred : :obj:`torch.Tensor`
        predicted data
    y_true : :obj:`torch.Tensor`
        true data
    masks : :obj:`torch.Tensor`, optional
        binary mask that is the same size as `y_pred` and `y_true`; by placing 0 entries in the
        mask, the corresponding dimensions will not contribute to the loss term, and will therefore
        not contribute to parameter updates

    Returns
    -------
    :obj:`torch.Tensor`
        mean square error computed across all dimensions

    """
    if masks is not None:
        return torch.mean(((y_pred - y_true) ** 2) * masks)
    else:
        return torch.mean((y_pred - y_true) ** 2)


def gaussian_ll(y_pred, y_mean, masks=None, std=1):
    """Compute multivariate Gaussian log-likelihood with a fixed diagonal noise covariance matrix.

    Parameters
    ----------
    y_pred : :obj:`torch.Tensor`
        predicted data of shape (n_frames, ...)
    y_mean : :obj:`torch.Tensor`
        true data of shape (n_frames, ...)
    masks : :obj:`torch.Tensor`, optional
        binary mask that is the same size as `y_pred` and `y_true`; by placing 0 entries in the
        mask, the corresponding dimensions will not contribute to the loss term, and will therefore
        not contribute to parameter updates
    std : :obj:`float`, optional
        fixed standard deviation for all dimensions in the multivariate Gaussian

    Returns
    -------
    :obj:`torch.Tensor`
        Gaussian log-likelihood summed across dims, averaged across batch

    """
    dims = y_pred.shape
    n_dims = np.prod(dims[1:])  # first value is n_frames in batch
    log_var = np.log(std ** 2)

    if masks is not None:
        diff_sq = ((y_pred - y_mean) ** 2) * masks
    else:
        diff_sq = (y_pred - y_mean) ** 2

    ll = - (0.5 * LN2PI + 0.5 * log_var) * n_dims - (0.5 / (std ** 2)) * diff_sq.sum(
        axis=tuple(1+np.arange(len(dims[1:]))))

    return torch.mean(ll)


def kl_div_to_std_normal(mu, logvar):
    """Compute element-wise KL(q(z) || N(0, 1)) where q(z) is a normal parameterized by mu, logvar.

    Parameters
    ----------
    mu : :obj:`torch.Tensor`
        mean parameter of shape (n_frames, n_dims)
    logvar
        log variance parameter of shape (n_frames, n_dims)

    Returns
    -------
    :obj:`torch.Tensor`
        KL divergence summed across dims, averaged across batch

    """
    kl = 0.5 * torch.sum(logvar.exp() - logvar + mu.pow(2) - 1, dim=1)
    return torch.mean(kl)


def gaussian_ll_to_mse(ll, n_dims, gaussian_std=1, mse_std=1):
    """Convert a Gaussian log-likelihood term to MSE by removing constants and swapping variances.

    - NOTE:
        does not currently return correct values if gaussian ll is computed with masks

    Parameters
    ----------
    ll : :obj:`float`
        original Gaussian log-likelihood
    n_dims : :obj:`int`
        number of dimensions in multivariate Gaussian
    gaussian_std : :obj:`float`
        std used to compute Gaussian log-likelihood
    mse_std : :obj:`float`
        std used to compute MSE

    Returns
    -------
    :obj:`float`
        MSE value

    """
    llc = np.copy(ll)
    llc += (0.5 * LN2PI + 0.5 * np.log(gaussian_std ** 2)) * n_dims  # remove constant
    llc *= -(gaussian_std ** 2) / 0.5  # undo scaling by variance
    llc /= n_dims  # change sum to mean
    llc *= 1.0 / (mse_std ** 2)  # scale by mse variance
    return llc
