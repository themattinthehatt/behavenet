"""Custom losses for PyTorch models."""

import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal


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
        binary mask that is the same size as `y` and `y_mu`; by placing 0 entries in the mask,
        the corresponding dimensions will not contribute to the loss term, and will therefore
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
