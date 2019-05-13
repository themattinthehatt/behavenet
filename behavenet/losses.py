import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianNegLogProb(_Loss):
    """
    Minimize negative Gaussian log probability with learned covariance matrix.

    For now the covariance matrix is not data-dependent
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        if reduction != 'mean':
            raise NotImplementedError
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target, precision):
        output_dim = target.shape[1]
        # print(input.shape)
        # print(target.shape)
        # print(precision.shape)
        dist = MultivariateNormal(
            loc=input,
            covariance_matrix=1e-3 * torch.eye(output_dim) + precision)
        return torch.mean(-dist.log_prob(target))
