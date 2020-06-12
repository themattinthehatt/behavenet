import numpy as np
import torch
from behavenet.fitting import losses

LN2PI = np.log(2 * np.pi)


def test_mse():

    # test basic
    x = torch.rand((5, 3))
    mse = losses.mse(x, x)
    assert mse == 0

    # test with masks
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    y = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float)
    m = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float)
    mse = losses.mse(x, y, m)
    assert mse == 0.5


def test_gaussian_ll():

    # test basic
    n_batch = 5
    n_dims = 3
    std = 1
    x = torch.rand((n_batch, n_dims))
    ll = losses.gaussian_ll(x, x, masks=None, std=std)
    assert ll == - (0.5 * LN2PI + 0.5 * np.log(std ** 2)) * n_dims

    # test with masks
    x = torch.ones(n_batch, n_dims)
    y = torch.zeros(n_batch, n_dims)
    m = torch.zeros(n_batch, n_dims)
    for b in range(n_batch):
        m[b, 0] = 1
    ll = losses.gaussian_ll(x, y, masks=m, std=std)
    assert ll == - (0.5 * LN2PI + 0.5 * np.log(std ** 2)) * n_dims - (0.5 / (std ** 2))


def test_kl_div_to_std_normal():

    mu = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)
    kl = losses.kl_div_to_std_normal(mu, logvar)
    assert kl == 0


def test_gaussian_ll_to_mse():

    n_batch = 5
    n_dims = 3
    std = 1
    x = torch.ones(n_batch, n_dims)
    y = torch.zeros(n_batch, n_dims)
    ll = losses.gaussian_ll(x, y, std=std)
    mse_ = 2 * (-ll - (0.5 * LN2PI + 0.5 * np.log(std ** 2)) * n_dims) / n_dims
    mse = losses.gaussian_ll_to_mse(ll.detach().numpy(), n_dims, gaussian_std=std, mse_std=1)
    assert np.allclose(mse, mse_.detach().numpy())
