import numpy as np
import torch
from behavenet.fitting import losses

LN2PI = np.log(2 * np.pi)


def test_mse():
    raise NotImplementedError


def test_gaussian_ll():
    raise NotImplementedError


def test_kl_div_to_std_normal():

    mu = []
    logvar = []
    chunks = losses.kl_div_to_std_normal(mu, logvar)
    assert chunks == 1

    pass


def test_gaussian_ll_to_mse():
    raise NotImplementedError
