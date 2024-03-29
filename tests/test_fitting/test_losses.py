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


def test_kl_div_to_std_normal():

    mu = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)
    kl = losses.kl_div_to_std_normal(mu, logvar)
    assert kl == 0


def test_index_code_mi():
    pass


def test_total_correlation():
    pass


def test_dimension_wise_kl_to_std_normal():
    pass


def test_decomposed_kl():

    n_batch = 5
    n_dims = 3
    z = torch.rand(n_batch, n_dims)
    mu = torch.rand(n_batch, n_dims)
    logvar = torch.rand(n_batch, n_dims)

    # compute terms individually
    ic1 = losses.index_code_mi(z, mu, logvar)
    tc1 = losses.total_correlation(z, mu, logvar)
    dw1 = losses.dimension_wise_kl_to_std_normal(z, mu, logvar)

    # compute terms together
    ic2, tc2, dw2 = losses.decomposed_kl(z, mu, logvar)

    assert ic1.item() == ic2.item()
    assert tc1.item() == tc2.item()
    assert dw1.item() == dw2.item()


def test_subspace_overlap():

    # A = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
    # B = torch.tensor([[0, 0, 1]]).float()
    # overlap = losses.subspace_overlap(A, B)
    # assert overlap == 0
    #
    # from scipy.linalg import null_space
    # M = np.random.randn(10, 15)  # M shape: [a, b]
    # N = null_space(M)  # N shape: [b, b - a]
    # overlap = losses.subspace_overlap(torch.from_numpy(M), torch.from_numpy(N.T))
    # assert np.isclose(overlap, 0)
    #
    # k = 10
    # M = torch.from_numpy(np.eye(k)).float()
    # overlap = losses.subspace_overlap(M, M)
    # assert overlap == 1 / k

    A = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
    B = torch.tensor([[0, 0, 1]]).float()
    overlap = losses.subspace_overlap(A, B)
    assert overlap == 0

    from scipy.linalg import null_space, orth
    M = orth(np.random.randn(10, 15))  # M shape: [a, b]
    N = null_space(M)  # N shape: [b, b - a]
    overlap = losses.subspace_overlap(torch.from_numpy(M), torch.from_numpy(N.T))
    assert np.isclose(overlap, 0)

    k = 10
    M = torch.from_numpy(np.eye(k)).float()
    overlap = losses.subspace_overlap(M, M)
    assert overlap == 2 * k / ((2 * k) ** 2)


def test_triplet_loss():

    from torch.nn import TripletMarginLoss
    tl = TripletMarginLoss(margin=1.0, p=2)

    # test with 2 datasets
    n_batch = 6
    n_dims = 3
    x = torch.zeros((n_batch, n_dims))
    y = torch.ones((n_batch, n_dims))
    datasets = np.concatenate([np.zeros((n_batch,)), np.ones((n_batch,))])
    loss = losses.triplet_loss(tl, torch.cat([x, y], 0), datasets)
    assert np.isclose(loss.item(), 0, atol=1e-5)

    x = torch.zeros((n_batch, n_dims))
    y = 2 * torch.ones((n_batch, n_dims))
    datasets = np.concatenate([np.zeros((n_batch,)), np.ones((n_batch,))])
    loss = losses.triplet_loss(tl, torch.cat([x, y], 0), datasets)
    assert np.isclose(loss.item(), 0, atol=1e-5)

    t1 = 0.50
    x = torch.zeros((n_batch, n_dims))
    y = t1 * torch.ones((n_batch, n_dims))
    datasets = np.concatenate([np.zeros((n_batch,)), np.ones((n_batch,))])
    loss = losses.triplet_loss(tl, torch.cat([x, y], 0), datasets)
    val = (-np.sqrt(n_dims * t1 ** 2) + 1) * 2 / 3
    assert np.isclose(loss.item(), val, atol=1e-5)

    # test with 3 datasets
    t1 = 0.25
    t2 = 0.50
    x = torch.zeros((n_batch, n_dims))
    y = t1 * torch.ones((n_batch, n_dims))
    z = t2 * torch.ones((n_batch, n_dims))
    datasets = np.concatenate([np.zeros((n_batch,)), np.ones((n_batch,)), 2 * np.ones((n_batch,))])
    loss = losses.triplet_loss(tl, torch.cat([x, y, z], 0), datasets)
    val1 = (-np.sqrt(n_dims * t1 ** 2) + 1)
    val2 = (-np.sqrt(n_dims * t2 ** 2) + 1)
    val = (4 * val1 + 2 * val2) / 6
    assert np.isclose(loss.item(), val, atol=1e-5)

    # test with 4 datasets
    n_batch = 9
    t1 = 0.1
    t2 = 0.2
    t3 = 0.3
    x = torch.zeros((n_batch, n_dims))
    y = t1 * torch.ones((n_batch, n_dims))
    z = t2 * torch.ones((n_batch, n_dims))
    v = t3 * torch.ones((n_batch, n_dims))
    datasets = np.concatenate(
        [np.zeros((n_batch,)), np.ones((n_batch,)), 2 * np.ones((n_batch,)),
         3 * np.ones((n_batch,))])
    loss = losses.triplet_loss(tl, torch.cat([x, y, z, v], 0), datasets)
    val1 = (-np.sqrt(n_dims * t1 ** 2) + 1)
    val2 = (-np.sqrt(n_dims * t2 ** 2) + 1)
    val3 = (-np.sqrt(n_dims * t3 ** 2) + 1)
    val = (6 * val1 + 4 * val2 + 2 * val3) / 12
    print(val)
    print(loss.item())
    assert np.isclose(loss.item(), val, atol=1e-5)
