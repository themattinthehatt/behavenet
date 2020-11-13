import pytest
import numpy as np
from behavenet.data import transforms


def test_compose():

    # select indices -> zscore
    idxs = np.array([0, 3])
    t = transforms.Compose([transforms.SelectIdxs(idxs), transforms.ZScore()])

    signal = np.random.randn(100, 4)
    s = t(signal)
    assert s.shape == (100, 2)
    assert np.allclose(np.mean(s, axis=0), [0, 0], atol=1e-3)
    assert np.allclose(np.std(s, axis=0), [1, 1], atol=1e-3)


def test_blockshuffle():

    def get_runs(sample):

        vals = np.unique(sample)
        n_time = len(sample)

        # mark first time point of state change with a nonzero number
        change = np.where(np.concatenate([[0], np.diff(sample)], axis=0) != 0)[0]
        # collect runs
        runs = {val: [] for val in vals}
        prev_beg = 0
        for curr_beg in change:
            runs[sample[prev_beg]].append(curr_beg - prev_beg)
            prev_beg = curr_beg
        runs[sample[-1]].append(n_time - prev_beg)
        return runs

    t = transforms.BlockShuffle(0)

    # signal has changed
    signal = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1])
    s = t(signal)
    assert not np.all(signal == s)

    # frequency of values unchanged
    n_ex_og = np.array([len(np.argwhere(signal == i)) for i in range(3)])
    n_ex_sh = np.array([len(np.argwhere(s == i)) for i in range(3)])
    assert np.all(n_ex_og == n_ex_sh)

    # distribution of runs unchanged
    runs_og = get_runs(signal)
    runs_sh = get_runs(s)
    for key in runs_og.keys():
        assert np.all(np.sort(np.array(runs_og[key])) == np.sort(np.array(runs_sh[key])))


def test_clipnormalize():

    # raise exception when clip value <= 0
    with pytest.raises(ValueError):
        transforms.ClipNormalize(0)

    # raise exception when clip value <= 0
    with pytest.raises(ValueError):
        transforms.ClipNormalize(-2.3)

    t = transforms.ClipNormalize(2)
    signal = np.random.randn(3, 3)
    s = t(signal)
    assert np.max(s) <= 1
    signal[0, 0] = 3
    s = t(signal)
    assert np.max(s) == 1


def test_makeonehot():

    t = transforms.MakeOneHot()

    # pass one hot array without modification
    signal = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    s = t(signal)
    assert np.all(signal == s)

    # correct one-hotting
    signal = np.array([3, 3, 2, 2, 0])
    s = t(signal)
    assert np.all(
        s == np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]]))


def test_makeonehot2d():

    t = transforms.MakeOneHot2D(4, 4)

    # correct one-hotting with int signal
    signal = np.array([[1, 2, 0, 3], [0, 2, 1, 1], [3, 0, 1, 2]])
    sp = np.zeros((3, 2, 4, 4))
    sp[0, 0, 0, 1] = 1
    sp[0, 1, 3, 2] = 1
    sp[1, 0, 1, 0] = 1
    sp[1, 1, 1, 2] = 1
    sp[2, 0, 1, 3] = 1
    sp[2, 1, 2, 0] = 1
    s = t(signal)
    assert np.all(s == sp)

    # correct one-hotting with float signal
    signal = np.array([[1.2, 2.1, 0.1, 2.9], [0.2, 1.7, 1.1, 0.9], [3.2, 0.4, 1.3, 1.6]])
    s = t(signal)
    assert np.all(s == sp)

    # correct clipping at boundaries
    t = transforms.MakeOneHot2D(3, 3)
    signal = np.array([[1, 2, 0, 3], [-1, 2, 1, 1], [3, -2, 1, 4]])
    sp = np.zeros((3, 2, 3, 3))
    sp[0, 0, 0, 1] = 1
    sp[0, 1, 2, 2] = 1
    sp[1, 0, 1, 0] = 1
    sp[1, 1, 1, 2] = 1
    sp[2, 0, 1, 2] = 1
    sp[2, 1, 2, 0] = 1
    s = t(signal)
    assert np.all(s == sp)

    # correct one-hotting with nans in signal
    t = transforms.MakeOneHot2D(4, 4)
    signal = np.array([[1, 2, 0, np.nan], [0, 2, 1, 1], [3, 0, np.nan, 2]])
    sp = np.zeros((3, 2, 4, 4))
    sp[0, 0, 0, 1] = 1
    sp[0, 1, 0, 2] = 1
    sp[1, 0, 1, 0] = 1
    sp[1, 1, 1, 2] = 1
    sp[2, 0, 0, 3] = 1
    sp[2, 1, 2, 0] = 1
    s = t(signal)
    assert np.all(s == sp)


def test_motionenergy():

    T = 100
    D = 4
    t = transforms.MotionEnergy()
    signal = np.random.randn(T, D)
    s = t(signal)
    me = np.vstack([np.zeros((1, signal.shape[1])), np.abs(np.diff(signal, axis=0))])
    assert s.shape == (T, D)
    assert np.allclose(s, me, atol=1e-3)
    assert np.all(me >= 0)


def test_selectindxs():

    idxs = np.array([0, 3])
    t = transforms.SelectIdxs(idxs)

    signal = np.random.randn(5, 4)
    s = t(signal)
    assert s.shape == (5, 2)
    assert np.all(signal[:, idxs] == s)


def test_threshold():

    # raise exception when bin size <= 0
    with pytest.raises(ValueError):
        transforms.Threshold(1, 0)

    # raise exception when threshold < 0
    with pytest.raises(ValueError):
        transforms.Threshold(-1, 1)

    # no thresholding with 0 threshold
    t = transforms.Threshold(0, 1)
    signal = np.random.uniform(0, 4, (5, 4))
    s = t(signal)
    assert s.shape == (5, 4)

    # correct thresholding
    t = transforms.Threshold(1, 1e3)
    signal = np.random.uniform(2, 4, (5, 4))
    signal[:, 0] = 0
    s = t(signal)
    assert s.shape == (5, 3)


def test_zscore():

    t = transforms.ZScore()
    signal = 10 + 0.3 * np.random.randn(100, 3)
    s = t(signal)
    assert s.shape == (100, 3)
    assert np.allclose(np.mean(s, axis=0), [0, 0, 0], atol=1e-3)
    assert np.allclose(np.std(s, axis=0), [1, 1, 1], atol=1e-3)
