import pytest
import numpy as np
import pickle
from behavenet.data.data_generator import split_trials, _load_pkl_dict


def test_split_trials():

    # return correct number of trials 1
    splits = split_trials(100, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 80
    assert len(splits['val']) == 10
    assert len(splits['test']) == 10

    # return correct number of trials 2
    splits = split_trials(10, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 8
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    # return correct number of trials 3
    splits = split_trials(11, 0, 8, 1, 1, 0)
    assert len(splits['train']) == 8
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    # raise exception when not enough trials 1
    with pytest.raises(ValueError):
        split_trials(6, 0, 8, 1, 1, 0)

    # raise exception when not enough trials 2
    with pytest.raises(ValueError):
        split_trials(11, 0, 8, 1, 1, 1)

    # properly insert gap trials
    splits = split_trials(13, 0, 8, 1, 1, 1)
    assert len(splits['train']) == 8
    assert len(splits['val']) == 1
    assert len(splits['test']) == 1

    max_train = np.max(splits['train'])
    assert not np.any(splits['val'] == max_train + 1)
    assert not np.any(splits['test'] == max_train + 1)

    max_val = np.max(splits['val'])
    assert not np.any(splits['test'] == max_val + 1)


def test_load_pkl_dict(tmpdir):

    # make tmp pickled dict file
    key = 'test'
    tmp_dict = {key: [np.random.randn(4, i) for i in [3, 5]]}
    path = tmpdir.join('test.pkl')
    with open(path, 'wb') as f:
        pickle.dump(tmp_dict, f)

    # return all data in list
    data1 = _load_pkl_dict(path, key, idx=None)
    assert len(data1) == 2
    assert data1[0].shape == (4, 3)
    assert data1[1].shape == (4, 5)

    # return selected index in list of length 1
    data2 = _load_pkl_dict(path, key, idx=0)
    assert len(data2) == 1
    assert data2[0].shape == (4, 3)

    # return selected index in list of length 1
    data3 = _load_pkl_dict(path, key, idx=1)
    assert len(data3) == 1
    assert data3[0].shape == (4, 5)
