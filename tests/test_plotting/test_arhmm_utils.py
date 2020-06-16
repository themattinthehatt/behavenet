import numpy as np
from behavenet.plotting import arhmm_utils


def test_get_discrete_chunks():

    states = [
        np.array([0, 1, 1, 1, 2, 2, 0]),
        np.array([3, 3, 3, 4, 4, 2, 2, 2])
    ]

    chunks = arhmm_utils.get_discrete_chunks(states, include_edges=True)
    assert np.all(chunks[0] == np.array([[0, 0, 1], [0, 6, 7]]))
    assert np.all(chunks[1] == np.array([[0, 1, 4]]))
    assert np.all(chunks[2] == np.array([[0, 4, 6], [1, 5, 8]]))
    assert np.all(chunks[3] == np.array([[1, 0, 3]]))
    assert np.all(chunks[4] == np.array([[1, 3, 5]]))

    chunks = arhmm_utils.get_discrete_chunks(states, include_edges=False)
    assert np.all(chunks[0] == np.array([]))
    assert np.all(chunks[1] == np.array([[0, 1, 4]]))
    assert np.all(chunks[2] == np.array([[0, 4, 6]]))
    assert np.all(chunks[3] == np.array([]))
    assert np.all(chunks[4] == np.array([[1, 3, 5]]))


def test_get_state_durations():

    # construct mock HMM class that passes argument through function `most_likely_states`
    class HMM(object):
        @classmethod
        def most_likely_states(cls, x):
            return x
    hmm = HMM()
    latents = [
        np.array([0, 1, 1, 1, 2, 2, 0]),
        np.array([3, 3, 3, 4, 4, 2, 2, 2]),
        np.array([0, 0, 0, 3, 3, 3, 1, 1, 2])
    ]

    durations = arhmm_utils.get_state_durations(latents, hmm, include_edges=True)
    assert np.all(durations[0] == np.array([1, 1, 3]))
    assert np.all(durations[1] == np.array([3, 2]))
    assert np.all(durations[2] == np.array([2, 3, 1]))
    assert np.all(durations[3] == np.array([3, 3]))

    durations = arhmm_utils.get_state_durations(latents, hmm, include_edges=False)
    assert np.all(durations[0] == np.array([]))
    assert np.all(durations[1] == np.array([3, 2]))
    assert np.all(durations[2] == np.array([2]))
    assert np.all(durations[3] == np.array([3]))
