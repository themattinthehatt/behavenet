import h5py
import os
import numpy as np
import pytest
from behavenet.data import utils


def test_get_data_generator_inputs():

    hparams = {
        'data_dir': 'ddir', 'results_dir': 'rdir',
        'lab': 'lab0', 'expt': 'expt0', 'animal': 'animal0', 'session': 'session0'}
    session_dir = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'])
    hdf5_path = os.path.join(session_dir, 'data.hdf5')
    sess_ids = [
        {'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
         'session': hparams['session']}]

    # ae
    hparams['model_class'] = 'ae'
    hparams_, signals, transforms, paths = utils.get_data_generator_inputs(hparams, sess_ids)
    assert signals[0] == ['images']
    assert transforms[0] == [None]
    assert paths[0] == [hdf5_path]

    hparams['model_class'] = 'ae'
    hparams['use_output_mask'] = True
    hparams_, signals, transforms, paths = utils.get_data_generator_inputs(hparams, sess_ids)
    assert signals[0] == ['images', 'masks']
    assert transforms[0] == [None, None]
    assert paths[0] == [hdf5_path, hdf5_path]
    hparams['use_output_mask'] = False

    # cond-ae

    # ae_latents

    # neural-ae

    # ae-neural

    # neural-arhmm

    # arhmm-neural

    # arhmm

    # arhmm-labels

    # bayesian-decoding

    # labels-images

    # labels


def test_get_transforms_paths():

    hparams = {
        'data_dir': 'ddir', 'results_dir': 'rdir',
        'lab': 'lab', 'expt': 'expt', 'animal': 'animal', 'session': 'session'}
    session_dir = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'])
    hdf5_path = os.path.join(session_dir, 'data.hdf5')
    sess_id_str = str('%s_%s_%s_%s_' % (
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))

    # -----------
    # neural data
    # -----------

    # spikes, no thresholding
    hparams['neural_type'] = 'spikes'
    hparams['neural_thresh'] = 0
    transform, path = utils.get_transforms_paths('neural', hparams, sess_id=None)
    assert path == hdf5_path
    assert transform is None

    # spikes, thresholding
    hparams['neural_type'] = 'spikes'
    hparams['neural_thresh'] = 1
    hparams['neural_bin_size'] = 1
    transform, path = utils.get_transforms_paths('neural', hparams, sess_id=None)
    assert path == hdf5_path
    assert transform.__repr__().find('Threshold') > -1

    # calcium, no zscoring
    hparams['neural_type'] = 'ca'
    hparams['model_type'] = 'ae-neural'
    transform, path = utils.get_transforms_paths('neural', hparams, sess_id=None)
    assert path == hdf5_path
    assert transform is None

    # calcium, zscoring
    hparams['neural_type'] = 'ca'
    hparams['model_type'] = 'neural-ae'
    transform, path = utils.get_transforms_paths('neural', hparams, sess_id=None)
    assert path == hdf5_path
    assert transform.__repr__().find('ZScore') > -1

    # raise exception for incorrect neural type
    hparams['neural_type'] = 'wf'
    with pytest.raises(ValueError):
        utils.get_transforms_paths('neural', hparams, sess_id=None)

    # TODO: test subsampling methods

    # ----------
    # ae latents
    # ----------

    hparams['session_dir'] = session_dir
    hparams['ae_model_type'] = 'conv'
    hparams['n_ae_latents'] = 8
    hparams['ae_experiment_name'] = 'tt_expt_ae'
    hparams['ae_version'] = 0

    ae_path = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'], 'ae', hparams['ae_model_type'],
        '%02i_latents' % hparams['n_ae_latents'], hparams['ae_experiment_name'])

    # user-defined latent path
    hparams['ae_latents_file'] = 'path/to/latents'
    transform, path = utils.get_transforms_paths('ae_latents', hparams, sess_id=None)
    assert path == hparams['ae_latents_file']
    assert transform is None
    hparams.pop('ae_latents_file')

    # build pathname from hparams
    transform, path = utils.get_transforms_paths('ae_latents', hparams, sess_id=None)
    assert path == os.path.join(
        ae_path, 'version_%i' % hparams['ae_version'], '%slatents.pkl' % sess_id_str)
    assert transform is None

    # TODO: use get_best_model_version()

    # ------------
    # arhmm states
    # ------------

    hparams['n_ae_latents'] = 8
    hparams['n_arhmm_states'] = 2
    hparams['kappa'] = 0
    hparams['noise_type'] = 'gaussian'
    hparams['arhmm_experiment_name'] = 'tt_expt_arhmm'
    hparams['arhmm_version'] = 1

    arhmm_path = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'], 'arhmm', '%02i_latents' % hparams['n_ae_latents'],
        '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
        hparams['noise_type'], hparams['arhmm_experiment_name'])

    # user-defined state path
    hparams['arhmm_states_file'] = 'path/to/states'
    transform, path = utils.get_transforms_paths('arhmm_states', hparams, sess_id=None)
    assert path == hparams['arhmm_states_file']
    assert transform is None
    hparams.pop('arhmm_states_file')

    # build pathname from hparams
    transform, path = utils.get_transforms_paths('arhmm_states', hparams, sess_id=None)
    assert path == os.path.join(
        arhmm_path, 'version_%i' % hparams['arhmm_version'], '%sstates.pkl' % sess_id_str)
    assert transform is None

    # include shuffle transform
    hparams['shuffle_rng_seed'] = 0
    transform, path = utils.get_transforms_paths('arhmm_states', hparams, sess_id=None)
    assert path == os.path.join(
        arhmm_path, 'version_%i' % hparams['arhmm_version'], '%sstates.pkl' % sess_id_str)
    assert transform.__repr__().find('BlockShuffle') > -1

    # TODO: use get_best_model_version()

    # ---------------------
    # neural ae predictions
    # ---------------------

    hparams['n_ae_latents'] = 8
    hparams['neural_ae_model_type'] = 'linear'
    hparams['neural_ae_experiment_name'] = 'tt_expt_ae_decoder'
    hparams['neural_ae_version'] = 2

    ae_pred_path = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'], 'neural-ae', '%02i_latents' % hparams['n_ae_latents'],
        hparams['neural_ae_model_type'], 'all', hparams['neural_ae_experiment_name'])

    # user-defined predictions path
    hparams['ae_predictions_file'] = 'path/to/predictions'
    transform, path = utils.get_transforms_paths('neural_ae_predictions', hparams, sess_id=None)
    assert path == hparams['ae_predictions_file']
    assert transform is None
    hparams.pop('ae_predictions_file')

    # build pathname from hparams
    transform, path = utils.get_transforms_paths('neural_ae_predictions', hparams, sess_id=None)
    assert path == os.path.join(
        ae_pred_path, 'version_%i' % hparams['neural_ae_version'],
        '%spredictions.pkl' % sess_id_str)
    assert transform is None

    # TODO: use get_best_model_version()

    # ------------------------
    # neural arhmm predictions
    # ------------------------

    hparams['n_ae_latents'] = 8
    hparams['n_arhmm_states'] = 10
    hparams['kappa'] = 0
    hparams['noise_type'] = 'studentst'
    hparams['neural_arhmm_model_type'] = 'linear'
    hparams['neural_arhmm_experiment_name'] = 'tt_expt_ae_decoder'
    hparams['neural_arhmm_version'] = 3

    arhmm_pred_path = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'], 'neural-arhmm', '%02i_latents' % hparams['n_ae_latents'],
        '%02i_states' % hparams['n_arhmm_states'], '%.0e_kappa' % hparams['kappa'],
        hparams['neural_arhmm_model_type'], 'all', hparams['neural_arhmm_experiment_name'])

    # user-defined predictions path
    hparams['arhmm_predictions_file'] = 'path/to/predictions'
    transform, path = utils.get_transforms_paths('neural_arhmm_predictions', hparams, sess_id=None)
    assert path == hparams['arhmm_predictions_file']
    assert transform is None
    hparams.pop('arhmm_predictions_file')

    # build pathname from hparams
    transform, path = utils.get_transforms_paths('neural_arhmm_predictions', hparams, sess_id=None)
    assert path == os.path.join(
        arhmm_pred_path, 'version_%i' % hparams['neural_arhmm_version'],
        '%spredictions.pkl' % sess_id_str)
    assert transform is None

    # TODO: use get_best_model_version()

    # -----
    # other
    # -----

    with pytest.raises(ValueError):
        utils.get_transforms_paths('invalid', hparams, sess_id=None)


def test_load_labels_like_latents():
    # TODO
    pass


def test_get_region_list(tmpdir):

    # make tmp hdf5 file
    path = tmpdir.join('data.hdf5')
    idx_data = {
        'i0': np.array([0, 1, 2]),
        'i1': np.array([3, 4, 5]),
        'i2': np.array([6, 7, 8])}
    with h5py.File(path, 'w') as f:
        group0 = f.create_group('group0')
        groupa = f.create_group('groupa')
        group1 = group0.create_group('group1')
        group1.create_dataset('i0', data=idx_data['i0'])
        group1.create_dataset('i1', data=idx_data['i1'])
        group1.create_dataset('i2', data=idx_data['i2'])

    # correct indices are returned
    hparams = {
        'data_dir': tmpdir,
        'lab': '',
        'expt': '',
        'animal': '',
        'session': '',
        'subsample_idxs_group_0': 'group0',
        'subsample_idxs_group_1': 'group1'}
    idx_return = utils.get_region_list(hparams)
    for key in idx_data.keys():
        assert np.all(idx_data[key] == idx_return[key])

    # raise exception when first group is invalid
    hparams['subsample_idxs_group_0'] = 'group2'
    hparams['subsample_idxs_group_1'] = 'group1'
    with pytest.raises(ValueError):
        utils.get_region_list(hparams)

    # raise exception when first group contains no second group
    hparams['subsample_idxs_group_0'] = 'groupa'
    hparams['subsample_idxs_group_1'] = 'group1'
    with pytest.raises(ValueError):
        utils.get_region_list(hparams)

    # raise exception when second group is invalid
    hparams['subsample_idxs_group_0'] = 'group0'
    hparams['subsample_idxs_group_1'] = 'group2'
    with pytest.raises(ValueError):
        utils.get_region_list(hparams)
