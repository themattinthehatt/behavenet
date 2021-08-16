"""Utility functions for constructing inputs to data generators."""

import os
import numpy as np
import pickle

from behavenet.fitting.utils import export_session_info_to_csv

# to ignore imports for sphinx-autoapidoc
__all__ = [
    'get_data_generator_inputs', 'build_data_generator', 'check_same_training_split',
    'get_transforms_paths', 'load_labels_like_latents', 'get_region_list']


def get_data_generator_inputs(hparams, sess_ids, check_splits=True):
    """Helper function for generating signals, transforms and paths.

    Parameters
    ----------
    hparams : :obj:`dict`
        required keys: 'data_dir', 'lab', 'expt', 'animal', 'session', 'model_class', and model
        parameters associated with the 'model_class'; see :func:`get_transforms_paths` for these
        parameters
    sess_ids : :obj:`list` of :obj:`dict`
        each list entry is a session-specific dict with keys 'lab', 'expt', 'animal', 'session'
    check_splits : :obj:`bool`, optional
        check data splits and data rng seed between hparams and loaded model outputs (e.g. latents)

    Returns
    -------
    :obj:`tuple`
        - hparams (:obj:`dict`): updated with model-specific information like input and output size
        - signals (:obj:`list`): session-specific signals
        - transforms (:obj:`list`): session-specific transforms
        - paths (:obj:`list`): session-specific paths

    """

    # TODO: add support for decoding HMM states
    # TODO: move input_signal/output_signal/etc to another function? (not needed for data gen)

    signals_list = []
    transforms_list = []
    paths_list = []

    for sess_id in sess_ids:

        data_dir = os.path.join(
            hparams['data_dir'], sess_id['lab'], sess_id['expt'],
            sess_id['animal'], sess_id['session'])

        # get neural signals/transforms/path
        if hparams['model_class'].find('neural') > -1:
            neural_transform, neural_path = get_transforms_paths(
                'neural', hparams, sess_id=sess_id, check_splits=check_splits)
        else:
            neural_transform = None
            neural_path = None

        # get model-specific signals/transforms/paths
        if hparams['model_class'] == 'ae' \
                or hparams['model_class'] == 'vae' \
                or hparams['model_class'] == 'beta-tcvae':

            signals = ['images']
            transforms = [None]
            paths = [os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'cond-ae' \
                or hparams['model_class'] == 'cond-ae-msp' \
                or hparams['model_class'] == 'cond-vae' \
                or hparams['model_class'] == 'ps-vae' \
                or hparams['model_class'] == 'msps-vae':

            signals = ['images', 'labels']
            transforms = [None, None]
            paths = [os.path.join(data_dir, 'data.hdf5'), os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            if hparams.get('use_label_mask', False) and (
                    hparams['model_class'] == 'cond-ae-msp'
                    or hparams['model_class'] == 'ps-vae'):
                signals.append('labels_masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            if hparams.get('conditional_encoder', False):
                from behavenet.data.transforms import MakeOneHot2D
                signals.append('labels_sc')
                transforms.append(MakeOneHot2D(hparams['y_pixels'], hparams['x_pixels']))
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'ae_latents':

            ae_transform, ae_path = get_transforms_paths(
                'ae_latents', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['ae_latents']
            transforms = [ae_transform]
            paths = [ae_path]

        elif hparams['model_class'] == 'neural-ae':

            hparams['input_signal'] = 'neural'
            hparams['output_signal'] = 'ae_latents'
            hparams['output_size'] = hparams['n_ae_latents']
            if hparams['model_type'][-2:] == 'mv':
                hparams['noise_dist'] = 'gaussian-full'
            else:
                hparams['noise_dist'] = 'gaussian'

            ae_transform, ae_path = get_transforms_paths(
                'ae_latents', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['neural', 'ae_latents']
            transforms = [neural_transform, ae_transform]
            paths = [neural_path, ae_path]

        elif hparams['model_class'] == 'neural-ae-me':

            hparams['input_signal'] = 'neural'
            hparams['output_signal'] = 'ae_latents'
            hparams['output_size'] = hparams['n_ae_latents']
            if hparams['model_type'][-2:] == 'mv':
                hparams['noise_dist'] = 'gaussian-full'
            else:
                hparams['noise_dist'] = 'gaussian'

            ae_transform, ae_path = get_transforms_paths(
                'ae_latents_me', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['neural', 'ae_latents']
            transforms = [neural_transform, ae_transform]
            paths = [neural_path, ae_path]

        elif hparams['model_class'] == 'ae-neural':

            hparams['input_signal'] = 'ae_latents'
            hparams['output_signal'] = 'neural'
            hparams['output_size'] = None  # to fill in after data is loaded
            if hparams['neural_type'] == 'ca':
                if hparams['model_type'][-2:] == 'mv':
                    hparams['noise_dist'] = 'gaussian-full'
                else:
                    hparams['noise_dist'] = 'gaussian'
            elif hparams['neural_type'] == 'spikes':
                hparams['noise_dist'] = 'poisson'

            ae_transform, ae_path = get_transforms_paths(
                'ae_latents', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['neural', 'ae_latents']
            transforms = [neural_transform, ae_transform]
            paths = [neural_path, ae_path]

        elif hparams['model_class'] == 'neural-labels':

            hparams['input_signal'] = 'neural'
            hparams['output_signal'] = 'labels'
            hparams['output_size'] = hparams['n_labels']
            if hparams['model_type'][-2:] == 'mv':
                hparams['noise_dist'] = 'gaussian-full'
            else:
                hparams['noise_dist'] = 'gaussian'

            signals = ['neural', 'labels']
            transforms = [neural_transform, None]
            paths = [neural_path, os.path.join(data_dir, 'data.hdf5')]

        elif hparams['model_class'] == 'labels-neural':

            hparams['input_signal'] = 'labels'
            hparams['output_signal'] = 'neural'
            hparams['output_size'] = None  # to fill in after data is loaded
            if hparams['neural_type'] == 'ca':
                if hparams['model_type'][-2:] == 'mv':
                    hparams['noise_dist'] = 'gaussian-full'
                else:
                    hparams['noise_dist'] = 'gaussian'
            elif hparams['neural_type'] == 'spikes':
                hparams['noise_dist'] = 'poisson'

            signals = ['neural', 'labels']
            transforms = [neural_transform, None]
            paths = [neural_path, os.path.join(data_dir, 'data.hdf5')]

        elif hparams['model_class'] == 'neural-arhmm':

            hparams['input_signal'] = 'neural'
            hparams['output_signal'] = 'arhmm_states'
            hparams['output_size'] = hparams['n_arhmm_states']
            hparams['noise_dist'] = 'categorical'

            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['neural', 'arhmm_states']
            transforms = [neural_transform, arhmm_transform]
            paths = [neural_path, arhmm_path]

        elif hparams['model_class'] == 'arhmm-neural':

            hparams['input_signal'] = 'arhmm_states'
            hparams['output_signal'] = 'neural'
            hparams['output_size'] = None  # to fill in after data is loaded
            if hparams['neural_type'] == 'ca':
                if hparams['model_type'][-2:] == 'mv':
                    hparams['noise_dist'] = 'gaussian-full'
                else:
                    hparams['noise_dist'] = 'gaussian'
            elif hparams['neural_type'] == 'spikes':
                hparams['noise_dist'] = 'poisson'

            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['neural', 'arhmm_states']
            transforms = [neural_transform, arhmm_transform]
            paths = [neural_path, arhmm_path]

        elif hparams['model_class'] == 'arhmm' or hparams['model_class'] == 'hmm':

            ae_transform, ae_path = get_transforms_paths(
                'ae_latents', hparams, sess_id=sess_id, check_splits=check_splits)

            signals = ['ae_latents']
            transforms = [ae_transform]
            paths = [ae_path]
            if hparams.get('load_videos', False):
                signals.append('images')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'arhmm-labels' or hparams['model_class'] == 'hmm-labels':

            signals = ['labels']
            transforms = [None]
            paths = [os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('load_videos', False):
                signals.append('images')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'bayesian-decoding':

            # get autoencoder latents info
            ae_transform, ae_path = get_transforms_paths(
                'ae_latents', hparams, sess_id=sess_id, check_splits=check_splits)

            # get arhmm states info
            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id, check_splits=check_splits)

            # get neural-ae info
            neural_ae_transform, neural_ae_path = get_transforms_paths(
                'neural_ae_predictions', hparams, None, check_splits=check_splits)

            # get neural-arhmm info
            neural_arhmm_transform, neural_arhmm_path = get_transforms_paths(
                'neural_arhmm_predictions', hparams, None, check_splits=check_splits)

            # put it all together
            signals = [
                'ae_latents',
                'ae_predictions',
                'arhmm_predictions',
                'arhmm_states']
            transforms = [
                ae_transform,
                neural_ae_transform,
                neural_arhmm_transform,
                arhmm_transform]
            paths = [
                ae_path,
                neural_ae_path,
                neural_arhmm_path,
                arhmm_path]
            if hparams.get('load_videos', False):
                signals.append('images')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'labels-images':

            hparams['input_signal'] = 'labels'
            hparams['output_signal'] = 'images'

            signals = ['images', 'labels']
            transforms = [None, None]
            paths = [os.path.join(data_dir, 'data.hdf5'), os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))
            # signals.append('labels_masks')
            # transforms.append(None)
            # paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'labels' or hparams['model_class'] == 'labels_sc':

            signals = [hparams['model_class']]
            transforms = [None]
            paths = [os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_label_mask', False):
                signals.append('labels_masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'labels_masks':

            signals = [hparams['model_class']]
            transforms = [None]
            paths = [os.path.join(data_dir, 'data.hdf5')]

        else:
            raise ValueError('"%s" is an invalid model_class' % hparams['model_class'])

        signals_list.append(signals)
        transforms_list.append(transforms)
        paths_list.append(paths)

    return hparams, signals_list, transforms_list, paths_list


def build_data_generator(hparams, sess_ids, export_csv=True):
    """Helper function to build data generator from hparams dict.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain information specifying data inputs to model
    sess_ids : :obj:`list` of :obj:`dict`
        each entry is a session dict with keys 'lab', 'expt', 'animal', 'session'
    export_csv : :obj:`bool`, optional
        export csv file containing session info (useful when fitting multi-sessions)

    Returns
    -------
    :obj:`ConcatSessionsGenerator` object
        data generator

    """
    from behavenet.data.data_generator import ConcatSessionsGenerator, ConcatSessionsGeneratorMulti
    print('using data from following sessions:')
    for ids in sess_ids:
        print('%s' % os.path.join(
            hparams['save_dir'], ids['lab'], ids['expt'], ids['animal'], ids['session']))
    hparams, signals, transforms, paths = get_data_generator_inputs(hparams, sess_ids)
    if hparams.get('trial_splits', None) is not None:
        # assumes string of form 'train;val;test;gap'
        trs = [int(tr) for tr in hparams['trial_splits'].split(';')]
        trial_splits = {'train_tr': trs[0], 'val_tr': trs[1], 'test_tr': trs[2], 'gap_tr': trs[3]}
    else:
        trial_splits = None
    print('constructing data generator...', end='')
    if hparams.get('n_sessions_per_batch', 1) == 1:
        data_generator = ConcatSessionsGenerator(
            hparams['data_dir'], sess_ids,
            signals_list=signals, transforms_list=transforms, paths_list=paths,
            device=hparams['device'], as_numpy=hparams['as_numpy'],
            batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed_data'],
            trial_splits=trial_splits, train_frac=hparams['train_frac'])
    else:
        data_generator = ConcatSessionsGeneratorMulti(
            hparams['data_dir'], sess_ids,
            signals_list=signals, transforms_list=transforms, paths_list=paths,
            device=hparams['device'], as_numpy=hparams['as_numpy'],
            batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed_data'],
            trial_splits=trial_splits, train_frac=hparams['train_frac'],
            n_sessions_per_batch=hparams['n_sessions_per_batch'])
    # csv order will reflect dataset order in data generator
    if export_csv:
        export_session_info_to_csv(os.path.join(
            hparams['expt_dir'], str('version_%i' % hparams['version'])), sess_ids)
    print('done')
    print(data_generator)
    return data_generator


def check_same_training_split(model_path, hparams):
    """Ensure data rng seed and trial splits are same for two models."""

    import_params_file = os.path.join(os.path.dirname(model_path), 'meta_tags.pkl')
    import_params = pickle.load(open(import_params_file, 'rb'))

    if import_params['rng_seed_data'] != hparams['rng_seed_data'] and \
            hparams.get('check_rng_seed_data', True):
        raise ValueError('Different data random seed from existing models')

    if import_params['trial_splits'] != hparams['trial_splits'] and \
            hparams.get('check_trial_splits', True):
        raise ValueError('Different trial split from existing models')


def get_transforms_paths(data_type, hparams, sess_id, check_splits=True):
    """Helper function for generating session-specific transforms and paths.

    Parameters
    ----------
    data_type : :obj:`str`
        'neural' | 'ae_latents' | 'arhmm_states' | 'neural_ae_predictions' |
        'neural_arhmm_predictions'
    hparams : :obj:`dict`
        - required keys for :obj:`data_type=neural`: 'neural_type', 'neural_thresh'
        - required keys for :obj:`data_type=ae_latents`: 'ae_experiment_name', 'ae_model_type',
          'n_ae_latents', 'ae_version' or 'ae_latents_file'; this last option defines either the
          specific ae version (as 'best' or an int) or a path to a specific ae latents pickle file.
        - required keys for :obj:`data_type=arhmm_states`: 'arhmm_experiment_name',
          'n_arhmm_states', 'kappa', 'noise_type', 'n_ae_latents', 'arhmm_version' or
          'arhmm_states_file'; this last option defines either the specific arhmm version (as
          'best' or an int) or a path to a specific ae latents pickle file.
        - required keys for :obj:`data_type=neural_ae_predictions`: 'neural_ae_experiment_name',
          'neural_ae_model_type', 'neural_ae_version' or 'ae_predictions_file' plus keys for neural
          and ae_latents data types.
        - required keys for :obj:`data_type=neural_arhmm_predictions`:
          'neural_arhmm_experiment_name', 'neural_arhmm_model_type', 'neural_arhmm_version' or
          'arhmm_predictions_file', plus keys for neural and arhmm_states data types.
    sess_id : :obj:`dict`
        each list entry is a session-specific dict with keys 'lab', 'expt', 'animal', 'session'
    check_splits : :obj:`bool`, optional
        check data splits and data rng seed between hparams and loaded model outputs (e.g. latents)

    Returns
    -------
    :obj:`tuple`
        - transform (:obj:`behavenet.data.transforms.Transform` object): session-specific transform
        - path (:obj:`str`): session-specific path

    """

    from behavenet.data.transforms import BlockShuffle
    from behavenet.data.transforms import Compose
    from behavenet.data.transforms import MotionEnergy
    from behavenet.data.transforms import SelectIdxs
    from behavenet.data.transforms import Threshold
    from behavenet.data.transforms import ZScore
    from behavenet.fitting.utils import get_best_model_version
    from behavenet.fitting.utils import get_expt_dir

    # check for multisession by comparing hparams and sess_id
    hparams_ = {key: hparams[key] for key in ['lab', 'expt', 'animal', 'session']}
    if sess_id is None:
        sess_id = hparams_

    sess_id_str = str('%s_%s_%s_%s_' % (
        sess_id['lab'], sess_id['expt'], sess_id['animal'], sess_id['session']))

    if data_type == 'neural':

        check_splits = False

        path = os.path.join(
            hparams['data_dir'], sess_id['lab'], sess_id['expt'], sess_id['animal'],
            sess_id['session'], 'data.hdf5')

        transforms_ = []

        # filter neural data by indices (regions, cell types, etc)
        if hparams.get('subsample_method', 'none') != 'none':
            # get indices
            sampling = hparams['subsample_method']
            idxs_name = hparams['subsample_idxs_name']
            idxs_dict = get_region_list(hparams)
            if sampling == 'single':
                idxs = idxs_dict[idxs_name]
            elif sampling == 'loo':
                idxs = []
                for idxs_key, idxs_val in idxs_dict.items():
                    if idxs_key != idxs_name:
                        idxs.append(idxs_val)
                idxs = np.concatenate(idxs)
            else:
                raise ValueError('"%s" is an invalid index sampling option' % sampling)
            transforms_.append(SelectIdxs(idxs, str('%s-%s' % (idxs_name, sampling))))

        # filter neural data by activity
        if hparams['neural_type'] == 'spikes':
            if hparams['neural_thresh'] > 0:
                transforms_.append(Threshold(
                    threshold=hparams['neural_thresh'],
                    bin_size=hparams['neural_bin_size']))
        elif hparams['neural_type'] == 'ca':
            if hparams['model_type'][-6:] != 'neural':
                # don't zscore if predicting calcium activity
                transforms_.append(ZScore())
        elif hparams['neural_type'] == 'ca-zscored':
            pass
        else:
            raise ValueError('"%s" is an invalid neural type' % hparams['neural_type'])

        # compose filters
        if len(transforms_) == 0:
            transform = None
        else:
            transform = Compose(transforms_)

    elif data_type == 'ae_latents' or data_type == 'latents' \
            or data_type == 'ae_latents_me' or data_type == 'latents_me':

        if data_type == 'ae_latents_me' or data_type == 'latents_me':
            transform = MotionEnergy()
        else:
            transform = None

        if 'ae_latents_file' in hparams:
            path = hparams['ae_latents_file']
        else:
            ae_dir = get_expt_dir(
                hparams, model_class=hparams['ae_model_class'],
                expt_name=hparams['ae_experiment_name'],
                model_type=hparams['ae_model_type'])
            if 'ae_version' in hparams and hparams['ae_version'] != 'best':
                # json args read as strings
                if isinstance(hparams['ae_version'], str):
                    hparams['ae_version'] = int(hparams['ae_version'])
                ae_version = str('version_%i' % hparams['ae_version'])
            else:
                ae_version = 'version_%i' % get_best_model_version(ae_dir, 'val_loss')[0]
            ae_latents = str('%slatents.pkl' % sess_id_str)
            path = os.path.join(ae_dir, ae_version, ae_latents)

    elif data_type == 'arhmm_states' or data_type == 'states':

        if hparams.get('shuffle_rng_seed') is not None:
            transform = BlockShuffle(hparams['shuffle_rng_seed'])
        else:
            transform = None

        if 'arhmm_states_file' in hparams:
            path = hparams['arhmm_states_file']
        else:
            arhmm_dir = get_expt_dir(
                hparams, model_class='arhmm', expt_name=hparams['arhmm_experiment_name'])
            if 'arhmm_version' in hparams and isinstance(hparams['arhmm_version'], int):
                arhmm_version = str('version_%i' % hparams['arhmm_version'])
            else:
                arhmm_version = 'version_%i' % get_best_model_version(
                    arhmm_dir, 'val_loss', best_def='min')[0]
            arhmm_states = str('%sstates.pkl' % sess_id_str)
            path = os.path.join(arhmm_dir, arhmm_version, arhmm_states)

    elif data_type == 'neural_ae_predictions' or data_type == 'ae_predictions':

        transform = None

        if 'ae_predictions_file' in hparams:
            path = hparams['ae_predictions_file']
        else:
            neural_ae_dir = get_expt_dir(
                hparams, model_class='neural-ae',
                expt_name=hparams['neural_ae_experiment_name'],
                model_type=hparams['neural_ae_model_type'])
            if 'neural_ae_version' in hparams and isinstance(hparams['neural_ae_version'], int):
                neural_ae_version = str('version_%i' % hparams['neural_ae_version'])
            else:
                neural_ae_version = 'version_%i' % get_best_model_version(
                    neural_ae_dir, 'val_loss')[0]
            neural_ae_predictions = str('%spredictions.pkl' % sess_id_str)
            path = os.path.join(neural_ae_dir, neural_ae_version, neural_ae_predictions)

    elif data_type == 'neural_arhmm_predictions' or data_type == 'arhmm_predictions':

        transform = None

        if 'arhmm_predictions_file' in hparams:
            path = hparams['arhmm_predictions_file']
        else:
            neural_arhmm_dir = get_expt_dir(
                hparams, model_class='neural-arhmm',
                expt_name=hparams['neural_arhmm_experiment_name'],
                model_type=hparams['neural_arhmm_model_type'])
            if 'neural_arhmm_version' in hparams and \
                    isinstance(hparams['neural_arhmm_version'], int):
                neural_arhmm_version = str('version_%i' % hparams['neural_arhmm_version'])
            else:
                neural_arhmm_version = 'version_%i' % get_best_model_version(
                    neural_arhmm_dir, 'val_loss')[0]
            neural_arhmm_predictions = str('%spredictions.pkl' % sess_id_str)
            path = os.path.join(neural_arhmm_dir, neural_arhmm_version, neural_arhmm_predictions)

    else:
        raise ValueError('"%s" is an invalid data_type' % data_type)

    # check training data split is the same
    if check_splits:
        check_same_training_split(path, hparams)

    return transform, path


def load_labels_like_latents(hparams, sess_ids, sess_idx, data_key='labels'):
    """Load labels from hdf5 in the same dictionary format that latents are saved.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain data generator params
    sess_ids : :obj:`list` of :obj:`dict`
        each entry is a session dict with keys 'lab', 'expt', 'animal', 'session'
    sess_idx : :obj:`int`
        session index into data generator
    data_key : :obj:`str`, optional
        key to index hdf5 file (name of hdf5 group)

    Returns
    -------
    :obj:`dict`
        - latents (:obj:`list` of :obj:`np.ndarray`)
        - trials (:obj:`dict`) with keys `train`, `test`, and `val`

    """
    import copy

    hparams_new = copy.deepcopy(hparams)
    hparams_new['model_class'] = data_key
    hparams_new['device'] = 'cpu'
    hparams_new['as_numpy'] = True
    hparams_new['batch_load'] = False
    hparams_new['n_sessions_per_batch'] = 1

    data_generator = build_data_generator(hparams_new, sess_ids, export_csv=False)
    dtypes = data_generator._dtypes

    labels = [np.array([]) for _ in range(data_generator.datasets[sess_idx].n_trials)]
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, sess = data_generator.next_batch(dtype)
            if sess == sess_idx:
                labels[data['batch_idx'].item()] = data[data_key][0][0]
    all_labels = {
        'latents': labels,  # name latents to match with old analysis code
        'trials': data_generator.datasets[sess_idx].batch_idxs}
    return all_labels


def get_region_list(hparams, group_0='regions', group_1='indxs'):
    """Get brain regions and their indices into neural data.

    Parameters
    ----------
    hparams : :obj:`dict` or :obj:`namespace` object
        required keys: 'data_dir', 'lab', 'expt', 'animal', 'session'
    group_0 : :obj:`str`, optional
        top-level HDF5 group that contains second-level groups of neural indices
    group_1 : :obj:`str`, optional
        second-level HDF5 group that contains datasets of neural indices

    Returns
    -------
    :obj:`dict`
        keys are groups of indices defined in :obj:`data.hdf5` file (for example all indices
        associated with a single brain region)

    """
    import h5py

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    group_0 = hparams.get('subsample_idxs_group_0', group_0)
    group_1 = hparams.get('subsample_idxs_group_1', group_1)

    data_file = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'], hparams['animal'],
        hparams['session'], 'data.hdf5')

    with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:

        hdf5_groups_0 = list(f)
        if group_0 not in hdf5_groups_0:
            raise ValueError('"{}" is not a group in {}; must choose from {}'.format(
                group_0, data_file, hdf5_groups_0))

        hdf5_groups_1 = list(f[group_0])
        if len(hdf5_groups_1) == 0:
            raise ValueError('No index groups found in "%s" group of %s' % (group_0, data_file))
        if group_1 not in hdf5_groups_1:
            raise ValueError('"{}" is not a group in {} group; must choose from {}'.format(
                group_1, group_0, hdf5_groups_1))

        idx_keys = list(f[group_0][group_1])
        idxs = {idx: np.ravel(f[group_0][group_1][idx][()]) for idx in idx_keys}

    return idxs
