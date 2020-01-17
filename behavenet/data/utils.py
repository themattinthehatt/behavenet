"""Utility functions for constructing inputs to data generators."""

import os
import numpy as np


def get_data_generator_inputs(hparams, sess_ids):
    """Helper function for generating signals, transforms and paths.

    Parameters
    ----------
    hparams : :obj:`dict`
        required keys: 'data_dir', 'lab', 'expt', 'animal', 'session', 'model_class', and model
        parameters associated with the 'model_class'; see :func:`get_transforms_paths` for these
        parameters
    sess_ids : :obj:`list` of :obj:`dict`
        each list entry is a session-specific dict with keys 'lab', 'expt', 'animal', 'session'

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
                'neural', hparams, sess_id=sess_id)
        else:
            neural_transform = None
            neural_path = None

        # get model-specific signals/transforms/paths
        if hparams['model_class'] == 'ae':

            signals = ['images']
            transforms = [None]
            paths = [os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'ae_latents':

            ae_transform, ae_path = get_transforms_paths('ae_latents', hparams, sess_id=sess_id)

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

            ae_transform, ae_path = get_transforms_paths('ae_latents', hparams, sess_id=sess_id)

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

            ae_transform, ae_path = get_transforms_paths('ae_latents', hparams, sess_id=sess_id)

            signals = ['neural', 'ae_latents']
            transforms = [neural_transform, ae_transform]
            paths = [neural_path, ae_path]

        elif hparams['model_class'] == 'neural-arhmm':

            hparams['input_signal'] = 'neural'
            hparams['output_signal'] = 'arhmm_states'
            hparams['output_size'] = hparams['n_arhmm_states']
            hparams['noise_dist'] = 'categorical'

            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id)

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
                'arhmm_states', hparams, sess_id=sess_id)

            signals = ['neural', 'arhmm_states']
            transforms = [neural_transform, arhmm_transform]
            paths = [neural_path, arhmm_path]

        elif hparams['model_class'] == 'arhmm' or hparams['model_class'] == 'hmm':

            ae_transform, ae_path = get_transforms_paths('ae_latents', hparams, sess_id=sess_id)

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
                'ae_latents', hparams, sess_id=sess_id)

            # get arhmm states info
            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id)

            # get neural-ae info
            neural_ae_transform, neural_ae_path = get_transforms_paths(
                'neural_ae_predictions', hparams, None)

            # get neural-arhmm info
            neural_arhmm_transform, neural_arhmm_path = get_transforms_paths(
                'neural_arhmm_predictions', hparams, None)

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

        else:
            raise ValueError('"%s" is an invalid model_class' % hparams['model_class'])

        signals_list.append(signals)
        transforms_list.append(transforms)
        paths_list.append(paths)

    return hparams, signals_list, transforms_list, paths_list


def get_transforms_paths(data_type, hparams, sess_id):
    """Helper function for generating session-specific transforms and paths.

    Parameters
    ----------
    data_type : :obj:`str`
        'neural' | 'ae_latents' | 'arhmm_states' | 'neural_ae_predictions' |
        'neural_arhmm_predictions'
    hparams : :obj:`dict`
        - required keys for :obj:`data_type=neural`: 'neural_type', 'neural_thresh'
        - required keys for :obj:`data_type=ae_latents`: 'ae_experiment_name', 'ae_model_type', 'n_ae_latents', 'ae_version' or 'ae_latents_file'; this last option defines either the specific ae version (as 'best' or an int) or a path to a specific ae latents pickle file.
        - required keys for :obj:`data_type=arhmm_states`: 'arhmm_experiment_name', 'n_arhmm_states', 'kappa', 'noise_type', 'n_ae_latents', 'arhmm_version' or 'arhmm_states_file'; this last option defines either the specific arhmm version (as 'best' or an int) or a path to a specific ae latents pickle file.
        - required keys for :obj:`data_type=neural_ae_predictions`: 'neural_ae_experiment_name', 'neural_ae_model_type', 'neural_ae_version' or 'ae_predictions_file' plus keys for neural and ae_latents data types.
        - required keys for :obj:`data_type=neural_arhmm_predictions`: 'neural_arhmm_experiment_name', 'neural_arhmm_model_type', 'neural_arhmm_version' or 'arhmm_predictions_file', plus keys for neural and arhmm_states data types.
    sess_id : :obj:`dict`
        each list entry is a session-specific dict with keys 'lab', 'expt', 'animal', 'session'

    Returns
    -------
    :obj:`tuple`
        - hparams (:obj:`dict`): updated with model-specific information like input and output size
        - signals (:obj:`list`): session-specific signals
        - transforms (:obj:`list`): session-specific transforms
        - paths (:obj:`list`): session-specific paths

    """

    from behavenet.data.transforms import SelectIdxs
    from behavenet.data.transforms import Threshold
    from behavenet.data.transforms import ZScore
    from behavenet.data.transforms import BlockShuffle
    from behavenet.data.transforms import Compose
    from behavenet.fitting.utils import get_best_model_version
    from behavenet.fitting.utils import get_expt_dir

    # check for multisession by comparing hparams and sess_id
    hparams_ = {key: hparams[key] for key in ['lab', 'expt', 'animal', 'session']}
    if sess_id is None:
        sess_id = hparams_

    sess_id_str = str('%s_%s_%s_%s_' % (
        sess_id['lab'], sess_id['expt'], sess_id['animal'], sess_id['session']))

    if data_type == 'neural':

        path = os.path.join(
            hparams['data_dir'], sess_id['lab'], sess_id['expt'], sess_id['animal'],
            sess_id['session'], 'data.hdf5')

        transforms_ = []

        # filter neural data by region
        if hparams.get('subsample_regions', 'none') != 'none':
            # get region and indices
            sampling = hparams['subsample_regions']
            region_name = hparams['region']
            regions = get_region_list(hparams)
            if sampling == 'single':
                idxs = regions[region_name]
            elif sampling == 'loo':
                idxs = []
                for reg_name, reg_idxs in regions.items():
                    if reg_name != region_name:
                        idxs.append(reg_idxs)
                idxs = np.concatenate(idxs)
            else:
                raise ValueError('"%s" is an invalid region sampling option' % sampling)
            transforms_.append(SelectIdxs(idxs, str('%s-%s' % (region_name, sampling))))

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
        else:
            raise ValueError('"%s" is an invalid neural type' % hparams['neural_type'])

        # compose filters
        if len(transforms_) == 0:
            transform = None
        else:
            transform = Compose(transforms_)

    elif data_type == 'ae_latents' or data_type == 'latents':
        ae_dir = get_expt_dir(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        transform = None

        if 'ae_latents_file' in hparams:
            path = hparams['ae_latents_file']
        else:
            if 'ae_version' in hparams and isinstance(hparams['ae_version'], int):
                ae_version = str('version_%i' % hparams['ae_version'])
            else:
                ae_version = 'version_%i' % get_best_model_version(ae_dir, 'val_loss')[0]
            ae_latents = str('%slatents.pkl' % sess_id_str)
            path = os.path.join(ae_dir, ae_version, ae_latents)

    elif data_type == 'arhmm_states' or data_type == 'states':

        arhmm_dir = get_expt_dir(
            hparams, model_class='arhmm', expt_name=hparams['arhmm_experiment_name'])

        if hparams.get('shuffle_rng_seed') is not None:
            transform = BlockShuffle(hparams['shuffle_rng_seed'])
        else:
            transform = None

        if 'arhmm_state_file' in hparams:
            path = hparams['arhmm_state_file']
        else:
            if 'arhmm_version' in hparams and isinstance(hparams['arhmm_version'], int):
                arhmm_version = str('version_%i' % hparams['arhmm_version'])
            else:
                arhmm_version = 'version_%i' % get_best_model_version(
                    arhmm_dir, 'val_loss', best_def='max')[0]
            arhmm_states = str('%sstates.pkl' % sess_id_str)
            path = os.path.join(arhmm_dir, arhmm_version, arhmm_states)

    elif data_type == 'neural_ae_predictions' or data_type == 'ae_predictions':

        neural_ae_dir = get_expt_dir(
            hparams, model_class='neural-ae',
            expt_name=hparams['neural_ae_experiment_name'],
            model_type=hparams['neural_ae_model_type'])

        transform = None
        if 'ae_predictions_file' in hparams:
            path = hparams['ae_predictions_file']
        else:
            if 'neural_ae_version' in hparams and isinstance(hparams['neural_ae_version'], int):
                neural_ae_version = str('version_%i' % hparams['neural_ae_version'])
            else:
                neural_ae_version = 'version_%i' % get_best_model_version(
                    neural_ae_dir, 'val_loss')[0]
            neural_ae_predictions = str('%spredictions.pkl' % sess_id_str)
            path = os.path.join(neural_ae_dir, neural_ae_version, neural_ae_predictions)

    elif data_type == 'neural_arhmm_predictions' or data_type == 'arhmm_predictions':

        neural_arhmm_dir = get_expt_dir(
            hparams, model_class='neural-arhmm',
            expt_name=hparams['neural_arhmm_experiment_name'],
            model_type=hparams['neural_arhmm_model_type'])

        transform = None
        if 'arhmm_predictions_file' in hparams:
            path = hparams['arhmm_predictions_file']
        else:
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

    return transform, path


def load_labels_like_latents(hparams, sess_ids, sess_idx):
    """Load labels from hdf5 in the same dictionary format that latents are saved.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain data generator params
    sess_ids : :obj:`list` of :obj:`dict`
        each entry is a session dict with keys 'lab', 'expt', 'animal', 'session'
    sess_idx : :obj:`int`
        session index into data generator

    Returns
    -------
    :obj:`dict`
        - latents (:obj:`list` of :obj:`np.ndarray`)
        - trials (:obj:`dict`) with keys `train`, `test`, and `val`

    """
    from behavenet.fitting.utils import build_data_generator

    hparams['as_numpy'] = True
    data_generator = build_data_generator(hparams, sess_ids, export_csv=False)
    dtypes = data_generator._dtypes

    labels = [np.array([]) for _ in range(data_generator.datasets[sess_idx].n_trials)]
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, sess = data_generator.next_batch(dtype)
            labels[data['batch_idx'].item()] = data['labels'][0][0]
    all_labels = {
        'latents': labels,  # name latents to match with old analysis code
        'trials': data_generator.datasets[sess_idx].batch_idxs}
    return all_labels


def get_region_list(hparams):
    """Get brain regions and their indices into neural data.

    Parameters
    ----------
    hparams : :obj:`dict` or :obj:`namespace` object
        required keys: 'data_dir', 'lab', 'expt', 'animal', 'session'

    Returns
    -------
    :obj:`dict`
        keys are brain regions defined in :obj:`data.hdf5` file

    """
    import h5py

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    data_file = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'],
        hparams['animal'], hparams['session'], 'data.hdf5')

    with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
        idx_types = list(f['regions'])
        if 'indxs_consolidate' in idx_types:
            regions = list(f['regions']['indxs_consolidate'].keys())
            idxs = {reg: np.ravel(f['regions']['indxs_consolidate'][reg][()]) for
                     reg in regions}
        elif 'indxs_consolidate_lr' in idx_types:
            regions = list(f['regions']['indxs_consolidate_lr'].keys())
            idxs = {reg: np.ravel(f['regions']['indxs_consolidate_lr'][reg][()])
                     for reg in regions}
        else:
            regions = list(f['regions']['indxs'])
            idxs = {reg: np.ravel(f['regions']['indxs'][reg][()])
                     for reg in regions}

    return idxs
