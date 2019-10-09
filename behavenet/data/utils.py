import os
import numpy as np
import pickle
import torch
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_output_session_dir
from behavenet.fitting.utils import get_best_model_version


def get_data_generator_inputs(hparams, sess_ids):
    """
    Helper function for generating signals, transforms and paths for
    common models

    Args:
        hparams (dict):
            - model_class
        sess_ids (list of dicts)
    """

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
            hparams['output_signal'] = 'arhmm'
            hparams['output_size'] = hparams['n_arhmm_states']
            hparams['noise_dist'] = 'categorical'

            arhmm_transform, arhmm_path = get_transforms_paths(
                'arhmm_states', hparams, sess_id=sess_id)

            signals = ['neural', 'arhmm']
            transforms = [neural_transform, arhmm_transform]
            paths = [neural_path, arhmm_path]

        elif hparams['model_class'] == 'arhmm-neural':

            hparams['input_signal'] = 'arhmm'
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

            signals = ['neural', 'arhmm']
            transforms = [neural_transform, arhmm_transform]
            paths = [neural_path, arhmm_path]

        elif hparams['model_class'] == 'arhmm':

            ae_transform, ae_path = get_transforms_paths('ae_latents', hparams, sess_id=sess_id)

            signals = ['ae_latents', 'images']
            transforms = [ae_transform, None]
            paths = [ae_path, os.path.join(data_dir, 'data.hdf5')]
            if hparams.get('use_output_mask', False):
                signals.append('masks')
                transforms.append(None)
                paths.append(os.path.join(data_dir, 'data.hdf5'))

        elif hparams['model_class'] == 'arhmm-decoding':

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
                'ae',
                'images',
                'ae_predictions',
                'arhmm_predictions',
                'arhmm']
            transforms = [
                ae_transform,
                None,
                neural_ae_transform,
                neural_arhmm_transform,
                arhmm_transform]
            paths = [
                ae_path,
                os.path.join(data_dir, 'data.hdf5'),
                neural_ae_path,
                neural_arhmm_path,
                arhmm_path]
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

    from behavenet.data.transforms import SelectIndxs
    from behavenet.data.transforms import Threshold
    from behavenet.data.transforms import ZScore
    from behavenet.data.transforms import BlockShuffle
    from behavenet.data.transforms import Compose

    # check for multisession by comparing hparams and sess_id
    hparams_ = {key: hparams[key] for key in ['lab', 'expt', 'animal', 'session']}
    if sess_id is None:
        sess_id = hparams_

    if hparams_ != sess_id:
        sess_id_str = str('%s_%s_%s_%s_' % (
            sess_id['lab'], sess_id['expt'], sess_id['animal'], sess_id['session']))
    else:
        sess_id_str = ''

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
                indxs = regions[region_name]
            elif sampling == 'loo':
                indxs = []
                for reg_name, reg_indxs in regions.items():
                    if reg_name != region_name:
                        indxs.append(reg_indxs)

                indxs = np.concatenate(indxs)
            else:
                raise ValueError(
                    '"%s" is an invalid region sampling option' % sampling)
            transforms_.append(SelectIndxs(
                indxs, str('%s-%s' % (region_name, sampling))))

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
            raise ValueError(
                '"%s" is an invalid neural type' % hparams['neural_type'])

        # compose filters
        if len(transforms_) == 0:
            transform = None
        else:
            transform = Compose(transforms_)

    elif data_type == 'ae_latents':
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
                ae_version = get_best_model_version(ae_dir, 'val_loss')[0]
            ae_latents = str('%slatents.pkl' % sess_id_str)
            path = os.path.join(ae_dir, ae_version, ae_latents)

    elif data_type == 'arhmm_states':

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
                arhmm_version = get_best_model_version(arhmm_dir, 'val_ll', best_def='max')[0]
            arhmm_states = str('%sstates.pkl' % sess_id_str)
            path = os.path.join(arhmm_dir, arhmm_version, arhmm_states)

    elif data_type == 'neural_ae_predictions':

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
                neural_ae_version = get_best_model_version(neural_ae_dir, 'val_loss')[0]
            neural_ae_predictions = str('%spredictions.pkl' % sess_id_str)
            path = os.path.join(neural_ae_dir, neural_ae_version, neural_ae_predictions)

    elif data_type == 'neural_arhmm_predictions':

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
                neural_arhmm_version = get_best_model_version(neural_arhmm_dir, 'val_loss')[0]
            neural_arhmm_predictions = str('%spredictions.pkl' % sess_id_str)
            path = os.path.join(neural_arhmm_dir, neural_arhmm_version, neural_arhmm_predictions)

    else:
        raise ValueError('"%s" is an invalid data_type' % data_type)

    return transform, path


def get_region_list(hparams):
    """
    Get regions and their indexes into neural data for current session

    Args:
        hparams (dict or namespace object):

    Returns:
        (dict)
    """
    import h5py

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    data_file = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'],
        hparams['animal'], hparams['session'], 'data.hdf5')

    with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
        indx_types = list(f['regions'])
        if 'indxs_consolidate' in indx_types:
            regions = list(f['regions']['indxs_consolidate'].keys())
            indxs = {reg: np.ravel(f['regions']['indxs_consolidate'][reg][()]) for
                     reg in regions}
        elif 'indxs_consolidate_lr' in indx_types:
            regions = list(f['regions']['indxs_consolidate_lr'].keys())
            indxs = {reg: np.ravel(f['regions']['indxs_consolidate_lr'][reg][()])
                     for reg in regions}
        else:
            regions = list(f['regions']['indxs'])
            indxs = {reg: np.ravel(f['regions']['indxs'][reg][()])
                     for reg in regions}

    return indxs


def get_best_model_and_data(hparams, Model, load_data=True, version='best', data_kwargs=None):
    """
    Helper function for loading the best model defined by hparams out of all available test-tube
    versions, as well as the associated data used to fit the model.

    Args:
        hparams (dict):
        Model (behavenet.models object:
        load_data (bool, optional):
        version (str or int, optional):
        data_kwargs (dict, optional): kwargs for data generator

    Returns:
        (tuple): (model, data generator)
    """

    from behavenet.data.data_generator import ConcatSessionsGenerator

    # get session_dir
    hparams['session_dir'], sess_ids = get_output_session_dir(hparams)
    expt_dir = get_expt_dir(hparams)

    # get best model version
    if version == 'best':
        best_version = get_best_model_version(expt_dir)[0]
    else:
        if isinstance(version, str) and version[0] == 'v':
            # assume we got a string of the form 'version_XX'
            best_version = version
        else:
            best_version = str('version_{}'.format(version))
    # get int representation as well
    best_version_int = int(best_version.split('_')[1])
    version_dir = os.path.join(expt_dir, best_version)
    arch_file = os.path.join(version_dir, 'meta_tags.pkl')
    model_file = os.path.join(version_dir, 'best_val_model.pt')
    if not os.path.exists(model_file) and not os.path.exists(model_file + '.meta'):
        model_file = os.path.join(version_dir, 'best_val_model.ckpt')
    print('Loading model defined in %s' % arch_file)

    with open(arch_file, 'rb') as f:
        hparams_new = pickle.load(f)

    # update paths if performing analysis on a different machine
    hparams_new['data_dir'] = hparams['data_dir']
    hparams_new['session_dir'] = hparams['session_dir']
    hparams_new['expt_dir'] = expt_dir
    hparams_new['use_output_mask'] = hparams.get('use_output_mask', False)
    hparams_new['device'] = 'cpu'

    # build data generator
    hparams_new, signals, transforms, paths = get_data_generator_inputs(hparams_new, sess_ids)
    if load_data:
        # sometimes we want a single data_generator for multiple models
        if data_kwargs is None:
            data_kwargs = {}
        data_generator = ConcatSessionsGenerator(
            hparams_new['data_dir'], sess_ids,
            signals_list=signals, transforms_list=transforms, paths_list=paths,
            device=hparams_new['device'], as_numpy=hparams_new['as_numpy'],
            batch_load=hparams_new['batch_load'], rng_seed=hparams_new['rng_seed'], **data_kwargs)
    else:
        data_generator = None

    # build models
    model = Model(hparams_new)
    model.version = best_version_int
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    model.to(hparams_new['device'])
    model.eval()

    return model, data_generator
