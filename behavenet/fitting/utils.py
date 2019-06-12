import os
import pickle
import numpy as np
import torch


def get_subdirs(path):
    """get all first-level subdirectories in a given path (no recursion)"""
    try:
        return next(os.walk(path))[1]
    except StopIteration:
        raise Exception('%s does not contain any subdirectories' % path)


def get_output_session_dir(hparams):
    """
    Get session-level directory for saving model outputs.

    If 'lab' == 'all', an error is thrown since multiple-lab runs are not supp
    If 'expt' == 'all', all sessions from all animals from all expts from the
        specified lab are used; the session_dir will then be
        `tt_save_path/lab/multisession-xx`
    If 'animal' == 'all', all sessions from all animals in the specified expt
        are used; the session_dir will then be
        `tt_save_path/lab/expt/multisession-xx`
    If 'session' == 'all', all sessions from the specified animal are used; the
        session_dir will then be
        `tt_save_path/lab/expt/animal/multisession-xx`
    If none of 'lab', 'expt', 'animal' or 'session' is 'all', session_dir is
        `tt_save_path/lab/expt/animal/session`

    The `multisession-xx` directory will contain a file `session_info.csv`
    which will contain information about the sessions that comprise the
    multisession; this file is used to determine whether or not a new
    multisession directory needs to be created.

    # TODO: currently searches RESULTS path instead of DATA path (need both)
    """

    import csv

    if 'sessions_csv' in hparams and len(hparams['sessions_csv']) > 0:
        # load from csv
        # TODO: collect sessions directly from session_info.csv file
        pass

    # get session dir (can include multiple sessions)
    sessions_single = []
    sessions_multi_paths = []
    lab = hparams['lab']
    if lab == 'all':
        raise ValueError('multiple labs not currently supported')
    elif hparams['expt'] == 'all':
        # get all experiments from one lab
        expts = get_subdirs(os.path.join(hparams['tt_save_path'], lab))
        for expt in expts:
            if expt[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(
                    hparams['tt_save_path'], lab, expt))
                continue
            else:
                animals = get_subdirs(
                    os.path.join(hparams['tt_save_path'], lab, expt))
            for animal in animals:
                if animal[:5] == 'multi':
                    continue
                else:
                    sessions = get_subdirs(os.path.join(
                        hparams['tt_save_path'], lab, expt, animal))
                for session in sessions:
                    if session[:5] == 'multi':
                        continue
                    else:
                        # record bottom-level single-session directory
                        sessions_single.append({
                            'tt_save_path': hparams['tt_save_path'],
                            'lab': lab, 'expt': expt, 'animal': animal,
                            'session': session})
        session_dir_base = os.path.join(hparams['tt_save_path'], lab)
    elif hparams['animal'] == 'all':
        # get all animals from one experiment
        expt = hparams['expt']
        animals = get_subdirs(os.path.join(hparams['tt_save_path'], lab, expt))
        for animal in animals:
            if animal[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(
                    hparams['tt_save_path'], lab, expt, animal))
                continue
            else:
                sessions = get_subdirs(
                    os.path.join(hparams['tt_save_path'], lab, expt, animal))
            for session in sessions:
                if session[:5] == 'multi':
                    continue
                else:
                    # record bottom-level single-session directory
                    sessions_single.append({
                        'tt_save_path': hparams['tt_save_path'],
                        'lab': lab, 'expt': expt, 'animal': animal,
                        'session': session})
        session_dir_base = os.path.join(hparams['tt_save_path'], lab, expt)
    elif hparams['session'] == 'all':
        # get all sessions from one animal
        expt = hparams['expt']
        animal = hparams['animal']
        sessions = get_subdirs(
            os.path.join(hparams['tt_save_path'], lab, expt, animal))
        for session in sessions:
            if session[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(
                    hparams['tt_save_path'], lab, expt, animal, session))
                continue
            else:
                # record bottom-level single-session directory
                sessions_single.append({
                    'tt_save_path': hparams['tt_save_path'],
                    'lab': lab, 'expt': expt, 'animal': animal,
                    'session': session})
        session_dir_base = os.path.join(
            hparams['tt_save_path'], lab, expt, animal)
    else:
        sessions_single.append({
            'tt_save_path': hparams['tt_save_path'],
            'lab': hparams['lab'], 'expt': hparams['expt'],
            'animal': hparams['animal'], 'session': hparams['session']})
        session_dir_base = os.path.join(
            hparams['tt_save_path'], hparams['lab'], hparams['expt'],
            hparams['animal'], hparams['session'])

    # construct session_dir
    if len(sessions_single) > 1:
        # check if this combo of experiments exists in prev multi-sessions
        found_match = False
        for session_multi in sessions_multi_paths:
            csv_file = os.path.join(session_multi, 'session_info.csv')
            sessions_multi = read_session_info_from_csv(csv_file)
            # compare to collection of single sessions above
            set_l1 = set(tuple(sorted(d.items())) for d in sessions_single)
            set_l2 = set(tuple(sorted(d.items())) for d in sessions_multi)
            set_diff = set_l1.symmetric_difference(set_l2)
            if len(set_diff) == 0:
                # found match; record index
                found_match = True
                multi_indx = int(session_multi.split('-')[-1])
                break

        # create new multi-index if match was not found
        if not found_match:
            multi_indxs = [int(session_multi.split('-')[-1])
                           for session_multi in sessions_multi_paths]
            if len(multi_indxs) == 0:
                multi_indx = 0
            else:
                multi_indx = max(multi_indxs) + 1
        else:
            pass

        session_dir = os.path.join(
            session_dir_base, 'multisession-%02i' % multi_indx)

    else:
        session_dir = session_dir_base

    return session_dir, sessions_single


def get_output_dirs(hparams, model_class=None, model_type=None, expt_name=None):

    if model_class is None:
        model_class = hparams['model_class']

    if model_type is None:
        model_type = hparams['model_type']

    if expt_name is None:
        expt_name = hparams['experiment_name']

    # get results dir
    if model_class == 'ae':
        results_dir = os.path.join(
            hparams['session_dir'], 'ae', model_type,
            '%02i_latents' % hparams['n_ae_latents'])
    elif model_class == 'neural-ae':
        # TODO: include brain region, ae version
        results_dir = os.path.join(
            hparams['session_dir'], 'neural-ae',
            '%02i_latents' % hparams['n_ae_latents'],
            model_type)
    elif model_class == 'neural-arhmm':
        results_dir = os.path.join(
            hparams['session_dir'], 'neural-arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            model_type)
    elif model_class == 'arhmm':
        results_dir = os.path.join(
            hparams['session_dir'], 'arhmm',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'])
    elif model_class == 'arhmm-decoding':
        results_dir = os.path.join(
            hparams['session_dir'], 'arhmm-decoding',
            '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'],
            hparams['noise_type'])
    else:
        raise ValueError('"%s" is an invalid model class' % model_class)

    expt_dir = os.path.join(results_dir, 'test_tube_data', expt_name)

    return results_dir, expt_dir


def read_session_info_from_csv(session_file):
    """
    Read csv file that contains lab/expt/animal/session info

    Args:
        session_file (str): /full/path/to/session_info.csv

    Returns:
        (list of dicts)
    """
    import csv
    sessions_multi = []
    # load and parse csv file that contains single session info
    with open(session_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sessions_multi.append(dict(row))
    return sessions_multi


def export_session_info_to_csv(session_dir, ids_list):
    import csv
    session_file = os.path.join(session_dir, 'session_info.csv')
    with open(session_file, mode='w') as f:
        session_writer = csv.DictWriter(f, fieldnames=list(ids_list[0].keys()))
        session_writer.writeheader()
        for ids in ids_list:
            session_writer.writerow(ids)


def get_best_model_version(model_path, measure='val_loss', n_best=1, best_def='min'):
    """

    Args:
        model_path (str): test tube experiment directory containing version_%i
            subdirectories
        measure (str):

    Returns:
        str

    """

    import pandas as pd

    # gather all versions
    versions = get_subdirs(model_path)

    # load csv files with model metrics (saved out from test tube)
    metrics = []
    for i, version in enumerate(versions):
        # read metrics csv file
        try:
            metric = pd.read_csv(
                os.path.join(model_path, version, 'metrics.csv'))
        except:
            continue
        # get validation loss of best model
        if best_def == 'min':
            val_loss = metric[measure].min()
        elif best_def == 'max':
            val_loss = metric[measure].max()
        metrics.append(pd.DataFrame({
            'loss': val_loss,
            'version': version}, index=[i]))
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    # get version with smallest loss
    
    if n_best == 1:
        if best_def == 'min':
            best_versions = [metrics_df['version'][metrics_df['loss'].idxmin()]]
        elif best_def == 'max':
            best_versions = [metrics_df['version'][metrics_df['loss'].idxmax()]]
    else:
        if best_def == 'min':
            best_versions = np.asarray(metrics_df['version'][metrics_df['loss'].nsmallest(n_best,'all').index])
        elif best_def == 'max':
            raise NotImplementedError
        if best_versions.shape[0] != n_best:
            print('More versions than specified due to same validation loss')
        
    return best_versions


def get_best_model_and_data(hparams, Model, load_data=True, version='best'):

    from behavenet.data.data_generator import ConcatSessionsGenerator

    # get session_dir
    if len(hparams['sessions_csv']) > 0:
        # TODO: collect sessions directly from session_info.csv file
        raise NotImplementedError
    else:
        sess_dir, sess_ids = get_output_session_dir(hparams)
    results_dir, expt_dir = get_output_dirs(hparams)

    # get best model version
    if version == 'best':
        best_version = get_best_model_version(expt_dir)[0]
    else:
        if isinstance(version, str) and version[0] == 'v':
            # assume we got a string of the form 'version_XX'
            best_version = version
        else:
            best_version = str('version_{}'.format(version))
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
    hparams_new['session_dir'] = sess_dir
    hparams_new['results_dir'] = results_dir
    hparams_new['expt_dir'] = expt_dir
    hparams_new['use_output_mask'] = hparams['use_output_mask'] # TODO: get rid of eventually
    hparams_new['device'] = 'cpu'
    
    # build data generator
    hparams_new, signals, transforms, load_kwargs = get_data_generator_inputs(
        hparams_new)
    if load_data:
        # sometimes we want a single data_generator for multiple models
        data_generator = ConcatSessionsGenerator(
            hparams_new['data_dir'], sess_ids,
            signals=signals, transforms=transforms, load_kwargs=load_kwargs,
            device=hparams_new['device'], as_numpy=hparams_new['as_numpy'],
            batch_load=hparams_new['batch_load'], rng_seed=hparams_new['rng_seed'])
    else:
        data_generator = None

    # build models
    model = Model(hparams_new)
    model.version = best_version
    model.load_state_dict(torch.load(
        model_file, map_location=lambda storage, loc: storage))
    model.to(hparams_new['device'])
    model.eval()

    return model, data_generator


def experiment_exists(hparams):

    import pickle
    import copy

    try:
        tt_versions = get_subdirs(hparams['expt_dir'])
    except StopIteration:
        # no versions yet
        return False

    # get rid of extra dict
    # TODO: this is ugly and not easy to maintain
    hparams_less = copy.copy(hparams)
    hparams_less.pop('architecture_params', None)
    hparams_less.pop('list_index', None)
    hparams_less.pop('lab_example', None)
    hparams_less.pop('tt_nb_gpu_trials', None)
    hparams_less.pop('tt_nb_cpu_trials', None)
    hparams_less.pop('tt_nb_cpu_workers', None)
    hparams_less.pop('lib', None)
    hparams_less.pop('use_output_mask', None)
    hparams_less.pop('ae_model_type', None)

    found_match = False
    for version in tt_versions:
        # load hparams
        version_file = os.path.join(hparams['expt_dir'], version, 'meta_tags.pkl')
        try:
            with open(version_file, 'rb') as f:
                hparams_ = pickle.load(f)
            if all([hparams_[key] == hparams_less[key] for key in hparams_less.keys()]):
                # found match - did it finish training?
                if hparams_['training_completed']:
                    found_match = True
                    print('model found with complete training; aborting')
                    break
            # else:
            #     print()
            #     print()
            #     for key in hparams_less.keys():
            #         val1 = hparams_[key]
            #         val2 = hparams_less[key]
            #         if val1 != val2:
            #             print('Key: {}; val1: {}; val2 {}'.format(key, val1, val2))
        except IOError:
            continue

    return found_match


def export_hparams(hparams, exp):
    """
    Export hyperparameter dictionary as csv file (for easy human reading) and
    as a pickled dict (for easy loading)

    Args:
        hparams (dict):
        exp (test_tube.Experiment object):
    """

    import pickle

    # save out as pickle
    meta_file = os.path.join(
        hparams['expt_dir'], 'version_%i' % exp.version, 'meta_tags.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(hparams, f)

    # save out as csv
    exp.tag(hparams)
    exp.save()


def get_data_generator_inputs(hparams):
    """
    Helper function for generating signals, transforms and load_kwargs for
    common models
    """

    from behavenet.data.transforms import SelectRegion
    from behavenet.data.transforms import Threshold
    from behavenet.data.transforms import ZScore
    from behavenet.data.transforms import BlockShuffle
    from behavenet.data.transforms import Compose

    # get neural signals/transforms/load_kwargs
    if hparams['model_class'].find('neural') > -1:

        neural_transforms_ = []
        neural_kwargs = None

        # filter neural data by region
        if hparams['region'] != 'all':
            # get region and indices
            region_name = hparams['region']
            _, region_indxs = get_region_list(hparams)
            neural_transforms_.append(SelectRegion(
                region_name, region_indxs[region_name]))

        # filter neural data by activity
        if hparams['neural_type'] == 'spikes':
            if hparams['neural_thresh'] > 0:
                neural_transforms_.append(Threshold(
                    threshold=hparams['neural_thresh'],
                    bin_size=hparams['neural_bin_size']))
        elif hparams['neural_type'] == 'ca':
            neural_transforms_.append(ZScore())
        else:
            raise ValueError(
                '"%s" is an invalid neural type' % hparams['neural_type'])

        if len(neural_transforms_) == 0:
            neural_transforms = None
        else:
            neural_transforms = Compose(neural_transforms_)

    else:
        neural_transforms = None
        neural_kwargs = None

    # get model-specific signals/transforms/load_kwargs
    if hparams['model_class'] == 'ae':

        if hparams['use_output_mask']:
            signals = [hparams['signals'], 'masks']
            transforms = [hparams['transforms'], None]
            load_kwargs = [None, None]
        else:
            signals = [hparams['signals']]
            transforms = [hparams['transforms']]
            load_kwargs = [None]

    elif hparams['model_class'] == 'neural-ae':

        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'ae'
        hparams['output_size'] = hparams['n_ae_latents']
        if hparams['model_type'][-2:] == 'mv':
            hparams['noise_dist'] = 'gaussian-full'
        else:
            hparams['noise_dist'] = 'gaussian'

        _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        signals = ['neural', 'ae']
        transforms = [neural_transforms, ae_transforms]
        load_kwargs = [neural_kwargs, ae_kwargs]

    elif hparams['model_class'] == 'neural-arhmm':

        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'arhmm'
        hparams['output_size'] = hparams['n_arhmm_states']
        hparams['noise_dist'] = 'categorical'

        _, arhmm_dir = get_output_dirs(
            hparams, model_class='arhmm',
            expt_name=hparams['arhmm_experiment_name'])

        if 'shuffle_rng_seed' in hparams:
            arhmm_transforms = BlockShuffle(hparams['shuffle_rng_seed'])
        else:
            arhmm_transforms = None
        arhmm_kwargs = {
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        signals = ['neural', 'arhmm']
        transforms = [neural_transforms, arhmm_transforms]
        load_kwargs = [neural_kwargs, arhmm_kwargs]

    elif hparams['model_class'] == 'arhmm':

        _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        if hparams['use_output_mask']:
            signals = ['ae', 'images', 'masks']
            transforms = [ae_transforms, None, None]
            load_kwargs = [ae_kwargs, None, None]
        else:
            signals = ['ae', 'images']
            transforms = [ae_transforms, None]
            load_kwargs = [ae_kwargs, None]

    elif hparams['model_class'] == 'arhmm-decoding':

        _, ae_dir = get_output_dirs(
            hparams, model_class='ae',
            expt_name=hparams['ae_experiment_name'],
            model_type=hparams['ae_model_type'])

        ae_transforms = None
        ae_kwargs = {
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']} 

        _, ae_predictions_dir = get_output_dirs(
            hparams, model_class='neural-ae',
            expt_name=hparams['neural_ae_experiment_name'],
            model_type=hparams['neural_ae_model_type'])
        ae_predictions_transforms = None
        ae_predictions_kwargs = {
            'model_dir': ae_predictions_dir,
            'model_version': hparams['neural_ae_version']} 

        _, arhmm_predictions_dir = get_output_dirs(
            hparams, model_class='neural-arhmm',
            expt_name=hparams['neural_arhmm_experiment_name'],
            model_type=hparams['neural_arhmm_model_type'])
        arhmm_predictions_transforms = None
        arhmm_predictions_kwargs = {
            'model_dir': arhmm_predictions_dir,
            'model_version': hparams['neural_arhmm_version']} 

        _, arhmm_dir = get_output_dirs(
            hparams, model_class='arhmm',
            expt_name=hparams['arhmm_experiment_name'])
        arhmm_transforms = None
        arhmm_kwargs = {
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        if hparams['use_output_mask']:
            signals = ['ae', 'images', 'masks','ae_predictions', 'arhmm_predictions', 'arhmm']
            transforms = [ae_transforms, None, None, ae_predictions_transforms, arhmm_predictions_transforms, arhmm_transforms]
            load_kwargs = [ae_kwargs, None, None, ae_predictions_kwargs, arhmm_predictions_kwargs, arhmm_kwargs]
        else:
            signals = ['ae', 'images', 'ae_predictions', 'arhmm_predictions', 'arhmm']
            transforms = [ae_transforms, None, ae_predictions_transforms, arhmm_predictions_transforms, arhmm_transforms]
            load_kwargs = [ae_kwargs, None, ae_predictions_kwargs, arhmm_predictions_kwargs, arhmm_kwargs]

    else:
        raise ValueError('"%s" is an invalid model_class' % hparams['model_class'])

    return hparams, signals, transforms, load_kwargs


def add_lab_defaults_to_parser(parser, lab=None):

    if lab == 'musall':
        parser.add_argument('--n_input_channels', default=2, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', default=128, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', default=128, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', default='musall', type=str)
        parser.add_argument('--expt', default='vistrained', type=str)
        parser.add_argument('--animal', default='mSM30', type=str)
        parser.add_argument('--session', default='10-Oct-2017', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='ca', choices=['spikes', 'ca'])
    elif lab == 'steinmetz':
        parser.add_argument('--n_input_channels', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', default=192, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', default=112, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', default='steinmetz', type=str)
        parser.add_argument('--expt', default='2-probe', type=str)
        parser.add_argument('--animal', default='mouse-01', type=str)
        parser.add_argument('--session', default='session-01', type=str)
        parser.add_argument('--neural_bin_size', default=39.61, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])
    elif lab == 'steinmetz-face':
        parser.add_argument('--n_input_channels', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', default=128, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', default=128, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', '-b', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', default='steinmetz', type=str)
        parser.add_argument('--expt', default='2-probe-face', type=str)
        parser.add_argument('--animal', default='mouse-01', type=str)
        parser.add_argument('--session', default='session-01', type=str)
        parser.add_argument('--neural_bin_size', default=39.61, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])
    elif lab == 'datta':
        parser.add_argument('--n_input_channels', default=1, help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', default=80, help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', default=80, help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=True, action='store_true')
        parser.add_argument('--approx_batch_size', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', default='datta', type=str)
        parser.add_argument('--expt', default='inscopix', type=str)
        parser.add_argument('--animal', default='15566', type=str)
        parser.add_argument('--session', default='2018-11-27', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='ca', choices=['spikes', 'ca'])
    else:
        parser.add_argument('--n_input_channels', help='list of n_channels', type=int)
        parser.add_argument('--x_pixels', help='number of pixels in x dimension', type=int)
        parser.add_argument('--y_pixels', help='number of pixels in y dimension', type=int)
        parser.add_argument('--use_output_mask', default=False, action='store_true')
        parser.add_argument('--approx_batch_size', default=200, help='batch_size', type=int) # approximate batch size for memory calculation
        parser.add_argument('--lab', type=str)
        parser.add_argument('--expt', type=str)
        parser.add_argument('--animal', type=str)
        parser.add_argument('--session', type=str)
        parser.add_argument('--neural_bin_size', default=None, help='ms')
        parser.add_argument('--neural_type', default='spikes', choices=['spikes', 'ca'])


def get_lab_example(hparams, lab):
    if lab == 'steinmetz':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '2-probe'
        hparams['animal'] = 'mouse-01'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 12
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 25
    if lab == 'steinmetz-face':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '2-probe-face'
        hparams['animal'] = 'mouse-01'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 12
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 25
    elif lab == 'musall':
        hparams['lab'] = 'musall'
        hparams['expt'] = 'vistrained'
        hparams['animal'] = 'mSM30'
        hparams['session'] = '10-Oct-2017'
        hparams['n_ae_latents'] = 16
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 30  # is this correct?
    elif lab == 'datta':
        hparams['lab'] = 'datta'
        hparams['expt'] = 'inscopix'
        hparams['animal'] = '15566'
        hparams['session'] = '2018-11-27'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = True
        hparams['frame_rate'] = 30


def get_region_list(hparams):
    """
    Get regions and their indexes into neural data for current session

    Args:
        hparams (dict or namespace object):

    Returns:
        (tuple)
            regions (list of strs)
            indxs (dict)
    """
    import h5py

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    data_file = os.path.join(
        hparams['data_dir'], hparams['lab'], hparams['expt'],
        hparams['animal'], hparams['session'], 'data.hdf5')

    with h5py.File(data_file, 'r', libver='latest', swmr=True) as f:
        indx_types = list(f['regions'])
        if 'indxs_consolidate_lr' in indx_types:
            regions = list(f['regions']['indxs_consolidate_lr'].keys())
            indxs = {reg: f['regions']['indxs_consolidate_lr'][reg][()]
                     for reg in regions}
        elif 'indxs_consolidate' in indx_types:
            regions = list(f['regions']['indxs_consolidate'].keys())
            indxs = {reg: f['regions']['indxs_consolidate'][reg][()]
                     for reg in regions}
        else:
            regions = list(f['regions']['indxs'])
            indxs = {reg: f['regions']['indxs'][reg][()] for reg in regions}

    return regions, indxs
