import os
import numpy as np


def get_subdirs(path):
    """get all first-level subdirectories in a given path (no recursion)"""
    if not os.path.exists(path):
        raise ValueError('%s is not a path' % path)
    try:
        return next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)


def get_output_session_dir(hparams, path_type='save'):
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

    Args:
        hparams (dict):
        path_type (str, optional): 'save' to use hparams['tt_save_path'],
            'data' to use hparams['data_dir']; note that using path_type='data'
            will not return multisession directories

    Returns:
        (tuple): (session_dir, sessions_single)
    """

    if 'sessions_csv' in hparams and len(hparams['sessions_csv']) > 0:
        # load from csv
        # TODO: collect sessions directly from session_info.csv file
        pass

    if path_type == 'save':
        base_dir = hparams['tt_save_path']
    elif path_type == 'data':
        base_dir = hparams['data_dir']
    else:
        raise ValueError('"%s" is an invalid path_type' % path_type)

    # get session dir (can include multiple sessions)
    sessions_single = []
    sessions_multi_paths = []
    lab = hparams['lab']
    if lab == 'all':
        raise ValueError('multiple labs not currently supported')
    elif hparams['expt'] == 'all':
        # get all experiments from one lab
        expts = get_subdirs(os.path.join(base_dir, lab))
        for expt in expts:
            if expt[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(base_dir, lab, expt))
                continue
            else:
                animals = get_subdirs(os.path.join(base_dir, lab, expt))
            for animal in animals:
                if animal[:5] == 'multi':
                    continue
                else:
                    sessions = get_subdirs(os.path.join(base_dir, lab, expt, animal))
                for session in sessions:
                    if session[:5] == 'multi':
                        continue
                    else:
                        # record bottom-level single-session directory
                        sessions_single.append({
                            'lab': lab, 'expt': expt, 'animal': animal, 'session': session})
        session_dir_base = os.path.join(base_dir, lab)
    elif hparams['animal'] == 'all':
        # get all animals from one experiment
        expt = hparams['expt']
        animals = get_subdirs(os.path.join(base_dir, lab, expt))
        for animal in animals:
            if animal[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(base_dir, lab, expt, animal))
                continue
            else:
                sessions = get_subdirs(os.path.join(base_dir, lab, expt, animal))
            for session in sessions:
                if session[:5] == 'multi':
                    continue
                else:
                    # record bottom-level single-session directory
                    sessions_single.append({
                        'lab': lab, 'expt': expt, 'animal': animal, 'session': session})
        session_dir_base = os.path.join(base_dir, lab, expt)
    elif hparams['session'] == 'all':
        # get all sessions from one animal
        expt = hparams['expt']
        animal = hparams['animal']
        sessions = get_subdirs(os.path.join(base_dir, lab, expt, animal))
        for session in sessions:
            if session[:5] == 'multi':
                # record top-level multi-session directory
                sessions_multi_paths.append(os.path.join(base_dir, lab, expt, animal, session))
                continue
            else:
                # record bottom-level single-session directory
                sessions_single.append({
                    'lab': lab, 'expt': expt, 'animal': animal, 'session': session})
        session_dir_base = os.path.join(base_dir, lab, expt, animal)
    else:
        sessions_single.append({
            'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
            'session': hparams['session']})
        session_dir_base = os.path.join(
            base_dir, hparams['lab'], hparams['expt'], hparams['animal'], hparams['session'])

    # construct session_dir
    if hparams.get('multisession', None) is not None:
        session_dir = os.path.join(session_dir_base, 'multisession-%02i' % hparams['multisession'])
    elif len(sessions_single) > 1:
        # check if this combo of experiments exists in prev multi-sessions
        found_match = False
        for session_multi in sessions_multi_paths:
            csv_file = os.path.join(session_multi, 'session_info.csv')
            sessions_multi = read_session_info_from_csv(csv_file)
            for d in sessions_multi:
                # save path doesn't matter for comparison
                d.pop('tt_save_path', None)
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
            multi_indxs = [
                int(session_multi.split('-')[-1]) for session_multi in sessions_multi_paths]
            if len(multi_indxs) == 0:
                multi_indx = 0
            else:
                multi_indx = max(multi_indxs) + 1
        else:
            pass

        session_dir = os.path.join(session_dir_base, 'multisession-%02i' % multi_indx)
    else:
        session_dir = session_dir_base

    return session_dir, sessions_single


def get_expt_dir(hparams, model_class=None, model_type=None, expt_name=None):
    """
    Get output directories associated with a particular model class/type/expt
    name.

    Args:
        hparams (dict):
        model_class (str, optional): will search `hparams` if not present
        model_type (str, optional): will search `hparams` if not present
        expt_name (str, optional): will search `hparams` if not present

    Returns:
        (str): contains data info (lab/expt/animal/session) as well as model
        info (e.g. n_ae_latents) and expt_name
    """

    import copy

    if model_class is None:
        model_class = hparams['model_class']

    if model_type is None:
        model_type = hparams['model_type']

    if expt_name is None:
        expt_name = hparams['experiment_name']

    # get results dir
    if model_class == 'ae':
        model_path = os.path.join('ae', model_type, '%02i_latents' % hparams['n_ae_latents'])
        if hparams.get('ae_multisession', None) is not None:
            # using a multisession autoencoder; assumes multisessionis at animal level
            # (rather than experiment level), i.e.
            # - latent session dir: lab/expt/animal/multisession-xx
            # - en/decoding session dir: lab/expt/animal/session
            hparams_ = copy.deepcopy(hparams)
            hparams_['session'] = 'all'
            hparams_['multisession'] = hparams['ae_multisession']
            session_dir, _ = get_output_session_dir(hparams_)
        else:
            session_dir = hparams['session_dir']
    elif model_class == 'neural-ae' or model_class == 'ae-neural':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'], model_type, brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'neural-arhmm' or model_class == 'arhmm-neural':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], model_type, brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'arhmm':
        model_path = os.path.join(
            'arhmm', '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], hparams['noise_type'])
        if hparams.get('arhmm_multisession', None) is not None:
            # using a multisession autoencoder with single session arhmm; assumes multisession
            # is at animal level (rather than experiment level), i.e.
            # - latent session dir: lab/expt/animal/multisession-xx
            # - arhmm session dir: lab/expt/animal/session
            hparams_ = copy.deepcopy(hparams)
            hparams_['session'] = 'all'
            hparams_['multisession'] = hparams['arhmm_multisession']
            session_dir, _ = get_output_session_dir(hparams_)
        else:
            session_dir = hparams['session_dir']
    elif model_class == 'bayesian-decoding':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            'bayesian-decoding', '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], hparams['noise_type'], brain_region)
        session_dir, _ = get_output_session_dir(hparams)
    else:
        raise ValueError('"%s" is an invalid model class' % model_class)
    expt_dir = os.path.join(session_dir, model_path, expt_name)

    return expt_dir


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
    if not os.path.isdir(session_dir):
        os.makedirs(session_dir)
    with open(session_file, mode='w') as f:
        session_writer = csv.DictWriter(f, fieldnames=list(ids_list[0].keys()))
        session_writer.writeheader()
        for ids in ids_list:
            session_writer.writerow(ids)


def contains_session(session_dir, session_id):
    """
    Helper function to determine if session defined by `session_id` dict is in
    the multi-session `session_dir`

    Args:
        session_dir (str):
        session_id (dict): must contain keys `lab`, `expt`, `animal` and
            `session`

    Returns:
        (bool)
    """
    session_ids = read_session_info_from_csv(os.path.join(session_dir, 'session_info.csv'))
    contains_sess = False
    for sess_id in session_ids:
        sess_id.pop('tt_save_path', None)
        if sess_id == session_id:
            contains_sess = True
            break
    return contains_sess


def find_session_dirs(hparams):
    """
    Helper function to find all session directories (single- and multi-session)
    that contain the session defined in `hparams`

    Args:
        hparams (dict): must contain keys `lab`, `expt`, `animal` and
            `session`

    Returns:
        (list of strs)
    """
    ids = {s: hparams[s] for s in ['lab', 'expt', 'animal', 'session']}
    lab = hparams['lab']
    expts = get_subdirs(os.path.join(hparams['tt_save_path'], lab))
    # need to grab all multi-sessions as well as the single session
    session_dirs = []  # full paths
    session_ids  = []  # dict of lab/expt/animal/session
    for expt in expts:
        if expt[:5] == 'multi':
            session_dir = os.path.join(hparams['tt_save_path'], lab, expt)
            if contains_session(session_dir, ids):
                session_dirs.append(session_dir)
                session_ids.append({
                    'lab': lab, 'expt': 'all', 'animal': '', 'session': '',
                    'multisession': int(expt[-2:])})
            continue
        else:
            animals = get_subdirs(os.path.join(
                hparams['tt_save_path'], lab, expt))
        for animal in animals:
            if animal[:5] == 'multi':
                session_dir = os.path.join(hparams['tt_save_path'], lab, expt, animal)
                if contains_session(session_dir, ids):
                    session_dirs.append(session_dir)
                    session_ids.append({
                        'lab': lab, 'expt': expt, 'animal': 'all', 'session': '',
                        'multisession': int(animal[-2:])})
                continue
            else:
                sessions = get_subdirs(os.path.join(
                    hparams['tt_save_path'], lab, expt, animal))
            for session in sessions:
                session_dir = os.path.join(
                    hparams['tt_save_path'], lab, expt, animal, session)
                if session[:5] == 'multi':
                    if contains_session(session_dir, ids):
                        session_dirs.append(session_dir)
                        session_ids.append({
                            'lab': lab, 'expt': expt, 'animal': animal, 'session': 'all',
                            'multisession': int(session[-2:])})
                else:
                    tmp_ids = {'lab': lab, 'expt': expt, 'animal': animal, 'session': session}
                    if tmp_ids == ids:
                        session_dirs.append(session_dir)
                        session_ids.append({
                            'lab': lab, 'expt': expt, 'animal': animal, 'session': session,
                            'multisession': None})
    return session_dirs, session_ids


def get_best_model_version(model_path, measure='val_loss', best_def='min', n_best=1):
    """
    Get best model version from test tube

    Args:
        model_path (str): test tube experiment directory containing version_%i
            subdirectories
        measure (str, optional): heading in csv file that is used to determine
            which model is best
        best_def (str, optional): how `measure` should be parsed; 'min' | 'max'
        n_best (int, optional): top `n_best` models are returned

    Returns:
        (str)
    """

    import pickle
    import pandas as pd

    # gather all versions
    versions = get_subdirs(model_path)

    # load csv files with model metrics (saved out from test tube)
    metrics = []
    for i, version in enumerate(versions):
        # make sure training has been completed
        with open(os.path.join(model_path, version, 'meta_tags.pkl'), 'rb') as f:
            meta_tags = pickle.load(f)
        if not meta_tags['training_completed']:
            continue
        # try:
        #     if meta_tags['n_lags']==16 or meta_tags['n_lags']==12:
        #         continue
        # except:
        #     pass

        try:

            # read metrics csv file
            metric = pd.read_csv(os.path.join(model_path, version, 'metrics.csv'))

            # ugly hack for now
            if model_path.find('arhmm') > -1 and model_path.find('neural') < 0:
                # throw error if not correct number of lags
                import pickle
                meta = os.path.join(model_path, version, 'meta_tags.pkl')
                with open(meta, 'rb') as f:
                    meta_tags = pickle.load(f)
                    if meta_tags['n_lags'] != 1:
                        raise Exception
        except:
            continue

        # get validation loss of best model
        if best_def == 'min':
            val_loss = metric[measure].min()
        elif best_def == 'max':
            val_loss = metric[measure].max()

        metrics.append(pd.DataFrame({'loss': val_loss, 'version': version}, index=[i]))
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
            best_versions = np.asarray(
                metrics_df['version'][metrics_df['loss'].nsmallest(n_best, 'all').index])
        elif best_def == 'max':
            raise NotImplementedError
        if best_versions.shape[0] != n_best:
            print('More versions than specified due to same validation loss')
        
    return best_versions


def experiment_exists(hparams, which_version=False):
    """
    Search test tube versions to find if an experiment with the same
    hyperparameters has been (successfully) fit

    Args:
        hparams (dict):
        which_version (bool): `True` to return version number

    Returns:
        (bool) if `which_version` is False
        (tuple) (bool, int) if `which_version` is True
    """

    import pickle
    import copy

    try:
        tt_versions = get_subdirs(hparams['expt_dir'])
    except StopIteration:
        # no versions yet
        if which_version:
            return False, None
        else:
            return False

    # get rid of parameters that are not model-specific
    # TODO: this is ugly and not easy to maintain
    hparams_less = copy.copy(hparams)
    hparams_less.pop('data_dir', None)
    hparams_less.pop('tt_save_path', None)
    hparams_less.pop('device', None)
    hparams_less.pop('as_numpy', None)
    hparams_less.pop('batch_load', None)
    hparams_less.pop('architecture_params', None)
    hparams_less.pop('list_index', None)
    hparams_less.pop('lab_example', None)
    hparams_less.pop('tt_n_gpu_trials', None)
    hparams_less.pop('tt_n_cpu_trials', None)
    hparams_less.pop('tt_n_cpu_workers', None)
    hparams_less.pop('use_output_mask', None)
    hparams_less.pop('ae_model_type', None)
    hparams_less.pop('subsample_regions', None)
    hparams_less.pop('reg_list', None)
    hparams_less.pop('version', None)
    hparams_less.pop('plot_n_frames', None)
    hparams_less.pop('plot_frame_rate', None)
    hparams_less.pop('ae_multisession', None)
    hparams_less.pop('best_version', None)
    found_match = False
    version = None
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
                    break
        except IOError:
            continue

    if which_version and found_match:
        return found_match, version
    elif which_version and not found_match:
        return found_match, None
    else:
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
        parser.add_argument('--trial_splits', default='8;1;1;0', type=str, help='i;j;k;l correspond to train;val;test;gap')
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
        parser.add_argument('--trial_splits', default='5;1;1;1', type=str, help='i;j;k;l correspond to train;val;test;gap')
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
        parser.add_argument('--trial_splits', default='5;1;1;1', type=str, help='i;j;k;l correspond to train;val;test;gap')
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
        parser.add_argument('--trial_splits', default='5;1;1;1', type=str, help='i;j;k;l correspond to train;val;test;gap')


def get_lab_example(hparams, lab):
    if lab == 'steinmetz':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '8-probe'
        hparams['animal'] = 'mouse-02'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 39.61
        hparams['x_pixels'] = 192
        hparams['y_pixels'] = 112
        hparams['n_input_channels'] = 1
        hparams['neural_bin_size'] = 1.0 / hparams['frame_rate']
        hparams['neural_type'] = 'spikes'
    if lab == 'steinmetz-face':
        hparams['lab'] = 'steinmetz'
        hparams['expt'] = '8-probe-face'
        hparams['animal'] = 'mouse-02'
        hparams['session'] = 'session-01'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 39.61
        hparams['x_pixels'] = 128
        hparams['y_pixels'] = 128
        hparams['n_input_channels'] = 1
        hparams['neural_bin_size'] = 1.0 / hparams['frame_rate']
        hparams['neural_type'] = 'spikes'
    elif lab == 'musall':
        hparams['lab'] = 'musall'
        hparams['expt'] = 'vistrained'
        hparams['animal'] = 'mSM36'
        hparams['session'] = '05-Dec-2017'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = False
        hparams['frame_rate'] = 30  # is this correct?
        hparams['x_pixels'] = 128
        hparams['y_pixels'] = 128
        hparams['n_input_channels'] = 2
        hparams['neural_bin_size'] = 1.0 / hparams['frame_rate']
        hparams['neural_type'] = 'ca'
    elif lab == 'datta':
        hparams['lab'] = 'datta'
        hparams['expt'] = 'inscopix'
        hparams['animal'] = '15566'
        hparams['session'] = '2018-11-27'
        hparams['n_ae_latents'] = 8
        hparams['use_output_mask'] = True
        hparams['frame_rate'] = 30
        hparams['x_pixels'] = 80
        hparams['y_pixels'] = 80
        hparams['n_input_channels'] = 1
        hparams['neural_bin_size'] = 1.0 / hparams['frame_rate']
        hparams['neural_type'] = 'ca'


def get_region_dir(hparams):
    if hparams.get('subsample_regions', 'none') == 'none':
        region_dir = 'all'
    elif hparams['subsample_regions'] == 'single':
        region_dir = str('%s-single' % hparams['region'])
    elif hparams['subsample_regions'] == 'loo':
        region_dir = str('%s-loo' % hparams['region'])
    else:
        raise ValueError('"%s" is an invalid regioin sampling type' % hparams['subsample_regions'])
    return region_dir


def create_tt_experiment(hparams):
    """
    Create test-tube experiment

    Args:
        hparams:

    Returns:
        tuple: (hparams, sess_ids, exp)
    """
    from test_tube import Experiment

    # get session_dir
    hparams['session_dir'], sess_ids = get_output_session_dir(hparams)

    if not os.path.isdir(hparams['session_dir']):
        os.makedirs(hparams['session_dir'])
        export_session_info_to_csv(hparams['session_dir'], sess_ids)
    hparams['expt_dir'] = get_expt_dir(hparams)
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])
    # print('')

    # check to see if experiment already exists
    if experiment_exists(hparams):
        print('Experiment exists! Aborting fit')
        return None, None, None

    # TODO: this was commented out in arhmm_decoding_grid_search - why?
    exp = Experiment(
        name=hparams['experiment_name'],
        debug=False,
        save_dir=os.path.dirname(hparams['expt_dir']))
    exp.save()
    hparams['version'] = exp.version

    return hparams, sess_ids, exp


def build_data_generator(hparams, sess_ids, export_csv=True):
    """

    Args:
        hparams (dict):
        sess_ids (list):
        export_csv (bool):

    Returns:
        ConcatSessionsGenerator
    """
    from behavenet.data.data_generator import ConcatSessionsGenerator
    from behavenet.data.utils import get_data_generator_inputs
    print('using data from following sessions:')
    for ids in sess_ids:
        print('%s' % os.path.join(
            hparams['tt_save_path'], ids['lab'], ids['expt'], ids['animal'], ids['session']))
    hparams, signals, transforms, paths = get_data_generator_inputs(hparams, sess_ids)
    if hparams.get('trial_splits', None) is not None:
        # assumes string of form 'train;val;test;gap'
        trs = [int(tr) for tr in hparams['trial_splits'].split(';')]
        trial_splits = {'train_tr': trs[0], 'val_tr': trs[1], 'test_tr': trs[2], 'gap_tr': trs[3]}
    else:
        trial_splits = None
    print('constructing data generator...', end='')
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], sess_ids,
        signals_list=signals, transforms_list=transforms, paths_list=paths,
        device=hparams['device'], as_numpy=hparams['as_numpy'], batch_load=hparams['batch_load'],
        rng_seed=hparams['rng_seed'], trial_splits=trial_splits)
    # csv order will reflect dataset order in data generator
    if export_csv:
        export_session_info_to_csv(os.path.join(
            hparams['expt_dir'], str('version_%i' % hparams['version'])), sess_ids)
    print('done')
    print(data_generator)
    return data_generator
