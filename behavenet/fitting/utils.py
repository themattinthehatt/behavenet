import os
import pickle
import numpy as np
from behavenet.data.utils import get_data_generator_inputs


def get_subdirs(path):
    """get all first-level subdirectories in a given path (no recursion)"""
    if not os.path.exists(path):
        raise ValueError('%s is not a path' % path)
    try:
        return next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)


def get_user_dir(type):
    """
    Get a directory from user-defined `directories` json file

    Args:
        type (str): 'data' | 'save' | 'fig'

    Returns:
        (str): absolute path for requested directory
    """
    import json
    from behavenet import get_params_dir
    dirs_file = os.path.join(get_params_dir(), 'directories')
    with open(dirs_file, 'r') as f:
        dirs = json.load(f)
    return dirs[str('%s_dir' % type)]


def get_data_dir():
    return get_user_dir('data')


def get_save_dir():
    return get_user_dir('save')


def get_fig_dir():
    return get_user_dir('fig')


def _get_multisession_paths(base_dir, lab='', expt='', animal=''):
    """
    Returns all paths in `sub_dirs` that start with `multi`. The absolute paths returned are
    determined by `base_dir`, `lab`, `expt`, `animal`, and `session` as follows:
    base_dir/lab/expt/animal/session/sub_dir
    Use empty strings to ignore one of the session id components.

    Args:
        base_dir (str):
        lab (str, optional):
        expt (str, optional):
        animal (str, optional):

    Returns:
        (list): list of absolute paths
    """
    sub_dirs = get_subdirs(os.path.join(base_dir, lab, expt, animal))
    multi_paths = []
    for sub_dir in sub_dirs:
        if sub_dir[:5] == 'multi':
            # record top-level multi-session directory
            multi_paths.append(os.path.join(base_dir, lab, expt, animal, sub_dir))
    return multi_paths


def _get_single_sessions(base_dir, depth, curr_depth):
    """
    Recursively search through non-multisession directories for all single sessions

    Args:
        base_dir (str):
        depth (int): depth of recursion
        curr_depth (int): current depth in recursion

    Returns:
        list of dicts: session ids for all single sessions in `base_dir`
    """
    session_list = []
    if curr_depth < depth:
        curr_depth += 1
        sub_dirs = get_subdirs(base_dir)
        for sub_dir in sub_dirs:
            if sub_dir[:12] != 'multisession':
                session_list += _get_single_sessions(
                    os.path.join(base_dir, sub_dir), depth=depth, curr_depth=curr_depth)
    elif curr_depth == depth:
        # take previous 4 directories (lab/expt/animal/session)
        sess_path = base_dir.split(os.sep)
        session_list = [{
            'lab': sess_path[-4],
            'expt': sess_path[-3],
            'animal': sess_path[-2],
            'session': sess_path[-1]}]
    return session_list


def get_session_dir(hparams, path_type='save'):
    """
    Get session-level directory for saving model outputs. Relies on hparams keys `sessions_csv`,
    `multisession`, `lab`, `expt`, `animal` and `session`.

    `sessions_csv` takes precedence. The value for this key is a non-empty string of the pattern
    `/path/to/session_info.csv`, where `session_info.csv` has 4 columns for lab, expt, animal and
    session.

    If `sessions_csv` is an empty string or the key is not in `hparams`, the following occurs:

    If 'lab' == 'all', an error is thrown since multiple-lab runs are not currently supported
    If 'expt' == 'all', all sessions from all animals from all expts from the specified lab are
        used; the session_dir will then be `save_dir/lab/multisession-xx`
    If 'animal' == 'all', all sessions from all animals in the specified expt are used; the
        session_dir will then be `save_dir/lab/expt/multisession-xx`
    If 'session' == 'all', all sessions from the specified animal are used; the session_dir will
        then be `save_dir/lab/expt/animal/multisession-xx`
    If none of 'lab', 'expt', 'animal' or 'session' is 'all', session_dir is
        `save_dir/lab/expt/animal/session`

    The `multisession-xx` directory will contain a file `session_info.csv` which will contain
    information about the sessions that comprise the multisession; this file is used to determine
    whether or not a new multisession directory needs to be created.

    Args:
        hparams (dict):
        path_type (str, optional): 'save' to use hparams['save_dir'], 'data' to use
        hparams['data_dir'] as base directory; note that using path_type='data' will not return
        multisession directories

    Returns:
        (tuple): (session_dir, sessions_single)
    """

    if path_type == 'save':
        base_dir = hparams['save_dir']
    elif path_type == 'data':
        base_dir = hparams['data_dir']
    else:
        raise ValueError('"%s" is an invalid path_type' % path_type)

    if len(hparams.get('sessions_csv', [])) > 0:
        # collect all single sessions from csv
        sessions_single = read_session_info_from_csv(hparams['sessions_csv'])
        labs, expts, animals, sessions = [], [], [], []
        for sess in sessions_single:
            sess.pop('tt_save_path', None)  # TODO: remove
            sess.pop('save_dir', None)
            labs.append(sess['lab'])
            expts.append(sess['expt'])
            animals.append(sess['animal'])
            sessions.append(sess['session'])
        # find appropriate session directory
        labs, expts, animals, sessions = \
            np.array(labs), np.array(expts), np.array(animals), np.array(sessions)
        lab, expt, animal, session = '', '', '', ''
        if len(np.unique(sessions)) == 1:
            # get single session from one animal
            lab, expt, animal, session = labs[0], expts[0], animals[0], sessions[0]
            session_dir_base = os.path.join(base_dir, lab, expt, animal, session)
        elif len(np.unique(animals)) == 1:
            # get all sessions from one animal
            lab, expt, animal = labs[0], expts[0], animals[0]
            session_dir_base = os.path.join(base_dir, lab, expt, animal)
        elif len(np.unique(expts)) == 1:
            lab, expt = labs[0], expts[0]
            # get all animals from one experiment
            session_dir_base = os.path.join(base_dir, lab, expt)
        elif len(np.unique(labs)) == 1:
            # get all experiments from one lab
            lab = labs[0]
            session_dir_base = os.path.join(base_dir, lab)
        else:
            raise NotImplementedError('multiple labs not currently supported')
        # find corresponding multisession (ok if they don't exist)
        multisession_paths = _get_multisession_paths(base_dir, lab=lab, expt=expt, animal=animal)

    else:
        # get session dirs (can include multiple sessions)
        lab = hparams['lab']
        if lab == 'all':
            raise NotImplementedError('multiple labs not currently supported')
        elif hparams['expt'] == 'all':
            # get all experiments from one lab
            multisession_paths = _get_multisession_paths(base_dir, lab=lab)
            sessions_single = _get_single_sessions(
                os.path.join(base_dir, lab), depth=3, curr_depth=0)
            session_dir_base = os.path.join(base_dir, lab)
        elif hparams['animal'] == 'all':
            # get all animals from one experiment
            expt = hparams['expt']
            multisession_paths = _get_multisession_paths(base_dir, lab=lab, expt=expt)
            sessions_single = _get_single_sessions(
                os.path.join(base_dir, lab, expt), depth=2, curr_depth=0)
            session_dir_base = os.path.join(base_dir, lab, expt)
        elif hparams['session'] == 'all':
            # get all sessions from one animal
            expt = hparams['expt']
            animal = hparams['animal']
            multisession_paths = _get_multisession_paths(
                base_dir, lab=lab, expt=expt, animal=animal)
            sessions_single = _get_single_sessions(
                os.path.join(base_dir, lab, expt, animal), depth=1, curr_depth=0)
            session_dir_base = os.path.join(base_dir, lab, expt, animal)
        else:
            multisession_paths = []
            sessions_single = [{
                'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
                'session': hparams['session']}]
            session_dir_base = os.path.join(
                base_dir, hparams['lab'], hparams['expt'], hparams['animal'], hparams['session'])

    # construct session_dir
    if hparams.get('multisession', None) is not None and len(hparams.get('sessions_csv', [])) == 0:
        session_dir = os.path.join(session_dir_base, 'multisession-%02i' % hparams['multisession'])
        # overwrite sessions_single with whatever is in requested multisession
        sessions_single = read_session_info_from_csv(os.path.join(session_dir, 'session_info.csv'))
        for sess in sessions_single:
            sess.pop('tt_save_path', None)  # TODO: remove
            sess.pop('save_dir', None)
    elif len(sessions_single) > 1:
        # check if this combo of experiments exists in previous multi-sessions
        found_match = False
        multi_indx = None
        for session_multi in multisession_paths:
            csv_file = os.path.join(session_multi, 'session_info.csv')
            sessions_multi = read_session_info_from_csv(csv_file)
            for d in sessions_multi:
                # save path doesn't matter for comparison
                d.pop('save_dir', None)
                d.pop('tt_save_path', None)  # TODO: remove
            # compare to collection of single sessions above
            set_l1 = set(tuple(sorted(d.items())) for d in sessions_single)
            set_l2 = set(tuple(sorted(d.items())) for d in sessions_multi)
            set_diff = set_l1.symmetric_difference(set_l2)
            if len(set_diff) == 0:
                # found match; record index
                found_match = True
                multi_indx = int(session_multi.split('-')[-1])
                break

        # create new multisession if match was not found
        if not found_match:
            multi_indxs = [
                int(session_multi.split('-')[-1]) for session_multi in multisession_paths]
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
            session_dir, _ = get_session_dir(hparams_)
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
    elif model_class == 'arhmm' or model_class == 'hmm':
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'],
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
            session_dir, _ = get_session_dir(hparams_)
        else:
            session_dir = hparams['session_dir']
    elif model_class == 'arhmm-decoding':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            'arhmm-decoding', '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            '%.0e_kappa' % hparams['kappa'], hparams['noise_type'], brain_region)
        session_dir = hparams['session_dir']
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
    Helper function to determine if session defined by `session_id` dict is in the multi-session
    `session_dir`

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
        sess_id.pop('save_dir', None)
        if sess_id == session_id:
            contains_sess = True
            break
    return contains_sess


def find_session_dirs(hparams):
    """
    Helper function to find all session directories (single- and multi-session) that contain the
    session defined in `hparams`

    Args:
        hparams (dict): must contain keys `lab`, `expt`, `animal` and
            `session`

    Returns:
        (list of strs)
    """
    ids = {s: hparams[s] for s in ['lab', 'expt', 'animal', 'session']}
    lab = hparams['lab']
    expts = get_subdirs(os.path.join(hparams['save_dir'], lab))
    # need to grab all multi-sessions as well as the single session
    session_dirs = []  # full paths
    session_ids  = []  # dict of lab/expt/animal/session
    for expt in expts:
        if expt[:5] == 'multi':
            session_dir = os.path.join(hparams['save_dir'], lab, expt)
            if contains_session(session_dir, ids):
                session_dirs.append(session_dir)
                session_ids.append({
                    'lab': lab, 'expt': 'all', 'animal': '', 'session': '',
                    'multisession': int(expt[-2:])})
            continue
        else:
            animals = get_subdirs(os.path.join(
                hparams['save_dir'], lab, expt))
        for animal in animals:
            if animal[:5] == 'multi':
                session_dir = os.path.join(hparams['save_dir'], lab, expt, animal)
                if contains_session(session_dir, ids):
                    session_dirs.append(session_dir)
                    session_ids.append({
                        'lab': lab, 'expt': expt, 'animal': 'all', 'session': '',
                        'multisession': int(animal[-2:])})
                continue
            else:
                sessions = get_subdirs(os.path.join(
                    hparams['save_dir'], lab, expt, animal))
            for session in sessions:
                session_dir = os.path.join(
                    hparams['save_dir'], lab, expt, animal, session)
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
    hparams_less.pop('save_dir', None)
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
    hparams_less.pop('session_dir', None)
    hparams_less.pop('expt_dir', None)

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
    meta_file = os.path.join(hparams['expt_dir'], 'version_%i' % exp.version, 'meta_tags.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(hparams, f)

    # save out as csv
    exp.tag(hparams)
    exp.save()


def get_lab_example(hparams, lab, expt):
    import json
    from behavenet import get_params_dir
    params_file = os.path.join(get_params_dir(), str('%s_%s_params' % (lab, expt)))
    with open(params_file, 'r') as f:
        dparams = json.load(f)
    hparams.update(dparams)


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
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
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
            hparams['save_dir'], ids['lab'], ids['expt'], ids['animal'], ids['session']))
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
        rng_seed=hparams['rng_seed_data'], trial_splits=trial_splits,
        train_frac=hparams['train_frac'])
    # csv order will reflect dataset order in data generator
    if export_csv:
        export_session_info_to_csv(os.path.join(
            hparams['expt_dir'], str('version_%i' % hparams['version'])), sess_ids)
    print('done')
    print(data_generator)
    return data_generator


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
        # read metrics csv file
        metric = pd.read_csv(os.path.join(model_path, version, 'metrics.csv'))
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
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
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


# TODO: delete
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
