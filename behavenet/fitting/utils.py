"""Utility functions for managing model paths and the hparams dict."""

import os
import pickle
import numpy as np

# to ignore imports for sphinx-autoapidoc
__all__ = [
    'get_subdirs', 'get_session_dir', 'get_expt_dir', 'read_session_info_from_csv',
    'export_session_info_to_csv', 'contains_session', 'find_session_dirs', 'experiment_exists',
    'get_model_params', 'export_hparams', 'get_lab_example', 'get_region_dir',
    'create_tt_experiment', 'get_best_model_version',
    'get_best_model_and_data']


def get_subdirs(path):
    """Get all first-level subdirectories in a given path (no recursion).

    Parameters
    ----------
    path : :obj:`str`
        absolute path

    Returns
    -------
    :obj:`list`
        first-level subdirectories in :obj:`path`

    """
    if not os.path.exists(path):
        raise NotADirectoryError('%s is not a path' % path)
    try:
        s = next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)
    if len(s) == 0:
        raise StopIteration('%s does not contain any subdirectories' % path)
    return s


def _get_multisession_paths(base_dir, lab='', expt='', animal=''):
    """Returns all paths in `base_dir` that start with `multi`.

    The absolute paths returned are determined by `base_dir`, `lab`, `expt`, `animal`, and
    `session` as follows: :obj:`base_dir/lab/expt/animal/session/sub_dir`

    Use empty strings to ignore one of the session id components.

    Parameters
    ----------
    base_dir : :obj:`str`
    lab : :obj:`str`, optional
    expt : :obj:`str`, optional
    animal : :obj:`str`, optional

    Returns
    -------
    :obj:`list`
        list of absolute paths

    """
    multi_paths = []
    try:
        sub_dirs = get_subdirs(os.path.join(base_dir, lab, expt, animal))
        for sub_dir in sub_dirs:
            if sub_dir[:5] == 'multi':
                # record top-level multi-session directory
                multi_paths.append(os.path.join(base_dir, lab, expt, animal, sub_dir))
    except ValueError:
        print('warning: did not find requested multisession(s)')
    except NotADirectoryError:
        print('warning: did not find any sessions')
    except StopIteration:
        print('warning: did not find any sessions')

    return multi_paths


def _get_single_sessions(base_dir, depth, curr_depth):
    """Recursively search through non-multisession directories for all single sessions.

    Parameters
    ----------
    base_dir : :obj:`str`
    depth : :obj:`int`
        depth of recursion
    curr_depth : :obj:`int`
        current depth in recursion

    Returns
    -------
    :obj:`list` of :obj:`dict`
        session ids for all single sessions in :obj:`base_dir`

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


def _get_transition_str(hparams):
    """

    Parameters
    ----------
    hparams : :obj:`dict`
        model hyperparameters; needs key 'transitions' and 'kappa' if using sticky transitions

    Returns
    -------
    :obj:`str`
        arhmm transition string used for model path specification

    """
    if hparams['transitions'] == 'sticky':
        return 'sticky_%.0e' % hparams['kappa']
    else:
        return hparams['transitions']


def get_session_dir(hparams, session_source='save'):
    """Get session-level directory for saving model outputs from hparams dict.

    Relies on hparams keys 'sessions_csv', 'multisession', 'lab', 'expt', 'animal', 'session'.

    The :obj:`sessions_csv` key takes precedence. The value for this key is a non-empty string of
    the pattern :obj:`/path/to/session_info.csv`, where `session_info.csv` has 4 columns for lab,
    expt, animal and session.

    If `sessions_csv` is an empty string or the key is not in `hparams`, the following occurs:

    - if :obj:`'lab' == 'all'`, an error is thrown since multiple-lab runs are not currently
      supported
    - if :obj:`'expt' == 'all'`, all sessions from all animals from all expts from the specified
      lab are used; the session_dir will then be :obj:`save_dir/lab/multisession-xx`
    - if :obj:`'animal' == 'all'`, all sessions from all animals in the specified expt are used;
      the session_dir will then be :obj:`save_dir/lab/expt/multisession-xx`
    - if :obj:`'session' == 'all'`, all sessions from the specified animal are used; the
      session_dir will then be :obj:`save_dir/lab/expt/animal/multisession-xx`
    - if none of 'lab', 'expt', 'animal' or 'session' is 'all', session_dir is
      :obj:`save_dir/lab/expt/animal/session`

    The :obj:`session_source` argument defines where the code looks for sessions whenever one
    of 'lab', 'expt', 'animal', or 'session' is :obj:`'all'`; if :obj:`'session_source' = 'data'`,
    the data directory is searched for sessions; if :obj:`'session_source' = 'save'`, the save
    directory is searched for sessions. This means that only sessions that have been previously
    used for fitting models will be included.

    The :obj:`multisession-xx` directory will contain a file :obj:`session_info.csv` which will
    contain information about the sessions that comprise the multisession; this file is used to
    determine whether or not a new multisession directory needs to be created.


    Parameters
    ----------
    hparams : :obj:`dict`
        requires `sessions_csv`, `multisession`, `lab`, `expt`, `animal` and `session`
    session_source : :obj:`str`, optional
        'save' to use hparams['save_dir'], 'data' to use hparams['data_dir'] as base directory;
        note that using :obj:`path_type='data'` will not return multisession directories

    Returns
    -------
    :obj:`tuple`
        - session_dir (:obj:`str`)
        - sessions_single (:obj:`list`)

    """

    save_dir = hparams['save_dir']
    if session_source == 'save':
        sess_dir = hparams['save_dir']
    elif session_source == 'data':
        sess_dir = hparams['data_dir']
    else:
        raise ValueError('"%s" is an invalid session_source' % session_source)

    if len(hparams.get('sessions_csv', [])) > 0:
        # collect all single sessions from csv
        sessions_single = read_session_info_from_csv(hparams['sessions_csv'])
        labs, expts, animals, sessions = [], [], [], []
        for sess in sessions_single:
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
            session_dir_base = os.path.join(save_dir, lab, expt, animal, session)
        elif len(np.unique(animals)) == 1:
            # get all sessions from one animal
            lab, expt, animal = labs[0], expts[0], animals[0]
            session_dir_base = os.path.join(save_dir, lab, expt, animal)
        elif len(np.unique(expts)) == 1:
            lab, expt = labs[0], expts[0]
            # get all animals from one experiment
            session_dir_base = os.path.join(save_dir, lab, expt)
        elif len(np.unique(labs)) == 1:
            # get all experiments from one lab
            lab = labs[0]
            session_dir_base = os.path.join(save_dir, lab)
        else:
            raise NotImplementedError('multiple labs not currently supported')
        # find corresponding multisession (ok if they don't exist)
        multisession_paths = _get_multisession_paths(save_dir, lab=lab, expt=expt, animal=animal)

    else:
        # get session dirs (can include multiple sessions)
        lab = hparams['lab']
        if lab == 'all':
            raise NotImplementedError('multiple labs not currently supported')
        elif hparams['expt'] == 'all':
            # get all experiments from one lab
            multisession_paths = _get_multisession_paths(save_dir, lab=lab)
            sessions_single = _get_single_sessions(
                os.path.join(sess_dir, lab), depth=3, curr_depth=0)
            session_dir_base = os.path.join(save_dir, lab)
        elif hparams['animal'] == 'all':
            # get all animals from one experiment
            expt = hparams['expt']
            multisession_paths = _get_multisession_paths(save_dir, lab=lab, expt=expt)
            sessions_single = _get_single_sessions(
                os.path.join(sess_dir, lab, expt), depth=2, curr_depth=0)
            session_dir_base = os.path.join(save_dir, lab, expt)
        elif hparams['session'] == 'all':
            # get all sessions from one animal
            expt = hparams['expt']
            animal = hparams['animal']
            multisession_paths = _get_multisession_paths(
                save_dir, lab=lab, expt=expt, animal=animal)
            sessions_single = _get_single_sessions(
                os.path.join(sess_dir, lab, expt, animal), depth=1, curr_depth=0)
            session_dir_base = os.path.join(save_dir, lab, expt, animal)
        else:
            multisession_paths = []
            sessions_single = [{
                'lab': hparams['lab'], 'expt': hparams['expt'], 'animal': hparams['animal'],
                'session': hparams['session']}]
            session_dir_base = os.path.join(
                save_dir, hparams['lab'], hparams['expt'], hparams['animal'], hparams['session'])

    # construct session_dir
    if hparams.get('multisession', None) is not None and len(hparams.get('sessions_csv', [])) == 0:
        session_dir = os.path.join(session_dir_base, 'multisession-%02i' % hparams['multisession'])
        # overwrite sessions_single with whatever is in requested multisession
        sessions_single = read_session_info_from_csv(os.path.join(session_dir, 'session_info.csv'))
        for sess in sessions_single:
            sess.pop('save_dir', None)
    elif len(sessions_single) > 1:
        # check if this combo of experiments exists in previous multi-sessions
        found_match = False
        multi_idx = None
        for session_multi in multisession_paths:
            csv_file = os.path.join(session_multi, 'session_info.csv')
            sessions_multi = read_session_info_from_csv(csv_file)
            for d in sessions_multi:
                # save path doesn't matter for comparison
                d.pop('save_dir', None)
            # compare to collection of single sessions above
            set_l1 = set(tuple(sorted(d.items())) for d in sessions_single)
            set_l2 = set(tuple(sorted(d.items())) for d in sessions_multi)
            set_diff = set_l1.symmetric_difference(set_l2)
            if len(set_diff) == 0:
                # found match; record index
                found_match = True
                multi_idx = int(session_multi.split('-')[-1])
                break

        # create new multisession if match was not found
        if not found_match:
            multi_idxs = [
                int(session_multi.split('-')[-1]) for session_multi in multisession_paths]
            if len(multi_idxs) == 0:
                multi_idx = 0
            else:
                multi_idx = max(multi_idxs) + 1
        else:
            pass

        session_dir = os.path.join(session_dir_base, 'multisession-%02i' % multi_idx)
    else:
        session_dir = session_dir_base

    return session_dir, sessions_single


def get_expt_dir(hparams, model_class=None, model_type=None, expt_name=None):
    """Get output directories associated with a particular model class/type/testtube expt name.

    Examples
    --------
    * autoencoder: :obj:`session_dir/ae/conv/08_latents/expt_name`
    * arhmm: :obj:`session_dir/arhmm/08_latents/16_states/stationary/gaussian/expt_name`
    * arhmm-labels: :obj:`session_dir/arhmm-labels/16_states/stationary/gaussian/expt_name`
    * neural->ae decoder: :obj:`session_dir/neural-ae/08_latents/mlp/mctx/expt_name`
    * neural->arhmm decoder:
      :obj:`session_dir/neural-ae/08_latents/16_states/stationary/mlp/mctx/expt_name`
    * bayesian decoder:
      :obj:`session_dir/arhmm-decoding/08_latents/16_states/stationary/gaussian/mctx/expt_name`

    Parameters
    ----------
    hparams : :obj:`dict`
        specify model hyperparameters
    model_class : :obj:`str`, optional
        will search :obj:`hparams` if not present
    model_type : :obj:`str`, optional
        will search :obj:`hparams` if not present
    expt_name : :obj:`str`, optional
        will search :obj:`hparams` if not present

    Returns
    -------
    :obj:`str`
        contains data info (lab/expt/animal/session) as well as model info (e.g. n_ae_latents) and
        expt_name

    """

    import copy

    if model_class is None:
        model_class = hparams['model_class']

    if model_type is None:
        model_type = hparams['model_type']

    if expt_name is None:
        expt_name = hparams['experiment_name']

    # get results dir
    if model_class == 'ae' \
            or model_class == 'vae' \
            or model_class == 'beta-tcvae' \
            or model_class == 'cond-vae' \
            or model_class == 'cond-ae' \
            or model_class == 'cond-ae-msp' \
            or model_class == 'ps-vae' \
            or model_class == 'msps-vae':
        model_path = os.path.join(
            model_class, model_type, '%02i_latents' % hparams['n_ae_latents'])
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
    elif model_class == 'neural-ae' or model_class == 'neural-ae-me' or model_class == 'ae-neural':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'], model_type, brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'neural-labels' or model_class == 'labels-neural':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(model_class, model_type, brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'neural-arhmm' or model_class == 'arhmm-neural':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            _get_transition_str(hparams), model_type, brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'arhmm' or model_class == 'hmm':
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            _get_transition_str(hparams), hparams['noise_type'])
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
    elif model_class == 'arhmm-labels' or model_class == 'hmm-labels':
        model_path = os.path.join(
            model_class, '%02i_states' % hparams['n_arhmm_states'],
            _get_transition_str(hparams), hparams['noise_type'])
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
    elif model_class == 'bayesian-decoding':
        brain_region = get_region_dir(hparams)
        model_path = os.path.join(
            model_class, '%02i_latents' % hparams['n_ae_latents'],
            '%02i_states' % hparams['n_arhmm_states'],
            _get_transition_str(hparams), hparams['noise_type'], brain_region)
        session_dir = hparams['session_dir']
    elif model_class == 'labels-images':
        model_path = os.path.join(model_class, model_type)
        session_dir = hparams['session_dir']
    else:
        raise ValueError('"%s" is an invalid model class' % model_class)
    expt_dir = os.path.join(session_dir, model_path, expt_name)

    return expt_dir


def read_session_info_from_csv(session_file):
    """Read csv file that contains lab/expt/animal/session info.

    Parameters
    ----------
    session_file : :obj:`str`
        /full/path/to/session_info.csv

    Returns
    -------
    :obj:`list` of :obj:`dict`
        dict for each session which contains lab/expt/animal/session

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
    """Export list of sessions to csv file.

    Parameters
    ----------
    session_dir : :obj:`str`
        absolute path for where to save :obj:`session_info.csv` file
    ids_list : :obj:`list` of :obj:`dict`
        dict for each session which contains lab/expt/animal/session

    """
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
    """Determine if session defined by `session_id` dict is in the multi-session `session_dir`.

    Parameters
    ----------
    session_dir : :obj:`str`
        absolute path to multi-session directory that contains a :obj:`session_info.csv` file
    session_id : :obj:`dict`
        must contain keys 'lab', 'expt', 'animal' and 'session'

    Returns
    -------
    :obj:`bool`

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
    """Find all session dirs (single- and multi-session) that contain the session in hparams.

    Parameters
    ----------
    hparams : :obj:`dict`
        must contain keys 'lab', 'expt', 'animal' and 'session'

    Returns
    -------
    :obj:`list` of :obj:`str`
        list of session directories containing session defined in :obj:`hparams`

    """
    # TODO: refactor like get_session_dir?
    ids = {s: hparams[s] for s in ['lab', 'expt', 'animal', 'session']}
    lab = hparams['lab']
    expts = get_subdirs(os.path.join(hparams['save_dir'], lab))
    # need to grab all multi-sessions as well as the single session
    session_dirs = []  # full paths
    session_ids = []  # dict of lab/expt/animal/session
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
            animals = get_subdirs(os.path.join(hparams['save_dir'], lab, expt))
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
                sessions = get_subdirs(os.path.join(hparams['save_dir'], lab, expt, animal))
            for session in sessions:
                session_dir = os.path.join(hparams['save_dir'], lab, expt, animal, session)
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
    """Search testtube versions to find if experiment with the same hyperparameters has been fit.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify a test tube experiment (model + training
        parameters)
    which_version : :obj:`bool`, optional
        :obj:`True` to return version number

    Returns
    -------
    variable
        - :obj:`bool` if :obj:`which_version=False`
        - :obj:`tuple` (:obj:`bool`, :obj:`int`) if :obj:`which_version=True`

    """

    import pickle

    # fill out path info if not present
    if 'expt_dir' not in hparams:
        if 'session_dir' not in hparams:
            hparams['session_dir'], _ = get_session_dir(
                hparams, session_source=hparams.get('all_source', 'save'))
        hparams['expt_dir'] = get_expt_dir(hparams)

    try:
        tt_versions = get_subdirs(hparams['expt_dir'])
    except StopIteration:
        # no versions yet
        if which_version:
            return False, None
        else:
            return False

    # get model-specific params
    hparams_less = get_model_params(hparams)

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
        return found_match, int(version.split('_')[-1])
    elif which_version and not found_match:
        return found_match, None
    else:
        return found_match


def get_model_params(hparams):
    """Returns dict containing all params considered essential for defining a model in that class.

    Parameters
    ----------
    hparams : :obj:`dict`
        all relevant hparams for the given model class will be pulled from this dict

    Returns
    -------
    :obj:`dict`
        hparams dict

    """

    model_class = hparams['model_class']

    # start with general params
    hparams_less = {
        'rng_seed_data': hparams['rng_seed_data'],
        'trial_splits': hparams['trial_splits'],
        'train_frac': hparams['train_frac'],
        'rng_seed_model': hparams['rng_seed_model'],
        'model_class': hparams['model_class'],
        'model_type': hparams['model_type'],
    }

    if model_class == 'ae' \
            or model_class == 'vae' \
            or model_class == 'beta-tcvae' \
            or model_class == 'cond-vae' \
            or model_class == 'cond-ae' \
            or model_class == 'cond-ae-msp' \
            or model_class == 'ps-vae' \
            or model_class == 'msps-vae':
        hparams_less['n_ae_latents'] = hparams['n_ae_latents']
        hparams_less['fit_sess_io_layers'] = hparams['fit_sess_io_layers']
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['l2_reg'] = hparams['l2_reg']
        if model_class == 'cond-ae' or model_class == 'cond-vae':
            hparams_less['conditional_encoder'] = hparams.get('conditional_encoder', False)
        if model_class == 'cond-ae-msp':
            hparams_less['msp.alpha'] = hparams['msp.alpha']
        if model_class == 'vae' or model_class == 'cond-vae':
            hparams_less['vae.beta'] = hparams['vae.beta']
            # hparams_less['vae.beta_anneal_epochs'] = hparams['vae.beta_anneal_epochs']
        if model_class == 'beta-tcvae':
            hparams_less['beta_tcvae.beta'] = hparams['beta_tcvae.beta']
        if model_class == 'ps-vae' or model_class == 'msps-vae':
            hparams_less['ps_vae.alpha'] = hparams['ps_vae.alpha']
            hparams_less['ps_vae.beta'] = hparams['ps_vae.beta']
            if model_class == 'msps-vae':
                hparams_less['ps_vae.delta'] = hparams['ps_vae.delta']
                hparams_less['n_background'] = hparams['n_background']
                hparams_less['n_sessions_per_batch'] = hparams['n_sessions_per_batch']
                # hparams_less['ps_vae.ms_loss'] = hparams['ps_vae.ms_loss']
    elif model_class == 'arhmm' or model_class == 'hmm':
        hparams_less['n_arhmm_lags'] = hparams['n_arhmm_lags']
        hparams_less['noise_type'] = hparams['noise_type']
        hparams_less['transitions'] = hparams['transitions']
        if hparams['transitions'] == 'sticky':
            hparams_less['kappa'] = hparams['kappa']
        hparams_less['ae_experiment_name'] = hparams['ae_experiment_name']
        hparams_less['ae_version'] = hparams['ae_version']
        hparams_less['ae_model_class'] = hparams['ae_model_class']
        hparams_less['ae_model_type'] = hparams['ae_model_type']
        hparams_less['n_ae_latents'] = hparams['n_ae_latents']
    elif model_class == 'arhmm-labels' or model_class == 'hmm-labels':
        hparams_less['n_arhmm_lags'] = hparams['n_arhmm_lags']
        hparams_less['noise_type'] = hparams['noise_type']
        hparams_less['transitions'] = hparams['transitions']
        if hparams['transitions'] == 'sticky':
            hparams_less['kappa'] = hparams['kappa']
    elif model_class == 'neural-ae' or model_class == 'neural-ae-me' or model_class == 'ae-neural':
        hparams_less['ae_experiment_name'] = hparams['ae_experiment_name']
        hparams_less['ae_version'] = hparams['ae_version']
        hparams_less['ae_model_class'] = hparams['ae_model_class']
        hparams_less['ae_model_type'] = hparams['ae_model_type']
        hparams_less['n_ae_latents'] = hparams['n_ae_latents']
    elif model_class == 'neural-labels' or model_class == 'labels-neural':
        pass
    elif model_class == 'neural-arhmm' or model_class == 'arhmm-neural':
        hparams_less['arhmm_experiment_name'] = hparams['arhmm_experiment_name']
        hparams_less['arhmm_version'] = hparams['arhmm_version']
        hparams_less['n_arhmm_states'] = hparams['n_arhmm_states']
        hparams_less['n_arhmm_lags'] = hparams['n_arhmm_lags']
        hparams_less['noise_type'] = hparams['noise_type']
        hparams_less['transitions'] = hparams['transitions']
        if hparams['transitions'] == 'sticky':
            hparams_less['kappa'] = hparams['kappa']
        hparams_less['ae_model_class'] = hparams['ae_model_class']
        hparams_less['ae_model_type'] = hparams['ae_model_type']
        hparams_less['n_ae_latents'] = hparams['n_ae_latents']
    elif model_class == 'bayesian-decoding':
        raise NotImplementedError
    elif model_class == 'labels-images':
        hparams_less['fit_sess_io_layers'] = hparams['fit_sess_io_layers']
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['l2_reg'] = hparams['l2_reg']
    else:
        raise NotImplementedError('"%s" is not a valid model class' % model_class)

    # decoder arch params
    if model_class == 'neural-ae' or model_class == 'neural-ae-me' or model_class == 'ae-neural' \
            or model_class == 'neural-arhmm' or model_class == 'arhmm-neural' \
            or model_class == 'neural-labels' or model_class == 'labels-neural':
        hparams_less['learning_rate'] = hparams['learning_rate']
        hparams_less['n_lags'] = hparams['n_lags']
        hparams_less['l2_reg'] = hparams['l2_reg']
        hparams_less['model_type'] = hparams['model_type']
        hparams_less['n_hid_layers'] = hparams['n_hid_layers']
        if hparams['n_hid_layers'] != 0:
            hparams_less['n_hid_units'] = hparams['n_hid_units']
        hparams_less['activation'] = hparams['activation']
        hparams_less['subsample_method'] = hparams['subsample_method']
        if hparams_less['subsample_method'] != 'none':
            hparams_less['subsample_idxs_name'] = hparams['subsample_idxs_name']
            hparams_less['subsample_idxs_group_0'] = hparams['subsample_idxs_group_0']
            hparams_less['subsample_idxs_group_1'] = hparams['subsample_idxs_group_1']

    return hparams_less


def export_hparams(hparams, exp):
    """Export hyperparameter dictionary.

    The dict is export once as a csv file (for easy human reading) and again as a pickled dict
    (for easy python loading/parsing).

    Parameters
    ----------
    hparams : :obj:`dict`
        hyperparameter dict to export
    exp : :obj:`test_tube.Experiment` object
        defines where parameters are saved

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
    """Helper function to load data-specific hyperparameters and update hparams.

    These values are loaded from the json file defined by :obj:`lab` and :obj:`expt` in the
    :obj:`.behavenet` user directory. See
    https://behavenet.readthedocs.io/en/latest/source/installation.html#adding-a-new-dataset
    for more information.

    Parameters
    ----------
    hparams : :obj:`dict`
        hyperparmeter dict to update
    lab : :obj:`str`
        lab id
    expt : :obj:`str`
        expt id

    """
    import json
    from behavenet import get_params_dir
    params_file = os.path.join(get_params_dir(), str('%s_%s_params.json' % (lab, expt)))
    with open(params_file, 'r') as f:
        dparams = json.load(f)
    hparams.update(dparams)


def get_region_dir(hparams):
    """Return brain region string that combines region name and inclusion info.

    If not subsampling regions, will return :obj:`'all'`

    If using neural activity from *only* specified region, will return e.g. :obj:`'mctx-single'`

    If using neural activity from all *but* specified region (leave-one-out), will return e.g.
    :obj:`'mctx-loo'`

    Parameters
    ----------
    hparams : :obj:`dict`
        must contain the key 'subsample_regions', else function assumes no subsampling

    Returns
    -------
    :obj:`str`
        region directory name

    """
    if hparams.get('subsample_method', 'none') == 'none':
        region_dir = 'all'
    elif hparams['subsample_method'] == 'single':
        region_dir = str('%s-single' % hparams['subsample_idxs_name'])
    elif hparams['subsample_method'] == 'loo':
        region_dir = str('%s-loo' % hparams['subsample_idxs_name'])
    else:
        raise ValueError('"%s" is an invalid sampling type' % hparams['subsample_method'])
    return region_dir


def create_tt_experiment(hparams):
    """Create test-tube experiment for logging training and storing models.

    Parameters
    ----------
    hparams : :obj:`dict`
        dictionary of hyperparameters defining experiment that will be saved as a csv file

    Returns
    -------
    :obj:`tuple`
        - if experiment defined by hparams already exists, returns :obj:`(None, None, None)`
        - if experiment does not exist, returns :obj:`(hparams, sess_ids, exp)`

    """
    from test_tube import Experiment

    # get session_dir
    hparams['session_dir'], sess_ids = get_session_dir(
        hparams, session_source=hparams.get('all_source', 'save'))
    if not os.path.isdir(hparams['session_dir']):
        os.makedirs(hparams['session_dir'])
        export_session_info_to_csv(hparams['session_dir'], sess_ids)
    hparams['expt_dir'] = get_expt_dir(hparams)
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])

    # check to see if experiment already exists
    if experiment_exists(hparams):
        return None, None, None

    exp = Experiment(
        name=hparams['experiment_name'],
        debug=False,
        save_dir=os.path.dirname(hparams['expt_dir']))
    exp.save()
    hparams['version'] = exp.version

    return hparams, sess_ids, exp


def get_best_model_version(expt_dir, measure='val_loss', best_def='min', n_best=1):
    """Get best model version from a test tube experiment.

    Parameters
    ----------
    expt_dir : :obj:`str`
        test tube experiment directory containing version_%i subdirectories
    measure : :obj:`str`, optional
        heading in csv file that is used to determine which model is best
    best_def : :obj:`str`, optional
        how :obj:`measure` should be parsed; 'min' | 'max'
    n_best : :obj:`int`, optional
        top `n_best` models are returned

    Returns
    -------
    :obj:`list`
        list of best models, with best first

    """
    import pickle
    import pandas as pd
    # gather all versions
    versions = get_subdirs(expt_dir)
    # load csv files with model metrics (saved out from test tube)
    metrics = []
    for i, version in enumerate(versions):
        # make sure training has been completed
        meta_file = os.path.join(expt_dir, version, 'meta_tags.pkl')
        if not os.path.exists(meta_file):
            continue
        with open(meta_file, 'rb') as f:
            meta_tags = pickle.load(f)
        if not meta_tags['training_completed']:
            continue
        # read metrics csv file
        metric = pd.read_csv(os.path.join(expt_dir, version, 'metrics.csv'))

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
    # convert string to integer
    best_versions = [int(version.split('_')[-1]) for version in best_versions]
    return best_versions


def get_best_model_and_data(hparams, Model=None, load_data=True, version='best', data_kwargs=None):
    """Load the best model (and data) defined by hparams out of all available test-tube versions.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify both a model and the associated data
    Model : :obj:`behavenet.models` object, optional
        model type
    load_data : :obj:`bool`, optional
        if `False` then data generator is not returned
    version : :obj:`str` or :obj:`int`, optional
        can be 'best' to load best model
    data_kwargs : :obj:`dict`, optional
        additional kwargs for data generator

    Returns
    -------
    :obj:`tuple`
        - model (:obj:`behavenet.models` object)
        - data generator (:obj:`ConcatSessionsGenerator` object or :obj:`NoneType`)

    """

    import torch
    from behavenet.data.data_generator import ConcatSessionsGenerator
    from behavenet.data.utils import get_data_generator_inputs

    # get session_dir
    hparams['session_dir'], sess_ids = get_session_dir(
        hparams, session_source=hparams.get('all_source', 'save'))
    expt_dir = get_expt_dir(hparams)

    # get best model version
    if version == 'best':
        best_version_int = get_best_model_version(expt_dir)[0]
        best_version = str('version_{}'.format(best_version_int))
    elif version is None:
        # try to match hparams
        _, version_hp = experiment_exists(hparams, which_version=True)
        best_version = str('version_{}'.format(version_hp))
    else:
        if isinstance(version, str) and version[0] == 'v':
            # assume we got a string of the form 'version_{%i}'
            best_version = version
        else:
            best_version = str('version_{}'.format(version))
    # get int representation as well
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
    hparams_new['use_label_mask'] = hparams.get('use_label_mask', False)
    hparams_new['device'] = hparams.get('device', 'cpu')

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
            batch_load=hparams_new['batch_load'], rng_seed=hparams_new['rng_seed_data'],
            train_frac=hparams_new['train_frac'], **data_kwargs)
    else:
        data_generator = None

    # build model
    if Model is None:
        if hparams['model_class'] == 'ae':
            from behavenet.models import AE as Model
        elif hparams['model_class'] == 'vae':
            from behavenet.models import VAE as Model
        elif hparams['model_class'] == 'cond-ae':
            from behavenet.models import ConditionalAE as Model
        elif hparams['model_class'] == 'cond-vae':
            from behavenet.models import ConditionalVAE as Model
        elif hparams['model_class'] == 'cond-ae-msp':
            from behavenet.models import AEMSP as Model
        elif hparams['model_class'] == 'beta-tcvae':
            from behavenet.models import BetaTCVAE as Model
        elif hparams['model_class'] == 'ps-vae':
            from behavenet.models import PSVAE as Model
        elif hparams['model_class'] == 'msps-vae':
            from behavenet.models import MSPSVAE as Model
        elif hparams['model_class'] == 'labels-images':
            from behavenet.models import ConvDecoder as Model
        elif hparams['model_class'] == 'neural-ae' or hparams['model_class'] == 'neural-ae-me' \
                or hparams['model_class'] == 'neural-arhmm' \
                or hparams['model_class'] == 'neural-labels':
            from behavenet.models import Decoder as Model
        elif hparams['model_class'] == 'ae-neural' or hparams['model_class'] == 'arhmm-neural' \
                or hparams['model_class'] == 'labels-neural':
            from behavenet.models import Decoder as Model
        elif hparams['model_class'] == 'arhmm':
            raise NotImplementedError('Cannot use get_best_model_and_data() for ssm models')
        else:
            raise NotImplementedError

    model = Model(hparams_new)
    model.version = int(best_version.split('_')[1])
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    model.to(hparams_new['device'])
    model.eval()

    return model, data_generator


def _clean_tt_dir(hparams):
    """Delete all (unnecessary) subdirectories in the model directory (created test-tube)"""
    import shutil
    # get subdirs
    version_dir = os.path.join(hparams['expt_dir'], 'version_%i' % hparams['version'])
    subdirs = get_subdirs(version_dir)
    for subdir in subdirs:
        shutil.rmtree(os.path.join(version_dir, subdir))


def _print_hparams(hparams):
    """Pretty print hparams to console."""
    import commentjson
    config_files = ['data', 'compute', 'training', 'model']
    for config_file in config_files:
        print('\n%s CONFIG:' % config_file.upper())
        config_json = commentjson.load(open(hparams['%s_config' % config_file], 'r'))
        for key in config_json.keys():
            print('    {}: {}'.format(key, hparams[key]))
    print('')
