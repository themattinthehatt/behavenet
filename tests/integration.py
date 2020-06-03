import argparse
import commentjson
import h5py
import json
import numpy as np
import os
import shutil
import subprocess
import time
from behavenet.fitting.utils import experiment_exists


# https://stackoverflow.com/a/39452138
CEND = '\33[0m'
BOLD = '\033[1m'
CBLACK = '\33[30m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'

DATA_DICT = {
    'lab': 'lab',
    'expt': 'expt',
    'animal': 'animal',
    'all_source': 'data',
    'n_input_channels': 1,
    'y_pixels': 64,
    'x_pixels': 48,
    'use_output_mask': False,
    'neural_bin_size': 25,
    'neural_type': 'ca',
    'approx_batch_size': 200
}
TEMP_DATA = {
    'n_batches': 22,
    'batch_lens': [20, 100],  # [min, max] of random uniform int
    'n_labels': 8,
    'n_neurons': 25
}
SESSIONS = ['sess-0', 'sess-1']

"""
TODO:
    - how to print traceback when testtube fails?
    - arhmm multisessions
    - other models (ae/arhmm-neural, arhmm w/ labels, labels->images decoder
"""


def make_tmp_data(data_dir):
    """Make hdf5 file with images, labels, and neural activity."""

    for session in SESSIONS:

        hdf5_file = os.path.join(
            data_dir, DATA_DICT['lab'], DATA_DICT['expt'], DATA_DICT['animal'], session,
            'data.hdf5')
        os.makedirs(os.path.dirname(hdf5_file))

        with h5py.File(hdf5_file, 'w', libver='latest', swmr=True) as f:

            f.swmr_mode = True  # single write multi-read

            # create image group
            group_i = f.create_group('images')
            # create neural data group
            group_n = f.create_group('neural')
            # create labels group
            group_l = f.create_group('labels')
            # createregion indices group
            group_r = f.create_group('regions')
            # add region indices data
            group_ri = group_r.create_group('indxs')
            group_ri.create_dataset('region-0', data=np.arange(10))
            group_ri.create_dataset('region-1', data=10 + np.arange(15))

            # create a dataset for each trial within groups
            for i in range(TEMP_DATA['n_batches']):

                batch_len = np.random.randint(
                    TEMP_DATA['batch_lens'][0], TEMP_DATA['batch_lens'][1])

                # image data
                image_size = (
                    batch_len, DATA_DICT['n_input_channels'], DATA_DICT['y_pixels'],
                    DATA_DICT['x_pixels'])
                batch_i = np.random.randint(0, 255, size=image_size)
                group_i.create_dataset('trial_%04i' % i, data=batch_i, dtype='uint8')

                # neural data
                batch_n = np.random.randn(batch_len, TEMP_DATA['n_neurons'])
                group_n.create_dataset('trial_%04i' % i, data=batch_n, dtype='float32')

                # label data
                batch_l = np.random.randn(batch_len, TEMP_DATA['n_labels'])
                group_l.create_dataset('trial_%04i' % i, data=batch_l, dtype='float32')


def get_model_config_files(model, json_dir):
    if model == 'ae' or model == 'arhmm' or model == 'cond-ae-msp':
        if model == 'cond-ae-msp':
            model = 'ae'
        model_json_dir = os.path.join(json_dir, '%s_jsons' % model)
        base_config_files = {
            'data': os.path.join(json_dir, 'data_default.json'),
            'model': os.path.join(model_json_dir, '%s_model.json' % model),
            'training': os.path.join(model_json_dir, '%s_training.json' % model),
            'compute': os.path.join(model_json_dir, '%s_compute.json' % model)}
    elif model == 'neural-ae' or model == 'neural-arhmm':
        m = 'decoding'
        s = model.split('-')[-1]
        model_json_dir = os.path.join(json_dir, '%s_jsons' % m)
        base_config_files = {
            'data': os.path.join(model_json_dir, '%s_data.json' % m),
            'model': os.path.join(model_json_dir, '%s_%s_model.json' % (m, s)),
            'training': os.path.join(model_json_dir, '%s_training.json' % m),
            'compute': os.path.join(model_json_dir, '%s_compute.json' % m)}
    else:
        raise NotImplementedError
    return base_config_files


def define_new_config_values(model, session='sess-0'):

    # data vals
    data_dict = {'session': session, 'all_source': 'data', **DATA_DICT}

    # training vals
    train_frac = 0.5
    trial_splits = '8;1;1;1'

    # compute vals
    gpu_id = 0

    # model vals: ae
    ae_expt_name = 'ae-expt'
    ae_model_type = 'conv'
    n_ae_latents = 6

    # model vals: arhmm
    arhmm_expt_name = 'arhmm-expt'
    n_arhmm_states = [2, 4]
    n_arhmm_lags = 1
    kappa = 0
    noise_type = 'gaussian'

    if model == 'ae':
        new_values = {
            'data': data_dict,
            'model': {
                'experiment_name': ae_expt_name,
                'model_type': ae_model_type,
                'n_ae_latents': n_ae_latents,
                'l2_reg': 0.0},
            'training': {
                'export_train_plots': False,
                'export_latents': True,
                'min_n_epochs': 1,
                'max_n_epochs': 1,
                'enable_early_stop': False,
                'train_frac': train_frac,
                'trial_splits': trial_splits},
            'compute': {
                'gpus_viz': str(gpu_id)}}
    elif model == 'cond-ae-msp':
        new_values = {
            'data': data_dict,
            'model': {
                'experiment_name': ae_expt_name,
                'model_type': ae_model_type,
                'n_ae_latents': n_ae_latents + TEMP_DATA['n_labels'],
                'l2_reg': 0.0,
                'msp_weight': 1e-5},
            'training': {
                'export_train_plots': False,
                'export_latents': True,
                'min_n_epochs': 1,
                'max_n_epochs': 1,
                'enable_early_stop': False,
                'train_frac': train_frac,
                'trial_splits': trial_splits},
            'compute': {
                'gpus_viz': str(gpu_id)}}
    elif model == 'arhmm':
        new_values = {
            'data': data_dict,
            'model': {
                'experiment_name': arhmm_expt_name,
                'n_arhmm_states': n_arhmm_states,
                'n_arhmm_lags': n_arhmm_lags,
                'kappa': kappa,
                'noise_type': noise_type,
                'ae_experiment_name': ae_expt_name,
                'ae_model_type': ae_model_type,
                'n_ae_latents': n_ae_latents},
            'training': {
                'export_train_plots': False,
                'export_states': True,
                'n_iters': 2,
                'train_frac': train_frac,
                'trial_splits': trial_splits},
            'compute': {
                'gpus_viz': str(gpu_id),
                'tt_n_cpu_workers': 2}}
    elif model == 'neural-ae':
        new_values = {
            'data': data_dict,
            'model': {
                'n_lags': 4,
                'n_max_lags': 8,
                'l2_reg': 1e-3,
                'ae_experiment_name': ae_expt_name,
                'ae_model_type': ae_model_type,
                'n_ae_latents': n_ae_latents,
                'model_type': 'ff',
                'n_hid_layers': 1,
                'n_hid_units': 16,
                'activation': 'relu'},
            'training': {
                'export_predictions': True,
                'min_n_epochs': 1,
                'max_n_epochs': 1,
                'enable_early_stop': False,
                'train_frac': train_frac,
                'trial_splits': trial_splits},
            'compute': {
                'gpus_viz': str(gpu_id),
                'tt_n_cpu_workers': 2}}
    elif model == 'neural-arhmm':
        new_values = {
            'data': data_dict,
            'model': {
                'n_lags': 2,
                'n_max_lags': 8,
                'l2_reg': 1e-3,
                'ae_model_type': ae_model_type,
                'n_ae_latents': n_ae_latents,
                'arhmm_experiment_name': arhmm_expt_name,
                'n_arhmm_states': n_arhmm_states[0],
                'n_arhmm_lags': n_arhmm_lags,
                'kappa': kappa,
                'noise_type': noise_type,
                'model_type': 'ff',
                'n_hid_layers': 1,
                'n_hid_units': [8, 16],
                'activation': 'relu'},
            'training': {
                'export_predictions': True,
                'min_n_epochs': 1,
                'max_n_epochs': 1,
                'enable_early_stop': False,
                'train_frac': train_frac,
                'trial_splits': trial_splits},
            'compute': {
                'gpus_viz': str(gpu_id),
                'tt_n_cpu_workers': 2}}
    else:
        raise NotImplementedError

    return new_values


def update_config_files(config_files, new_values, save_dir=None):
    """

    Parameters
    ----------
    config_files : :obj:`dict`
        absolute paths to base config files
    new_values : :obj:`dict` of :obj:`dict`
        keys correspond to those in :obj:`config_files`; values are dicts with key-value pairs
        defining which keys in the config file are updated with which values
    save_dir : :obj:`str` or :obj:`NoneType`, optional
        if not None, directory in which to save updated config files; filename will be same as
        corresponding base json

    Returns
    -------
    :obj:`tuple`
        (updated config dicts, updated config files)

    """
    new_config_dicts = {}
    new_config_files = {}
    for config_name, config_file in config_files.items():
        # load base config file into dict
        config_dict = commentjson.load(open(config_file, 'r'))
        # change key/value pairs
        for key, val in new_values[config_name].items():
            config_dict[key] = val
        new_config_dicts[config_name] = config_dict
        # save as new config file in save_dir
        if save_dir is not None:
            filename = os.path.join(save_dir, os.path.basename(config_file))
            new_config_files[config_name] = filename
            json.dump(config_dict, open(filename, 'w'))
    return new_config_dicts, new_config_files


def get_call_str(model, fitting_dir, config_files):
    call_str = [
        'python',
        os.path.join(fitting_dir, '%s_grid_search.py' % model),
        '--data_config', config_files['data'],
        '--model_config', config_files['model'],
        '--training_config', config_files['training'],
        '--compute_config', config_files['compute']]
    return call_str


def fit_model(model, fitting_dir, config_files):
    call_str = get_call_str(model, fitting_dir, config_files)
    try:
        subprocess.call(' '.join(call_str), shell=True)
        result_str = BOLD + CGREEN + 'passed' + CEND
    except BaseException as error:
        result_str = BOLD + CRED + 'failed: %s' % str(error) + CEND
    return result_str


def check_model(config_dicts, dirs):
    hparams = {
        **config_dicts['data'], **config_dicts['model'], **config_dicts['training'],
        **config_dicts['compute']}
    hparams['save_dir'] = dirs.save_dir
    hparams['data_dir'] = dirs.data_dir
    # pick out single model if multiple were fit with test tube
    for key, val in hparams.items():
        if isinstance(val, list):
            hparams[key] = val[-1]
    exists = experiment_exists(hparams)
    if exists:
        result_str = BOLD + CGREEN + 'passed' + CEND
    else:
        result_str = BOLD + CRED + 'failed' + CEND
    return result_str


def main(args):
    """Integration testing function.

    Must call from main behavenet directory as:
    $: python tests/integration.py

    """

    t_beg = time.time()

    # -------------------------------------------
    # setup
    # -------------------------------------------

    # create temp dir to store data
    if os.path.exists(args.data_dir):
        args.data_dir += '_tmp_data'
    os.mkdir(args.data_dir)

    # make temp data
    print('creating temp data...', end='')
    make_tmp_data(args.data_dir)
    print('done')

    # create temp dir to store results
    if os.path.exists(args.save_dir):
        args.save_dir += '_tmp_save'
    os.mkdir(args.save_dir)

    # update directories to include new temp dirs
    dirs_file = os.path.join(get_params_dir(), 'directories.json')
    dirs_old = json.load(open(dirs_file, 'r'))
    dirs_new = {'data_dir': args.data_dir, 'save_dir': args.save_dir}
    json.dump(dirs_new, open(dirs_file, 'w'))

    json_dir = os.path.join(os.getcwd(), 'behavenet', 'json_configs')
    fitting_dir = os.path.join(os.getcwd(), 'behavenet', 'fitting')

    # store results of tests
    print_strs = {}

    # -------------------------------------------
    # fit models
    # -------------------------------------------
    model_classes = ['ae', 'arhmm', 'neural-ae', 'neural-arhmm', 'ae', 'cond-ae-msp']
    model_files = ['ae', 'arhmm', 'decoder', 'decoder', 'ae', 'ae']  # %s_grid_search
    sessions = [SESSIONS[0], SESSIONS[0], SESSIONS[0], SESSIONS[0], 'all', SESSIONS[0]]
    for model_class, model_file, session in zip(model_classes, model_files, sessions):
        # modify example jsons
        base_config_files = get_model_config_files(model_class, json_dir)
        new_values = define_new_config_values(model_class, session)
        config_dicts, new_config_files = update_config_files(
            base_config_files, new_values, args.save_dir)
        # fit model
        print('\n\n---------------------------------------------------')
        print('model: %s' % model_class)
        print('session: %s' % session)
        print('---------------------------------------------------\n\n')
        fit_model(model_file, fitting_dir, new_config_files)
        # check model
        if session == 'all':
            model_key = '%s-multisession' % model_class
        else:
            model_key = model_class
        print_strs[model_key] = check_model(config_dicts, args)

    # -------------------------------------------
    # clean up
    # -------------------------------------------
    # restore old directories
    json.dump(dirs_old, open(dirs_file, 'w'))

    # remove temp dirs
    shutil.rmtree(args.data_dir)
    shutil.rmtree(args.save_dir)

    # -------------------------------------------
    # print results
    # -------------------------------------------
    print('\n%s================== Integration Test Results ==================%s\n' % (BOLD, CEND))
    for key, val in print_strs.items():
        print('%s: %s' % (key, val))

    t_end = time.time()
    print('\ntotal time to perform integration test: %s%f sec%s\n' % (BOLD, t_end - t_beg, CEND))


if __name__ == '__main__':

    from behavenet import get_params_dir

    # temp data/results directory
    dir_default = get_params_dir()

    # parse command line args and send to main test function
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=dir_default, type=str)
    parser.add_argument('--save_dir', default=dir_default, type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    namespace, _ = parser.parse_known_args()
    main(namespace)
