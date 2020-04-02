import argparse
import commentjson
import copy
import json
import os
import shutil
import subprocess
import time
from behavenet import get_params_dir
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

"""
TODO:
    - how to print traceback when testtube fails?
    - multisessions
    - other models (ae/arhmm-neural, arhmm w/ labels, labels->images decoder
"""


def get_model_config_files(model, json_dir):
    if model == 'ae' or model == 'arhmm':
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


def define_new_config_values(model):

    # data vals
    train_frac = 0.1
    trial_splits = '10;1;1;5'
    gpu_id = 0

    # ae vals
    ae_expt_name = 'ae-expt'
    ae_model_type = 'conv'
    n_ae_latents = 6

    # arhmm vals
    arhmm_expt_name = 'arhmm-expt'
    n_arhmm_states = [2, 4]
    n_arhmm_lags = 1
    kappa = 0
    noise_type = 'gaussian'

    if model == 'ae':
        new_values = {
            'data': {},
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
    elif model == 'arhmm':
        new_values = {
            'data': {},
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
            'data': {},
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
            'data': {},
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


def check_model(config_dicts, save_dir):
    hparams = {
        **config_dicts['data'], **config_dicts['model'], **config_dicts['training'],
        **config_dicts['compute']}
    hparams['save_dir'] = save_dir
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

    # create temp dir to store outputs
    if os.path.exists(args.save_dir):
        args.save_dir += '_tmp'
    os.mkdir(args.save_dir)

    # update directories to include new temp dir
    dirs_file = os.path.join(get_params_dir(), 'directories.json')
    dirs_old = json.load(open(dirs_file, 'r'))
    dirs_new = copy.deepcopy(dirs_old)
    dirs_new['save_dir'] = args.save_dir
    json.dump(dirs_new, open(dirs_file, 'w'))

    json_dir = os.path.join(os.getcwd(), 'behavenet', 'json_configs')
    fitting_dir = os.path.join(os.getcwd(), 'behavenet', 'fitting')

    # store results of tests
    print_strs = {}

    # -------------------------------------------
    # fit models
    # -------------------------------------------
    model_classes = ['ae', 'arhmm', 'neural-ae', 'neural-arhmm']
    model_files = ['ae', 'arhmm', 'decoder', 'decoder']
    # model_classes = ['ae']
    # model_files = ['ae']
    for model_class, model_file in zip(model_classes, model_files):
        # modify example jsons
        base_config_files = get_model_config_files(model_class, json_dir)
        new_values = define_new_config_values(model_class)
        config_dicts, new_config_files = update_config_files(
            base_config_files, new_values, args.save_dir)
        # fit model
        fit_model(model_file, fitting_dir, new_config_files)
        # check model
        print_strs[model_class] = check_model(config_dicts, args.save_dir)

    # -------------------------------------------
    # clean up
    # -------------------------------------------
    # restore old directories
    json.dump(dirs_old, open(dirs_file, 'w'))

    # remove temp dir
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

    from behavenet import get_user_dir

    # data directory default
    data_default = get_user_dir('data')

    # save directory default (will be deleted at end of test)
    save_default = os.path.join(get_user_dir('save'), 'tmp_int_test')

    # parse command line args and send to main test function
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=data_default, type=str)
    parser.add_argument('--save_dir', default=save_default, type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    namespace, _ = parser.parse_known_args()
    main(namespace)
