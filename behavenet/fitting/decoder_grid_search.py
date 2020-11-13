import os
import time
import numpy as np
import random
import torch
import pickle

from behavenet.data.utils import build_data_generator
from behavenet.fitting.hyperparam_utils import get_all_params
from behavenet.fitting.hyperparam_utils import get_slurm_params
from behavenet.fitting.training import fit
from behavenet.fitting.utils import _clean_tt_dir
from behavenet.fitting.utils import _print_hparams
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet.models import Decoder


def main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    # print hparams to console
    _print_hparams(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(1))

    # create test-tube experiment
    hparams, sess_ids, exp = create_tt_experiment(hparams)
    if hparams is None:
        print('Experiment exists! Aborting fit')
        return

    # build data generator
    data_generator = build_data_generator(hparams, sess_ids)

    ex_trial = data_generator.datasets[0].batch_idxs['train'][0]
    i_sig = hparams['input_signal']
    o_sig = hparams['output_signal']

    if hparams['model_class'] == 'neural-arhmm':
        hparams['input_size'] = data_generator.datasets[0][ex_trial][i_sig].shape[1]
        hparams['output_size'] = hparams['n_arhmm_states']
    elif hparams['model_class'] == 'arhmm-neural':
        hparams['input_size'] = hparams['n_arhmm_states']
        hparams['output_size'] = data_generator.datasets[0][ex_trial][o_sig].shape[1]
    elif hparams['model_class'] == 'neural-ae':
        hparams['input_size'] = data_generator.datasets[0][ex_trial][i_sig].shape[1]
        hparams['output_size'] = hparams['n_ae_latents']
    elif hparams['model_class'] == 'neural-ae-me':
        hparams['input_size'] = data_generator.datasets[0][ex_trial][i_sig].shape[1]
        hparams['output_size'] = hparams['n_ae_latents']
    elif hparams['model_class'] == 'ae-neural':
        hparams['input_size'] = hparams['n_ae_latents']
        hparams['output_size'] = data_generator.datasets[0][ex_trial][o_sig].shape[1]
    elif hparams['model_class'] == 'neural-labels':
        hparams['input_size'] = data_generator.datasets[0][ex_trial][i_sig].shape[1]
        hparams['output_size'] = hparams['n_labels']
    elif hparams['model_class'] == 'labels-neural':
        hparams['input_size'] = hparams['n_labels']
        hparams['output_size'] = data_generator.datasets[0][ex_trial][o_sig].shape[1]
    else:
        raise ValueError('%s is an invalid model class' % hparams['model_class'])

    if hparams['model_class'] == 'neural-ae' or hparams['model_class'] == 'neural-ae' \
            or hparams['model_class'] == 'ae-neural':
        hparams['ae_model_path'] = os.path.join(
            os.path.dirname(data_generator.datasets[0].paths['ae_latents']))
        hparams['ae_model_latents_file'] = data_generator.datasets[0].paths['ae_latents']
    elif hparams['model_class'] == 'neural-arhmm' or hparams['model_class'] == 'arhmm-neural':
        hparams['arhmm_model_path'] = os.path.dirname(
            data_generator.datasets[0].paths['arhmm_states'])
        hparams['arhmm_model_states_file'] = data_generator.datasets[0].paths['arhmm_states']

        # Store which AE was used for the ARHMM
        tags = pickle.load(open(os.path.join(hparams['arhmm_model_path'], 'meta_tags.pkl'), 'rb'))
        hparams['ae_model_latents_file'] = tags['ae_model_latents_file']

    # ####################
    # ### CREATE MODEL ###
    # ####################
    print('constructing model...', end='')
    torch.manual_seed(hparams['rng_seed_model'])
    torch_rng_seed = torch.get_rng_state()
    hparams['model_build_rng_seed'] = torch_rng_seed
    model = Decoder(hparams)
    model.to(hparams['device'])
    model.version = exp.version
    torch_rng_seed = torch.get_rng_state()
    hparams['training_rng_seed'] = torch_rng_seed

    # save out hparams as csv and dict for easy reloading
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    fit(hparams, model, data_generator, exp, method='nll')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)

    # get rid of unneeded logging info
    _clean_tt_dir(hparams)


if __name__ == '__main__':

    hyperparams = get_all_params('grid_search')

    if 'slurm' in hyperparams and hyperparams.slurm:

        cluster = get_slurm_params(hyperparams)

        if hyperparams.device == 'cuda' or hyperparams.device == 'gpu':
            cluster.optimize_parallel_cluster_gpu(
                main, hyperparams.tt_n_cpu_trials, hyperparams.experiment_name,
                job_display_name=None)

        elif hyperparams.device == 'cpu':
            cluster.optimize_parallel_cluster_cpu(
                main, hyperparams.tt_n_cpu_trials, hyperparams.experiment_name,
                job_display_name=None)

    else:
        if hyperparams.device == 'cuda' or hyperparams.device == 'gpu':
            if hyperparams.device == 'gpu':
                hyperparams.device = 'cuda'

            gpu_ids = hyperparams.gpus_viz.split(';')
            hyperparams.optimize_parallel_gpu(main, gpu_ids=gpu_ids)

        elif hyperparams.device == 'cpu':
            hyperparams.optimize_parallel_cpu(
                main,
                nb_trials=hyperparams.tt_n_cpu_trials,
                nb_workers=hyperparams.tt_n_cpu_workers)
