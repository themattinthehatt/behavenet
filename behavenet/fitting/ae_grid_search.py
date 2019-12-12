import os
import time
import numpy as np
import random
import torch

from behavenet.fitting.eval import export_latents_best
from behavenet.fitting.eval import export_train_plots
from behavenet.fitting.training import fit
from behavenet.fitting.utils import build_data_generator
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import _clean_tt_dir
from behavenet.fitting.utils import export_hparams
from behavenet.models import AE as AE
from behavenet.fitting.hyperparam_utils import get_all_params


def main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    if hparams['model_type'] == 'conv':
        # blend outer hparams with architecture hparams
        hparams = {**hparams, **hparams['architecture_params']}
    print('\nexperiment parameters:')
    print(hparams['data_config'])

    if hparams['model_type'] == 'conv' and hparams['n_ae_latents'] > hparams['max_latents']:
         raise ValueError('Number of latents higher than max latents, architecture will not work')

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

    # ####################
    # ### CREATE MODEL ###
    # ####################

    print('constructing model...', end='')
    torch.manual_seed(hparams['rng_seed_model'])
    torch_rnd_seed = torch.get_rng_state()
    hparams['model_build_rnd_seed'] = torch_rnd_seed
    hparams['n_datasets'] = len(sess_ids)
    model = AE(hparams)
    model.to(hparams['device'])
    model.version = exp.version
    torch_rnd_seed = torch.get_rng_state()
    hparams['training_rnd_seed'] = torch_rnd_seed

    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    fit(hparams, model, data_generator, exp, method='ae')

    # export training plots
    if hparams['export_train_plots']:
        print('creating training plots...', end='')
        version_dir = os.path.join(hparams['expt_dir'], 'version_%i' % hparams['version'])
        save_file = os.path.join(version_dir, 'loss_training')
        export_train_plots(hparams, 'train', save_file=save_file)
        save_file = os.path.join(version_dir, 'loss_validation')
        export_train_plots(hparams, 'val', save_file=save_file)
        print('done')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)

    # get rid of unneeded logging info
    _clean_tt_dir(hparams)


if __name__ == '__main__':

    hyperparams = get_all_params('grid_search')

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

    if hyperparams.export_latents_best:
        print('Exporting latents from current best model in experiment')
        export_latents_best(vars(hyperparams))
