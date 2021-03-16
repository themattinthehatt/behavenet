import os
import time
import numpy as np
import random
import torch
import math

from behavenet.data.utils import build_data_generator
from behavenet.fitting.eval import export_train_plots
from behavenet.fitting.hyperparam_utils import get_all_params
from behavenet.fitting.hyperparam_utils import get_slurm_params
from behavenet.fitting.training import fit
from behavenet.fitting.utils import _clean_tt_dir
from behavenet.fitting.utils import _print_hparams
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet.models.aes import load_pretrained_ae


def main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    if hparams['model_type'] == 'conv':
        # blend outer hparams with architecture hparams
        hparams = {**hparams['architecture_params'], **hparams}

    # print hparams to console
    _print_hparams(hparams)

    if hparams['model_type'] == 'conv' and hparams['n_ae_latents'] > hparams['max_latents']:
        raise ValueError('Number of latents higher than max latents, architecture will not work')

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(3))

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

    def set_n_labels(data_generator, hparams):
        data, _ = data_generator.next_batch('val')
        sh = data['labels'].shape
        hparams['n_labels'] = sh[2]  # [1, n_t, n_labels]

    print('constructing model...', end='')
    torch.manual_seed(hparams['rng_seed_model'])
    torch_rng_seed = torch.get_rng_state()
    hparams['model_build_rng_seed'] = torch_rng_seed
    hparams['n_datasets'] = len(sess_ids)
    if hparams['model_class'] == 'ae':
        from behavenet.models import AE as Model
    elif hparams['model_class'] == 'vae':
        from behavenet.models import VAE as Model
    elif hparams['model_class'] == 'beta-tcvae':
        from behavenet.models import BetaTCVAE as Model
    elif hparams['model_class'] == 'ps-vae':
        from behavenet.models import PSVAE as Model
        set_n_labels(data_generator, hparams)
    elif hparams['model_class'] == 'msps-vae':
        from behavenet.models import MSPSVAE as Model
        set_n_labels(data_generator, hparams)
    elif hparams['model_class'] == 'cond-vae':
        from behavenet.models import ConditionalVAE as Model
        set_n_labels(data_generator, hparams)
    elif hparams['model_class'] == 'cond-ae':
        from behavenet.models import ConditionalAE as Model
        set_n_labels(data_generator, hparams)
    elif hparams['model_class'] == 'cond-ae-msp':
        from behavenet.models import AEMSP as Model
        set_n_labels(data_generator, hparams)
    else:
        raise NotImplementedError(
            'The model class "%s" is not currently implemented' % hparams['model_class'])
    model = Model(hparams)
    model.to(hparams['device'])

    # load pretrained weights if specified
    model = load_pretrained_ae(model, hparams)

    # Parallelize over gpus if desired
    if hparams['n_parallel_gpus'] > 1:
        from behavenet.models import CustomDataParallel
        model = CustomDataParallel(model)

    model.version = exp.version
    torch_rng_seed = torch.get_rng_state()
    hparams['training_rng_seed'] = torch_rng_seed

    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

    # ###################
    # ### TRAIN MODEL ###
    # ###################

    print(model)

    fit(hparams, model, data_generator, exp, method='ae')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)

    # get rid of unneeded logging info
    _clean_tt_dir(hparams)

    # export training plots
    if hparams['export_train_plots']:
        print('creating training plots...', end='')
        version_dir = os.path.join(hparams['expt_dir'], 'version_%i' % hparams['version'])
        if hparams['model_class'] == 'msps-vae':
            from behavenet.plotting.cond_ae_utils import plot_mspsvae_training_curves
            save_file = os.path.join(version_dir, 'loss_training')
            plot_mspsvae_training_curves(
                hparams, alpha=hparams['ps_vae.alpha'], beta=hparams['ps_vae.beta'],
                delta=hparams['ps_vae.delta'], rng_seed_model=hparams['rng_seed_model'],
                n_latents=hparams['n_ae_latents'] - hparams['n_background'] - hparams['n_labels'],
                n_background=hparams['n_background'], n_labels=hparams['n_labels'], dtype='train',
                save_file=save_file, format='png', version_dir=version_dir)
            save_file = os.path.join(version_dir, 'loss_validation')
            plot_mspsvae_training_curves(
                hparams, alpha=hparams['ps_vae.alpha'], beta=hparams['ps_vae.beta'],
                delta=hparams['ps_vae.delta'], rng_seed_model=hparams['rng_seed_model'],
                n_latents=hparams['n_ae_latents'] - hparams['n_background'] - hparams['n_labels'],
                n_background=hparams['n_background'], n_labels=hparams['n_labels'], dtype='val',
                save_file=save_file, format='png', version_dir=version_dir)
        else:
            save_file = os.path.join(version_dir, 'loss_training')
            export_train_plots(hparams, 'train', save_file=save_file)
            save_file = os.path.join(version_dir, 'loss_validation')
            export_train_plots(hparams, 'val', save_file=save_file)
        print('done')


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
            # Set up gpu ids for parallel gpus
            parallel_gpu_ids = []
            for instance in range(math.ceil(len(gpu_ids) / hyperparams.n_parallel_gpus)):
                idx_beg = instance * hyperparams.n_parallel_gpus
                idx_end = (instance + 1) * hyperparams.n_parallel_gpus
                parallel_gpu_ids.append(','.join(gpu_ids[idx_beg:idx_end]))

            hyperparams.optimize_parallel_gpu(main, gpu_ids=parallel_gpu_ids)

        elif hyperparams.device == 'cpu':
            hyperparams.optimize_parallel_cpu(
                main,
                nb_trials=hyperparams.tt_n_cpu_trials,
                nb_workers=hyperparams.tt_n_cpu_workers)
