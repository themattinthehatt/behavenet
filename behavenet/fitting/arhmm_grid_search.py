import os
import time
import numpy as np
import random
import ssm
import pickle

from behavenet.data.utils import build_data_generator
from behavenet.fitting.eval import export_states
from behavenet.fitting.eval import export_train_plots
from behavenet.fitting.hyperparam_utils import get_all_params
from behavenet.fitting.hyperparam_utils import get_slurm_params
from behavenet.fitting.utils import _clean_tt_dir
from behavenet.fitting.utils import _print_hparams
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet.plotting.arhmm_utils import get_latent_arrays_by_dtype


def main(hparams):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    if hparams['transitions'] == 'sticky' and hparams['kappa'] == 0:
        print('Cannot fit sticky transitions with kappa=0! Aborting fit')
        return
    if hparams['transitions'] != 'sticky' and hparams['kappa'] > 0:
        print('Cannot fit %s transitions with kappa>0! Aborting fit' % hparams['transitions'])
        return

    # print hparams to console
    _print_hparams(hparams)

    # start at random times (so test tube creates separate folders)
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

    # get all latents in list
    n_datasets = len(data_generator)
    print('collecting observations from data generator...', end='')
    data_key = 'ae_latents'
    if hparams['model_class'].find('labels') > -1:
        data_key = 'labels'
    latents, trial_idxs = get_latent_arrays_by_dtype(
        data_generator, sess_idxs=list(range(n_datasets)), data_key=data_key)
    obs_dim = latents['train'][0].shape[1]

    hparams['total_train_length'] = np.sum([z.shape[0] for z in latents['train']])
    # get separated by dataset as well
    latents_sess = {d: None for d in range(n_datasets)}
    trial_idxs_sess = {d: None for d in range(n_datasets)}
    for d in range(n_datasets):
        latents_sess[d], trial_idxs_sess[d] = get_latent_arrays_by_dtype(
            data_generator, sess_idxs=d, data_key=data_key)
    print('done')

    if hparams['model_class'] == 'arhmm' or hparams['model_class'] == 'hmm':
        hparams['ae_model_path'] = os.path.join(
            os.path.dirname(data_generator.datasets[0].paths['ae_latents']))
        hparams['ae_model_latents_file'] = data_generator.datasets[0].paths['ae_latents']

    if hparams['n_arhmm_lags'] > 0:
        if hparams['model_class'][:5] != 'arhmm':  # 'arhmm' or 'arhmm-labels'
            raise ValueError('Must specify model_class as arhmm when using AR lags')
    else:
        if hparams['model_class'][:3] != 'hmm':  # 'hmm' or 'hmm-labels'
            raise ValueError('Must specify model_class as hmm when using 0 AR lags')

    # determine observation model
    if hparams['noise_type'] == 'gaussian':
        if hparams['n_arhmm_lags'] > 0:
            obs_type = 'ar'
        else:
            obs_type = 'gaussian'
    elif hparams['noise_type'] == 'studentst':
        if hparams['n_arhmm_lags'] > 0:
            obs_type = 'robust_ar'
        else:
            obs_type = 'studentst'
    elif hparams['noise_type'] == 'diagonal_gaussian':
        if hparams['n_arhmm_lags'] > 0:
            obs_type = 'diagonal_ar'
        else:
            obs_type = 'diagonal_gaussian'
    elif hparams['noise_type'] == 'diagonal_studentst':
        if hparams['n_arhmm_lags'] > 0:
            obs_type = 'diagonal_robust_ar'
        else:
            obs_type = 'diagonal_studentst'
    else:
        raise ValueError('%s is not a valid noise type' % hparams['noise_type'])

    if hparams['n_arhmm_lags'] > 0:
        obs_kwargs = {'lags': hparams['n_arhmm_lags']}
        obs_init_kwargs = {'localize': True}
    else:
        obs_kwargs = None
        obs_init_kwargs = {}

    # determine transition model
    if hparams['transitions'] == 'stationary' or hparams['transitions'] == 'standard':
        transitions = 'stationary'
        transition_kwargs = None
    elif hparams['transitions'] == 'sticky':
        transitions = 'sticky'
        transition_kwargs = {'kappa': hparams['kappa']}
    elif hparams['transitions'] == 'recurrent':
        transitions = 'recurrent'
        transition_kwargs = None
    elif hparams['transitions'] == 'recurrent_only':
        transitions = 'recurrent_only'
        transition_kwargs = None
    else:
        raise ValueError('%s is not a valid transition type' % hparams['transitions'])

    print('constructing model...', end='')
    np.random.seed(hparams['rng_seed_model'])
    hmm = ssm.HMM(
        hparams['n_arhmm_states'], obs_dim,
        observations=obs_type, observation_kwargs=obs_kwargs,
        transitions=transitions, transition_kwargs=transition_kwargs)
    hmm.initialize(latents['train'])
    hmm.observations.initialize(latents['train'], **obs_init_kwargs)
    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    hmm.hparams = hparams
    print('done')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # TODO: move fitting into own function
    # precompute normalizers
    n_datapoints = {}
    n_datapoints_sess = {}
    for dtype in {'train', 'val', 'test'}:
        n_datapoints[dtype] = np.vstack(latents[dtype]).size
        n_datapoints_sess[dtype] = {}
        for d in range(n_datasets):
            n_datapoints_sess[dtype][d] = np.vstack(latents_sess[d][dtype]).size

    val_ll_prev = np.inf
    tolerance = hparams.get('arhmm_es_tol', 0)
    # hmm.fit(
    #     latents['train'], method='em', num_iters=hparams['n_iters'], initialize=False,
    #     tolerance=tolerance)
    # epoch = hparams['n_iters']
    for epoch in range(hparams['n_iters'] + 1):
        # Note: the 0th epoch has no training (randomly initialized model is evaluated) so we cycle
        # through `n_iters` training epochs

        print('epoch %03i/%03i' % (epoch, hparams['n_iters']))
        if epoch > 0:
            hmm.fit(latents['train'], method='em', num_iters=1, initialize=False)

        # export aggregated metrics on train/val data
        tr_ll = -hmm.log_likelihood(latents['train']) / n_datapoints['train']
        val_ll = -hmm.log_likelihood(latents['val']) / n_datapoints['val']
        exp.log({
            'epoch': epoch, 'dataset': -1, 'tr_loss': tr_ll, 'val_loss': val_ll, 'trial': -1})

        # export individual session metrics on train/val data
        for d in range(data_generator.n_datasets):
            tr_ll = -hmm.log_likelihood(latents_sess[d]['train']) / n_datapoints_sess['train'][d]
            val_ll = -hmm.log_likelihood(latents_sess[d]['val']) / n_datapoints_sess['val'][d]
            exp.log({
                'epoch': epoch, 'dataset': d, 'tr_loss': tr_ll, 'val_loss': val_ll, 'trial': -1})

        # check for convergence
        if epoch > 10 and np.abs((val_ll - val_ll_prev) / val_ll) < tolerance:
            print('relative change less than tolerance=%1.2f; training terminating!' % tolerance)
            break

        val_ll_prev = val_ll

    # export individual session metrics on test data
    for d in range(n_datasets):
        for i, b in enumerate(trial_idxs_sess[d]['test']):
            n = latents_sess[d]['test'][i].size
            test_ll = -hmm.log_likelihood(latents_sess[d]['test'][i]) / n
            exp.log({'epoch': epoch, 'dataset': d, 'test_loss': test_ll, 'trial': b})
    exp.save()

    # reconfigure model/states by usage
    zs = [hmm.most_likely_states(x) for x in latents['train']]
    usage = np.bincount(np.concatenate(zs), minlength=hmm.K)
    perm = np.argsort(usage)[::-1]
    hmm.permute(perm)

    # save model
    filepath = os.path.join(hparams['expt_dir'], 'version_%i' % exp.version, 'best_val_model.pt')
    with open(filepath, 'wb') as f:
        pickle.dump(hmm, f)

    # ######################
    # ### EVALUATE ARHMM ###
    # ######################

    # export states
    if hparams['export_states']:
        export_states(hparams, data_generator, hmm)

    # export training plots
    if hparams['export_train_plots']:
        print('creating training plots...', end='')
        version_dir = os.path.join(hparams['expt_dir'], 'version_%i' % hparams['version'])
        save_file = os.path.join(version_dir, 'loss_training')
        export_train_plots(hparams, 'train', loss_type='ll', save_file=save_file)
        save_file = os.path.join(version_dir, 'loss_validation')
        export_train_plots(hparams, 'val', loss_type='ll', save_file=save_file)
        print('done')

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
