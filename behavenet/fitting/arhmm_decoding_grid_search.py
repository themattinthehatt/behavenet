import pickle
import torch.nn.functional as F
import torch
from pylds.lds_messages_interface import info_E_step, info_sample, filter_and_sample, E_step, kalman_filter
from test_tube import HyperOptArgumentParser

import os
import numpy as np
import time

from behavenet.fitting.utils import build_data_generator
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_user_dir

from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.fitting.utils import export_hparams
from ssm.primitives import hmm_expected_states, hmm_sample, viterbi
import random
from behavenet.data.utils import get_region_list
import scipy
from behavenet.models import AE as AE
import pandas as pd

# Let's see how far we can get with diagonal covariances
def _mean_potential(q_mu_x, q_Sigma_x, prior_mus, prior_Sigmas):
    T, D = q_mu_x.shape
    assert q_Sigma_x.shape == (D, D)
    assert prior_mus.shape == (T, D)
    assert prior_Sigmas.shape == (T, D, D)

    # Extract the diagonalss
    q_sigmas = np.tile(np.diag(q_Sigma_x)[None, :], (T, 1))
    prior_sigmas = prior_Sigmas[:, np.arange(D), np.arange(D)]

    # Make sure q_sigma <= prior_sigma
    bad_sigmas = q_sigmas >= prior_sigmas
    q_sigmas[bad_sigmas] = prior_sigmas[bad_sigmas] - 1e-4

    # Compute the effective potential q / p
    sigma_obs = 1 / (1 / q_sigmas - 1 / prior_sigmas)
    mu_obs = sigma_obs * (q_mu_x / q_sigmas - prior_mus / prior_sigmas)

    # embed sigma_obs into a big matrix
    Sigma_obs = np.zeros((T, D, D))
    Sigma_obs[:, np.arange(D), np.arange(D)] = sigma_obs

    assert mu_obs.shape == (T, D)
    assert Sigma_obs.shape == (T, D, D)
    return mu_obs, Sigma_obs


# Compute the info potentials for the initial condition
def _mean_params(mu0, Sigma0, As, bs, Qs, q_mu_x, q_Sigma_x, prior_mus, prior_Sigmas, z_sample):
    # parameter checking
    T = len(z_sample)
    K, D, _ = As.shape
    assert z_sample.dtype == int and np.all(z_sample >= 0) and np.all(z_sample < K)
    assert mu0.shape == (D,)
    assert Sigma0.shape == (D, D)
    assert bs.shape == (K, D)
    assert Qs.shape == (K, D, D)
    assert q_mu_x.shape == (T, D)
    assert q_Sigma_x.shape == (D, D)

    # Make pseudo-inputs (all ones) for bias terms
    inputs = np.ones((T, 1))

    # Compute the effective Kalman smoother parameters
    mu_obs, Sigma_obs = _mean_potential(q_mu_x, q_Sigma_x, prior_mus, prior_Sigmas)

    # Return Kalman smoother args
    return mu0, Sigma0, \
           As[z_sample], bs[:, :, None][z_sample], Qs[z_sample], \
           np.eye(D), np.zeros((D, 1)), Sigma_obs, \
           inputs, mu_obs

# Compute the info potentials for the initial condition
def _prior_mean_params(mu0, Sigma0, As, bs, Qs, z_sample):
    # parameter checking
    T = len(z_sample)
    K, D, _ = As.shape
    assert z_sample.dtype == int and np.all(z_sample >= 0) and np.all(z_sample < K)
    assert mu0.shape == (D,)
    assert Sigma0.shape == (D, D)
    assert bs.shape == (K, D)
    assert Qs.shape == (K, D, D)
    # assert q_mu_x.shape == (T, D)
    # assert q_Sigma_x.shape == (D, D)

    # Make pseudo-inputs (all ones) for bias terms
    inputs = np.ones((T, 1))

    # Compute the effective Kalman smoother parameters
    mu_obs = np.zeros((T, D))
    Sigma_obs = np.tile(1e8 * np.eye(D)[None, :, :], (T, 1, 1))

    # Return Kalman smoother args
    return mu0, Sigma0, \
           As[z_sample], bs[:, :, None][z_sample], Qs[z_sample], \
           np.eye(D), np.zeros((D, 1)), Sigma_obs, \
           inputs, mu_obs

def main(hparams):

    hparams = vars(hparams)
    hparams.pop('trials', False)
    hparams.pop('generate_trials', False)
    hparams.pop('optimize_parallel', False)
    hparams.pop('optimize_parallel_cpu', False)
    hparams.pop('optimize_parallel_gpu', False)
    hparams.pop('optimize_trials_parallel_gpu', False)
    print('\nexperiment parameters:')
    print(hparams)

    # Start at random times (so test tube creates separate folders)

    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(1))

    # create test-tube experiment
    hparams, sess_ids, exp = create_tt_experiment(hparams)
    if hparams is None:
        return

    hparams['training_completed'] = False

    data_generator = build_data_generator(hparams, sess_ids)
    np.random.seed(random.randint(0, 1000))
    # np.random.seed(hparams['rng_seed_model'])
    #
    # Get all latents/predictions in list
    trial_idxs = {}
    latents = {}
    latent_predictions = {}
    state_log_predictions = {}
    states = {}
    images={}
    for data_type in ['train', 'val', 'test']:
        trial_idxs[data_type] = np.sort(data_generator.datasets[0].batch_indxs[data_type])
        latents[data_type] = [data_generator.datasets[0][i_trial]['ae_latents'][:].cpu().detach().numpy()[
                              hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        latent_predictions[data_type] = [
            data_generator.datasets[0][i_trial]['ae_predictions'][:].cpu().detach().numpy()[
            hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        state_log_predictions[data_type] = [
            F.log_softmax(data_generator.datasets[0][i_trial]['arhmm_predictions'][:],
                          dim=1).cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in
            trial_idxs[data_type]]
        states[data_type] = [data_generator.datasets[0][i_trial]['arhmm_states'][:].cpu().detach().numpy()[
                             hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        if data_type == 'test':
            images[data_type] = [data_generator.datasets[0][i_trial]['images'][:].cpu().detach().numpy()[
                                 hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]

    # Load in ARHMM model and get parameters
    arhmm_fname = os.path.dirname(data_generator.datasets[0].paths['arhmm_states']) + '/best_val_model.pt'
    arhmm_model = pickle.load(open(arhmm_fname, 'rb'))

    K = arhmm_model.K
    D = latents['train'][0].shape[1]
    assert arhmm_model.D == D
    P = arhmm_model.transitions.transition_matrix
    As = arhmm_model.observations.As
    bs = arhmm_model.observations.bs
    Qs = arhmm_model.observations.Sigmas

    evals, evecs = np.linalg.eig(P.T)
    perm = np.argsort(evals)[::-1]
    evals, evecs = evals[perm], evecs[:, perm]
    assert np.allclose(evals[0], 1.0)
    if np.any(evecs[:, 0] <= 0):
        evecs[:, 0] = -1 * evecs[:, 0]
    assert np.all(evecs[:, 0] >= 0)
    pz_infty = np.real(evecs[:, 0] / evecs[:, 0].sum())

    # Use training data to compute stationary distributions and decoder covariances
    xs_flat = np.vstack(latents['train'])
    zs_flat = np.concatenate(states['train'], axis=0)
    xs_preds_flat = np.vstack(latent_predictions['train'])

    # Compute the variance of the continuous state decoder
    q_Sigma_x = np.cov((xs_flat - xs_preds_flat).T)
    q_std_x = np.sqrt(np.diag(q_Sigma_x))

    # Set the initial continuous state distribution
    mu0 = np.zeros(D)
    Sigma0 = np.eye(D)

    # Get AE decoder
    ae_model_file = os.path.join(os.path.dirname(data_generator.datasets[0].paths['ae_latents']),'best_val_model.pt')
    ae_arch = pickle.load(open(os.path.join(os.path.dirname(data_generator.datasets[0].paths['ae_latents']),'meta_tags.pkl'),'rb'))
    ae_model = AE(ae_arch)
    ae_model.load_state_dict(torch.load(ae_model_file, map_location=lambda storage, loc: storage))
    ae_model.eval();

    export_hparams(hparams, exp)

    print('Model loaded')
    print(trial_idxs['val'])

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    if hparams['search_type'] == 'grid_search':

        # Compute mean square error on the validation data
        val_mse_bayesian_per_batch = np.zeros((data_generator.n_tot_batches[data_type],))
        val_mse_ff_per_batch = np.zeros((data_generator.n_tot_batches[data_type],))
        data_type = 'val'
        possible_bad_trials=[]
        zs_real = [None]* data_generator.n_tot_batches['val']
        zs_ff_decoded = [None]* data_generator.n_tot_batches['val']
        zs_bayesian_decoded = [None]* data_generator.n_tot_batches['val']
        for i_batch in range(data_generator.n_tot_batches[data_type]):
            #print("val batch: ", i_batch)

            log_qz = state_log_predictions[data_type][i_batch]
            z_potential = log_qz - np.log(pz_infty)
            q_mu_x = latent_predictions[data_type][i_batch]

            # Approximate the posterior with many samples of z
            z_samples = [hmm_sample(np.log(pz_infty), np.log(P)[None, :, :], z_potential)]
            x_samples = []
            Ex_samples = []
            for i_smpl in range(hparams['n_samples']):
                # Compute the prior mean p(x | z)
                prior_args = _prior_mean_params(mu0, Sigma0, As, bs, Qs, z_samples[-1])
                _, prior_mus, prior_Sigmas = kalman_filter(*prior_args)

                # Resample x given discrete states and neural data
                args = _mean_params(mu0, Sigma0, As, bs, Qs, q_mu_x, q_Sigma_x, prior_mus, hparams['scale_factor'] * prior_Sigmas, z_samples[-1])
                x_samples.append(filter_and_sample(*args)[1])
                Ex_samples.append(E_step(*args)[1])

                # Resample z given only neural data
                z_samples.append(hmm_sample(np.log(pz_infty), np.log(P)[None, :, :], z_potential))

            Ex = np.mean(Ex_samples, axis=0)

            # Compute the mean squared error
            xs = latents[data_type][i_batch]
            val_mse_bayesian_per_batch[i_batch] = np.mean((Ex - xs)**2)
            val_mse_ff_per_batch[i_batch] = np.mean((q_mu_x - xs) ** 2)

            zs_real[i_batch] = states[data_type][i_batch]
            zs_ff_decoded[i_batch] = np.argmax(np.exp(log_qz), axis=1)
            zs_bayesian_decoded[i_batch] = viterbi(np.log(pz_infty), np.log(P)[None, :, :], z_potential)

            if (np.sum(zs_ff_decoded[i_batch]==zs_real[i_batch])/zs_real[i_batch].shape[0] - np.sum(zs_bayesian_decoded[i_batch]==zs_real[i_batch])/zs_real[i_batch].shape[0])>.05:
                possible_bad_trials.append(i_batch)
        eval_metrics={}
        eval_metrics['zs_real'] = zs_real
        eval_metrics['zs_ff_decoded'] = zs_ff_decoded
        eval_metrics['zs_bayesian_decoded'] = zs_bayesian_decoded
        eval_metrics['val_mse'] = val_mse_bayesian_per_batch
        eval_metrics['val_mse_ff'] = val_mse_ff_per_batch
        eval_metrics['possible_bad_trials'] = possible_bad_trials
        filepath = os.path.join(
            hparams['expt_dir'], 'version_%i' % exp.version, 'eval_metrics.pkl')
        # val_mse = np.mean(val_mse_bayesian_per_batch)
        # val_mse_ff = np.mean(val_mse_ff_per_batch)
        # filepath = os.path.join(
        #     hparams['expt_dir'], 'version_%i' % exp.version, 'possible_bad_trials')
        # np.save(filepath,possible_bad_trials)
        exp.log({'val_loss': np.mean(val_mse_bayesian_per_batch),'val_loss_ff':np.mean(val_mse_ff_per_batch)})
        exp.save()
        with open(filepath, 'wb') as handle:
            pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif hparams['search_type'] == 'best':

        eval_metrics={}
        example_traces_and_images={}

        # Choose random trials
        # n_examples = 5
        # which_trials = np.sort(np.random.choice(data_generator.n_tot_batches['test'], size=(n_examples,), replace=False))
        if hparams['lab']=='musall':
            which_trials = np.arange(54) #np.asarray([ 5,  7, 10, 24, 28, 32, 36, 39, 42, 47])
        else:
            which_trials = np.arange(54) #np.asarray([1, 3, 6, 7, 8])

        test_mse_bayesian_per_batch = np.zeros((data_generator.n_tot_batches['test'],))
        test_mse_ff_per_batch = np.zeros((data_generator.n_tot_batches['test'],))
        test_mse_Ex_per_batch = np.zeros((data_generator.n_tot_batches['test'],))
        test_mse_Exz_per_batch = np.zeros((data_generator.n_tot_batches['test'],))

        zs_real = [None]* data_generator.n_tot_batches['test']
        zs_ff_decoded = [None]* data_generator.n_tot_batches['test']
        zs_bayesian_decoded = [None]* data_generator.n_tot_batches['test']
        zs_bayesian_probs = [None]* data_generator.n_tot_batches['test']

        possible_bad_trials = []

        data_type = 'test'
        for i_batch in range(data_generator.n_tot_batches['test']):

            # Run the decoder
            log_qz = state_log_predictions[data_type][i_batch]
            z_potential = log_qz - np.log(pz_infty)
            q_mu_x = latent_predictions[data_type][i_batch]

            # Approximate the posterior with many samples of z
            z_samples = [hmm_sample(np.log(pz_infty), np.log(P)[None, :, :], z_potential)]
            x_samples = []
            Ex_samples = []
            for i_smpl in range(hparams['n_samples']):
                # Compute the prior mean p(x | z)
                prior_args = _prior_mean_params(mu0, Sigma0, As, bs, Qs, z_samples[-1])
                _, prior_mus, prior_Sigmas = kalman_filter(*prior_args)

                # Resample x given discrete states and neural data
                args = _mean_params(mu0, Sigma0, As, bs, Qs, q_mu_x, q_Sigma_x, prior_mus, hparams['scale_factor'] * prior_Sigmas,
                                    z_samples[-1])
                x_samples.append(filter_and_sample(*args)[1])
                Ex_samples.append(E_step(*args)[1])

                # Resample z given only neural data
                z_samples.append(hmm_sample(np.log(pz_infty), np.log(P)[None, :, :], z_potential))

            Ez, _, _ = hmm_expected_states(np.log(pz_infty), np.log(P)[None, :, :], z_potential)
            Ex = np.mean(Ex_samples, axis=0)
            stdx = np.std(x_samples, axis=0)

            if i_batch in which_trials:
                example_traces_and_images[i_batch]={}

                xs = latents[data_type][i_batch]
                Ez_behavior, _, _ = arhmm_model.expected_states(xs)
                example_traces_and_images[i_batch]['Ez_behavior'] = Ez_behavior
                example_traces_and_images[i_batch]['log_qz'] = log_qz
                example_traces_and_images[i_batch]['Ez'] = Ez
                example_traces_and_images[i_batch]['Ex'] = Ex
                example_traces_and_images[i_batch]['stdx'] = stdx
                example_traces_and_images[i_batch]['x_samples'] = x_samples
                example_traces_and_images[i_batch]['q_mu_x'] = q_mu_x
                example_traces_and_images[i_batch]['xs'] = xs
                example_traces_and_images[i_batch]['images'] = images[data_type][i_batch]
                decoded_images = ae_model.decoding(torch.tensor(Ex).float(), None, None).cpu().detach().numpy()
                example_traces_and_images[i_batch]['decoded_images'] = decoded_images

            # Compute the mean squared error
            xs = latents[data_type][i_batch]
            test_mse_bayesian_per_batch[i_batch] = np.mean((Ex - xs) ** 2)
            test_mse_ff_per_batch[i_batch] = np.mean((q_mu_x - xs) ** 2)

            Ez, _, _ = hmm_expected_states(np.log(pz_infty), np.log(P)[None, :, :], z_potential)
            zs_real[i_batch] = states[data_type][i_batch]
            zs_ff_decoded[i_batch] = np.argmax(np.exp(log_qz), axis=1)
            zs_bayesian_decoded[i_batch] = viterbi(np.log(pz_infty), np.log(P)[None, :, :], z_potential)

            if (np.sum(zs_ff_decoded[i_batch]==zs_real[i_batch])/zs_real[i_batch].shape[0] - np.sum(zs_bayesian_decoded[i_batch]==zs_real[i_batch])/zs_real[i_batch].shape[0])>.05:
                possible_bad_trials.append(i_batch)
            # elif 100*(test_mse_bayesian_per_batch[i_batch]-test_mse_ff_per_batch[i_batch])/test_mse_ff_per_batch[i_batch]>5:
            #     possible_bad_trials.append(i_batch)

            zs_bayesian_probs[i_batch] = Ez

            ## Get E[x]
            Ex_samples = []
            T = z_samples[0].shape[0]
            for i_smpl in range(hparams['n_samples']):
                z_sample = arhmm_model.sample(T)[0]
                # Compute the prior mean p(x | z)
                prior_args = _prior_mean_params(mu0, Sigma0, As, bs, Qs, z_sample)
                # Resample x given discrete states and neural data
                Ex_samples.append(E_step(*prior_args)[1])

            Ex_prior = np.mean(Ex_samples, axis=0)
            test_mse_Ex_per_batch[i_batch] = np.mean((Ex_prior - xs) ** 2)

            ## Get E[x|z]
            Ex_samples = []
            for i_smpl in range(hparams['n_samples']):
                z_sample = hmm_sample(np.log(pz_infty), np.log(P)[None, :, :], z_potential)
                # Compute the prior mean p(x | z)
                prior_args = _prior_mean_params(mu0, Sigma0, As, bs, Qs, z_sample)
                # Resample x given discrete states and neural data
                Ex_samples.append(E_step(*prior_args)[1])

            Exz_prior = np.mean(Ex_samples, axis=0)
            test_mse_Exz_per_batch[i_batch] = np.mean((Exz_prior - xs) ** 2)

        eval_metrics['test_mse_bayesian'] = test_mse_bayesian_per_batch
        eval_metrics['test_mse_ff'] = test_mse_ff_per_batch
        eval_metrics['test_mse_Ex'] = test_mse_Ex_per_batch
        eval_metrics['test_mse_Exz'] = test_mse_Exz_per_batch
        eval_metrics['possible_bad_trials'] = possible_bad_trials
        eval_metrics['zs_real'] = zs_real
        eval_metrics['zs_ff_decoded'] = zs_ff_decoded
        eval_metrics['zs_bayesian_decoded'] = zs_bayesian_decoded
        eval_metrics['test_mse_Ex'] = test_mse_Ex_per_batch
        eval_metrics['test_mse_Exz'] = test_mse_Exz_per_batch

        zs_real_flat = np.concatenate(zs_real, axis=0)
        zs_real_flat_train = np.concatenate(states['train'], axis=0)
        zs_ff_decoded_flat = np.concatenate(zs_ff_decoded, axis=0)
        zs_bayesian_decoded_flat = np.concatenate(zs_bayesian_decoded, axis=0)
        zs_bayesian_probs_flat = np.concatenate(zs_bayesian_probs, axis=0)

        training_means = np.mean(xs_flat, axis=0)
       #xs_flat_test = np.vstack(latents['test'])

        eval_metrics['test_mse_baseline'] = [np.mean((latents['test'][i]- training_means) ** 2) for i in range(data_generator.n_tot_batches['test'])] #np.mean((xs_flat_test - training_means) ** 2)

        eval_metrics['test_acc_baseline'] = np.sum((zs_real_flat == scipy.stats.mode(zs_real_flat_train).mode[0])) / zs_real_flat.shape[0]
        eval_metrics['test_acc_ff'] = np.sum((zs_real_flat == zs_ff_decoded_flat)) / zs_real_flat.shape[0]
        eval_metrics['test_acc_bayesian'] = np.sum((zs_real_flat == zs_bayesian_decoded_flat)) / zs_real_flat.shape[0]

        ## Make confusion matrix
        test_sts = np.unique(zs_real_flat)
        eval_metrics['confusion_matrix'] = np.zeros((test_sts.shape[0], test_sts.shape[0]))
        zs_bayesian_probs_flat_lim = zs_bayesian_probs_flat[:, test_sts]

        for i_state, state in enumerate(test_sts):
            eval_metrics['confusion_matrix'][i_state] = np.mean(zs_bayesian_probs_flat_lim[zs_real_flat == state], axis=0)


        eval_metrics['scale_factor'] = hparams['scale_factor']

        metric = pd.read_csv(os.path.join(os.path.dirname(hparams['best_val_path']), 'metrics.csv'))
        eval_metrics['val_mse'] = metric['val_loss'].min()
        eval_metrics['val_mse_ff'] = metric['val_loss_ff'].min()

        filepath = os.path.join(
            hparams['expt_dir'], 'version_%i' % exp.version, 'eval_metrics.pkl')

        with open(filepath, 'wb') as handle:
            pickle.dump(eval_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        filepath = os.path.join(
            hparams['expt_dir'], 'version_%i' % exp.version, 'example_traces_and_images.pkl')

        with open(filepath, 'wb') as handle:
            pickle.dump(example_traces_and_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    exp.save()
    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)


def get_params(strategy):
    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--search_type', type=str)  # grid_search, test
    parser.add_argument('--lab_example', type=str)  # musall, steinmetz, markowitz

    parser.add_argument('--tt_save_path', default=get_user_dir('save'), type=str)
    parser.add_argument('--data_dir', default=get_user_dir('data'), type=str)
    parser.add_argument('--model_class', default='bayesian-decoding', type=str)

    parser.add_argument('--model_type', default=None, type=str)

    # arguments for computing resources (n_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_n_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_workers', default=5, type=int)
    # parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--reg_list', default='none', type=str, choices=['none', 'arg', 'all'])
    parser.add_argument('--subsample_regions', default='none', choices=['none', 'single', 'loo'])
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=True)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)
    parser.add_argument('--train_frac', default=1.0, type=float)

    # add fitting arguments
    parser.add_argument('--val_check_interval', default=1)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)
    namespace, extra = parser.parse_known_args()

    # add regions to opt_list if desired
    if namespace.reg_list == 'all':
        parser.opt_list('--region', options=get_region_list(namespace), type=str, tunable=True)
    elif namespace.reg_list == 'arg':
        parser.add_argument('--region', default='all', type=str)
    elif namespace.reg_list == 'none':  # TODO: fix this ambiguity
        parser.add_argument('--region', default='all', type=str)
    else:
        raise ValueError(
            '"%s" is not a valid region_list' % namespace.region_list)

    get_bayesian_decoding_params(namespace, parser)

    return parser.parse_args()


def get_bayesian_decoding_params(namespace, parser):
    parser.add_argument('--ae_experiment_name', default='test_pt',type=str)
    parser.add_argument('--ae_version', default='best')

    parser.add_argument('--neural_ae_experiment_name', default='grid_search')
    parser.add_argument('--neural_ae_model_type', default='ff')
    parser.add_argument('--neural_ae_version', default='best')
    parser.add_argument('--ae_multisession', default=None, type=int)

    parser.add_argument('--neural_arhmm_experiment_name', default='grid_search')
    parser.add_argument('--neural_arhmm_model_type', default='ff')
    parser.add_argument('--neural_arhmm_version', default='best')

    parser.add_argument('--arhmm_experiment_name', default='diff_init_grid_search')
    parser.add_argument('--arhmm_version', default='best')

    parser.add_argument('--ae_model_type', default='conv')
    parser.add_argument('--n_ae_latents', default=12, type=int)
    parser.add_argument('--n_arhmm_states', default=32, type=int)
    parser.add_argument('--kappa', default=0, type=float)
    parser.add_argument('--noise_type', default='gaussian', type=str)
    parser.add_argument('--arhmm_multisession', default=None, type=int)
    parser.add_argument('--n_max_lags', default=8, type=int)
    #parser.add_argument('--rng_seed_model', default=0, type=int, help='control model initialization')  # TODO: add this to torch models

    parser.add_argument('--n_samples', default=100, type=int)

    # add neural arguments (others are dataset-specific)
    if namespace.search_type == 'grid_search':
        parser.add_argument('--experiment_name', '-en', default='grid_search_update', type=str)
        if namespace.lab == 'musall':
            parser.opt_list('--scale_factor', default=6, options=[1, 2, 4, 6, 8, 10], type=float, tunable=True)
        elif namespace.lab == 'steinmetz':
            parser.opt_list('--scale_factor', default=4000, options=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7], type=float, tunable=True) #[1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    elif namespace.search_type == 'best':

        from behavenet.fitting.utils import get_best_model_version
        namespace, extra = parser.parse_known_args()
        hparams_tmp = vars(namespace)
        hparams_tmp['experiment_name'] = 'grid_search'
        expt_dir = get_expt_dir(hparams_tmp)
        print(expt_dir)
        best_version = get_best_model_version(expt_dir)[0]
        print(best_version)
        best_file = os.path.join(expt_dir, best_version, 'meta_tags.pkl')
        print('Loading best bayesian decoder from %s' % best_file)
        with open(best_file, 'rb') as f:
            hparams_best = pickle.load(f)
        parser.add_argument('--scale_factor', default=hparams_best['scale_factor'], type=float)
        parser.add_argument('--best_version',  default=best_version, type=str)
        parser.add_argument('--best_val_path', default=best_file, type=str)
        parser.add_argument('--experiment_name', '-en', default='best_update', type=str)


if __name__ == '__main__':

    hyperparams = get_params('grid_search')

    for hyperparam_trial in hyperparams.trials(300):
        main(hyperparam_trial)
