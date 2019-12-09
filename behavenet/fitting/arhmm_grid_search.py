import os
import time
import numpy as np
import random
import ssm
import pickle
from test_tube import HyperOptArgumentParser

from behavenet.fitting.eval import export_states
from behavenet.fitting.eval import export_train_plots
from behavenet.fitting.utils import build_data_generator
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet import get_user_dir
from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.plotting.arhmm_utils import get_latent_arrays_by_dtype


def main(hparams):

    # turn matlab-style struct into dict
    hparams = vars(hparams)
    print('\nexperiment parameters:')
    print(hparams)

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
    latents, trial_idxs = get_latent_arrays_by_dtype(
        data_generator, sess_idxs=list(range(n_datasets)))
    hparams['total_train_length'] = np.sum([l.shape[0] for l in latents['train']])
    # get separated by dataset as well
    latents_sess = {d: None for d in range(n_datasets)}
    trial_idxs_sess = {d: None for d in range(n_datasets)}
    for d in range(n_datasets):
        latents_sess[d], trial_idxs_sess[d] = get_latent_arrays_by_dtype(
            data_generator, sess_idxs=d)
    print('done')

    hparams['ae_model_path'] = os.path.join(
        os.path.dirname(data_generator.datasets[0].paths['ae_latents']))

    # collect model constructor inputs
    if hparams['noise_type'] == 'gaussian':
        if hparams['n_arhmm_lags'] > 0:
            if hparams['model_class'] != 'arhmm':
                raise ValueError('Must specify model_class as arhmm when using AR lags')
            obs_type = 'ar'
        else:
            if hparams['model_class'] != 'hmm':
                raise ValueError('Must specify model_class as hmm when using 0 AR lags')
            obs_type = 'gaussian'
    elif hparams['noise_type'] == 'studentst':
        if hparams['n_arhmm_lags'] > 0:
            if hparams['model_class'] != 'arhmm':
                raise ValueError('Must specify model_class as arhmm when using AR lags')
            obs_type = 'robust_ar'
        else:
            if hparams['model_class'] != 'hmm':
                raise ValueError('Must specify model_class as hmm when using 0 AR lags')
            obs_type = 'studentst'
    else:
        raise ValueError('%s is not a valid noise type' % hparams['noise_type'])

    if hparams['n_arhmm_lags'] > 0:
        obs_kwargs = {'lags': hparams['n_arhmm_lags']}
        obs_init_kwargs = {'localize': True}
    else:
        obs_kwargs = None
        obs_init_kwargs = {}
    if hparams['kappa'] == 0:
        transitions = 'stationary'
        transition_kwargs = None
    else:
        transitions = 'sticky'
        transition_kwargs = {'kappa': hparams['kappa']}

    print('constructing model...', end='')
    np.random.seed(hparams['rng_seed_model'])
    hmm = ssm.HMM(
        hparams['n_arhmm_states'], hparams['n_ae_latents'],
        observations=obs_type, observation_kwargs=obs_kwargs,
        transitions=transitions, transition_kwargs=transition_kwargs)
    hmm.initialize(latents['train'])
    hmm.observations.initialize(latents['train'], **obs_init_kwargs)
    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # TODO: move fitting into own function
    # TODO: adopt early stopping strategy from ssm
    # precompute normalizers
    n_datapoints = {}
    n_datapoints_sess = {}
    for dtype in {'train', 'val', 'test'}:
        n_datapoints[dtype] = np.vstack(latents[dtype]).size
        n_datapoints_sess[dtype] = {}
        for d in range(n_datasets):
            n_datapoints_sess[dtype][d] = np.vstack(latents_sess[d][dtype]).size

    for epoch in range(hparams['n_iters'] + 1):
        # Note: the 0th epoch has no training (randomly initialized model is evaluated) so we cycle
        # through `n_iters` training epochs

        print('epoch %03i/%03i' % (epoch, hparams['n_iters']))
        if epoch > 0:
            hmm.fit(latents['train'], method='em', num_em_iters=1, initialize=False)

        # export aggregated metrics on train/val data
        tr_ll = hmm.log_likelihood(latents['train']) / n_datapoints['train']
        val_ll = hmm.log_likelihood(latents['val']) / n_datapoints['val']
        exp.log({
            'epoch': epoch, 'dataset': -1, 'tr_loss': tr_ll, 'val_loss': val_ll, 'trial': -1})

        # export individual session metrics on train/val data
        for d in range(data_generator.n_datasets):
            tr_ll = hmm.log_likelihood(latents_sess[d]['train']) / n_datapoints_sess['train'][d]
            val_ll = hmm.log_likelihood(latents_sess[d]['val']) / n_datapoints_sess['val'][d]
            exp.log({
                'epoch': epoch, 'dataset': d, 'tr_loss': tr_ll, 'val_loss': val_ll, 'trial': -1})

    # export individual session metrics on test data
    for d in range(n_datasets):
        for i, b in enumerate(trial_idxs_sess[d]['test']):
            n = latents_sess[d]['test'][i].size
            test_ll = hmm.log_likelihood(latents_sess[d]['test'][i]) / n
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


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--search_type', type=str)  # grid_search
    parser.add_argument('--lab_example', type=str)
    parser.add_argument('--save_dir', default=get_user_dir('save'), type=str)
    parser.add_argument('--data_dir', default=get_user_dir('data'), type=str)
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument('--model_class', default='arhmm', choices=['arhmm', 'hmm'], type=str)
    parser.add_argument('--sessions_csv', default='', type=str, help='specify multiple sessions')

    # arguments for computing resources (infer n_gpu_workers from visible gpus)
    parser.add_argument('--tt_n_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_workers', default=5, type=int)
    parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str)
    parser.add_argument('--as_numpy', action='store_true', default=True)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed_data', default=0, type=int, help='control data splits')

    parser.add_argument('--export_train_plots', action='store_true', default=False)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)

    # get model-type specific arguments
    get_arhmm_params(namespace, parser)

    return parser.parse_args()


def get_arhmm_params(namespace, parser):

    parser.add_argument('--ae_experiment_name', default='test_pt', type=str)
    parser.add_argument('--ae_version', default='best')
    parser.add_argument('--ae_model_type', default='conv', type=str)

    parser.add_argument('--n_arhmm_lags', default=1, type=int)
    parser.add_argument('--n_iters', default=150, type=int)

    # add experiment=specific arguments
    if namespace.search_type == 'test':

        parser.add_argument('--experiment_name', default='test', type=str)

        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--n_arhmm_states', default=2, type=int)
        parser.add_argument('--train_frac', default=1.0, type=float)
        parser.add_argument('--kappa', default=0, type=int)
        parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'studentst'], type=str)
        parser.add_argument('--rng_seed_model', default=0, type=int, help='control model initialization')
        parser.add_argument('--export_states', action='store_true', default=False)

    elif namespace.search_type == 'grid_search':

        parser.add_argument('--experiment_name', default='diff_init_grid_search', type=str)

        parser.add_argument('--train_frac', default=1.0, type=float)
        parser.add_argument('--n_ae_latents', default=8, type=int)
        # parser.opt_list('--n_ae_latents', default=12, options=[4, 8, 16], type=int, tunable=True)
        parser.opt_list('--n_arhmm_states', default=16, options=[2, 4, 8, 16, 32], type=int, tunable=True)
        parser.opt_list('--kappa', default=0, options=[1e2, 1e4, 1e6, 1e8, 1e10], type=int, tunable=False)
        parser.opt_list('--noise_type', default='gaussian', options=['gaussian', 'studentst'], type=str, tunable=False)
        parser.add_argument('--rng_seed_model', default=0, type=int, help='control model initialization')
        parser.add_argument('--export_states', action='store_true', default=False)

    elif namespace.search_type == 'data_amounts':

        parser.add_argument('--experiment_name', default='data_amount', type=str)

        parser.add_argument('--kappa', default=0, type=int)
        parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'studentst'], type=str)
        # parser.opt_list('--train_frac', default=1.0, options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], type=float, tunable=True)
        parser.opt_list('--train_frac', default=1.0, options=[0.1, 0.14, 0.19, 0.27, 0.37, 0.52, 0.72, 1.0], type=float, tunable=True)
        # parser.opt_list('--train_frac', default=1.0, options=[1.0], type=float, tunable=True)
        parser.opt_list('--n_ae_latents', default=12, options=[9, 12], type=int, tunable=True)
        parser.opt_list('--n_arhmm_states', default=14, options=[4, 8, 16, 32], type=int, tunable=True)
        parser.opt_list('--rng_seed_model', default=0, options=[0, 1, 2, 3, 4], type=int, tunable=True)
        parser.add_argument('--export_states', action='store_true', default=False)


if __name__ == '__main__':

    hyperparams = get_params('grid_search')

    t = time.time()
    if hyperparams.device == 'cuda' or hyperparams.device == 'gpu':
        if hyperparams.device == 'gpu':
            hyperparams.device = 'cuda'
        gpu_ids = hyperparams.gpus_viz.split(';')
        hyperparams.optimize_parallel_gpu(
            main,
            gpu_ids=gpu_ids,
            nb_trials=hyperparams.tt_n_gpu_trials,
            nb_workers=len(gpu_ids))
    elif hyperparams.device == 'cpu':
        hyperparams.optimize_parallel_cpu(
            main,
            nb_trials=hyperparams.tt_n_cpu_trials,
            nb_workers=hyperparams.tt_n_cpu_workers)

    print('Total fit time: {} sec'.format(time.time() - t))
