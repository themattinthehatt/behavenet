import os
import time
import numpy as np
import random
import ssm
import pickle
import matplotlib
from test_tube import HyperOptArgumentParser

from behavenet.fitting.eval import export_states
from behavenet.fitting.utils import build_data_generator
from behavenet.fitting.utils import create_tt_experiment
from behavenet.fitting.utils import export_hparams
from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.analyses.arhmm_utils import make_ind_arhmm_figures
from behavenet.analyses.arhmm_utils import make_overview_arhmm_figures
matplotlib.use('agg')


def main(hparams):

    # turn matlab-style struct into dict
    hparams = vars(hparams)
    print(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(0, 10))

    # create test-tube experiment
    hparams, sess_ids, exp = create_tt_experiment(hparams)

    # build data generator
    data_generator = build_data_generator(hparams, sess_ids)

    # ####################
    # ### CREATE MODEL ###
    # ####################

    hparams['ae_model_path'] = os.path.join(
        os.path.dirname(data_generator.datasets[0].paths['ae']))

    # Get all latents in list
    # TODO: currently only works for a single session
    trial_idxs = {}
    latents = {}
    for data_type in ['train', 'val', 'test']:
        if data_type == 'train' and hparams['train_percent'] < 1:
           n_batches = np.floor(
               hparams['train_percent']*len(data_generator.datasets[0].batch_indxs[data_type]))
           trial_idxs[data_type] = data_generator.datasets[0].batch_indxs[data_type][:int(n_batches)]
        else:
           trial_idxs[data_type] = data_generator.datasets[0].batch_indxs[data_type]
        latents[data_type] = [data_generator.datasets[0][i_trial]['ae'][:].cpu().detach().numpy() for i_trial in trial_idxs[data_type]]

    hparams['total_train_length'] = len(trial_idxs['train'])*data_generator.datasets[0][0]['images'].shape[0]

    if hparams['noise_type'] == 'gaussian':
        obv_type = 'ar'
    elif hparams['noise_type'] == 'studentst':
        obv_type = 'robust_ar'
    else:
        raise ValueError(hparams['noise_type']+' not a valid noise type')

    print('constructing model...', end='')
    if hparams['kappa'] == 0:
        print('no stickiness')
        hmm = ssm.HMM(
            hparams['n_arhmm_states'], hparams['n_ae_latents'],
            observations=obv_type,
            observation_kwargs=dict(lags=hparams['n_lags']))
    else:
        hmm = ssm.HMM(
            hparams['n_arhmm_states'], hparams['n_ae_latents'],
            observations=obv_type,
            observation_kwargs=dict(lags=hparams['n_lags']),
            transitions="sticky", transition_kwargs=dict(kappa=hparams['kappa']))
    hmm.initialize(latents['train'])
    hmm.observations.initialize(latents['train'], localize=True)

    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    train_ll = hmm.fit(
        latents['train'], method='em', num_em_iters=hparams['n_iters'],
        initialize=False)

    # Reconfigure model/states by usage
    zs = [hmm.most_likely_states(x) for x in latents['train']]
    usage = np.bincount(np.concatenate(zs), minlength=hmm.K)
    perm = np.argsort(usage)[::-1]
    hmm.permute(perm)

    # Save model
    filepath = os.path.join(
        hparams['expt_dir'], 'version_%i' % exp.version, 'best_val_model.pt')

    with open(filepath, "wb") as f:
        pickle.dump(hmm, f)   

    # ######################
    # ### EVALUATE ARHMM ###
    # ######################

    # Evaluate log likelihood of validation data
    validation_ll = hmm.log_likelihood(latents['val'])

    exp.log({'train_ll': np.mean(train_ll), 'val_ll': np.mean(validation_ll)})
    exp.save()

    # Export states
    if hparams['export_states']:
        export_states(hparams, data_generator, hmm)

    # ARHMM figures/videos
    if hparams['make_ind_plots']:
        print('creating individual arhmm figures...', end='')
        make_ind_arhmm_figures(hparams, exp, hmm, latents, trial_idxs, data_generator)
        print('done')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--search_type', type=str)  # grid_search
    parser.add_argument('--lab_example', type=str)  # musall, steinmetz, datta
    parser.add_argument('--tt_save_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument('--model_class', default='arhmm', choices=['arhmm'], type=str)
    parser.add_argument('--sessions_csv', default='', type=str)  # specify multiple sessions

    # arguments for computing resources (infer n_gpu_workers from visible gpus)
    parser.add_argument('--tt_n_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_workers', default=5, type=int)
    #parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    #parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str)
    parser.add_argument('--as_numpy', action='store_true', default=True)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)

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

    parser.add_argument('--n_lags', default=1, type=int)
    parser.add_argument('--n_iters', default=150, type=int)

    parser.add_argument('--plot_n_frames', default=400, type=int, help='number of frames in videos')
    parser.add_argument('--plot_frame_rate', default=7, help='frame rate for plotting videos, if "orig": use data frame rates')

    # add experiment=specific arguments
    if namespace.search_type == 'test':

        parser.add_argument('--experiment_name', default='test', type=str)

        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--n_arhmm_states', default=2, type=int)
        parser.add_argument('--train_percent', default=1.0, type=int)
        parser.opt_list('--train_percent', default=1, options=[0.2, 0.4, 0.6, 0.8], type=float, tunable=False)
        parser.opt_list('--kappa', default=0, options=[1e2, 1e4, 1e6, 1e8, 1e10], type=int, tunable=False)
        parser.opt_list('--noise_type', default='gaussian', options=['gaussian', 'studentst'], type=str, tunable=False)

        # plotting params
        parser.add_argument('--export_states', action='store_true', default=False)
        parser.add_argument('--make_ind_plots', action='store_true', default=False)
        parser.add_argument('--make_overview_plots', action='store_true', default=False)

    elif namespace.search_type == 'grid_search':

        parser.add_argument('--experiment_name', default='diff_init_grid_search', type=str)

        parser.add_argument('--train_percent', default=1.0, type=int)
        # parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.opt_list('--n_ae_latents', default=12, options=[3, 6, 9, 12], type=int, tunable=False)
        parser.opt_list('--n_arhmm_states', default=14, options=[2, 4, 8, 16, 32], type=int, tunable=True)
        parser.opt_list('--kappa', default=0, options=[1e2, 1e4, 1e6, 1e8, 1e10], type=int, tunable=False)
        parser.opt_list('--noise_type', default='gaussian', options=['gaussian', 'studentst'], type=str, tunable=False)

        # plotting params
        parser.add_argument('--export_states', action='store_true', default=True)
        parser.add_argument('--make_ind_plots', action='store_true', default=True)
        parser.add_argument('--make_overview_plots', action='store_true', default=True)

    elif namespace.search_type == 'data_amounts':

        parser.add_argument('--experiment_name', default='data_amount', type=str)

        parser.add_argument('--kappa', default=0, type=int)
        parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'studentst'], type=str)
        parser.opt_list('--train_percent', default=1.0, options=[0.2, 0.4, 0.6, 0.8, 1.0], type=float, tunable=True)
        parser.opt_list('--n_ae_latents', default=12, options=[3, 6, 9, 12], type=int, tunable=True)
        parser.opt_list('--n_arhmm_states', default=14, options=[2, 4, 8, 16, 32], type=int, tunable=True)

        # plotting params
        parser.add_argument('--export_states', action='store_true', default=False)
        parser.add_argument('--make_ind_plots', action='store_true', default=False)
        parser.add_argument('--make_overview_plots', action='store_true', default=False)


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

    if hyperparams.make_overview_plots:
        make_overview_arhmm_figures(hyperparams)
        
    print('Total fit time: {}'.format(time.time() - t))
