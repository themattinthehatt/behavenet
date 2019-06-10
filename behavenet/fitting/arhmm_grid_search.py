import os
import time
import numpy as np
import random
import torch
import sys
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import Decoder
from behavenet.training import fit
from behavenet.utils import export_states
from behavenet.fitting.utils import export_predictions_best
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import export_hparams
from behavenet.fitting.utils import get_data_generator_inputs
from behavenet.fitting.utils import get_output_dirs
from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.data.data_generator import ConcatSessionsGenerator
from behavenet.analyses.arhmm_utils import get_discrete_chunks
from behavenet.analyses.arhmm_utils import get_state_durations
from behavenet.analyses.arhmm_utils import relabel_states_by_use
from behavenet.analyses.arhmm_utils import make_syllable_movies
from behavenet.analyses.arhmm_utils import make_real_vs_generated_movies
from behavenet.analyses.arhmm_utils import make_ind_arhmm_figures
from behavenet.analyses.arhmm_utils import make_overview_arhmm_figures
import ssm
import pickle
import matplotlib
matplotlib.use('agg')


def main(hparams):

    # TODO: log files

    # turn matlab-style struct into dict
    hparams = vars(hparams)
    print(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(0, 10))

    # #########################
    # ### Create Experiment ###
    # #########################

    # get session_dir, results_dir (session_dir + decoding details),
    # expt_dir (results_dir + experiment details)
    hparams['session_dir'], hparams['results_dir'], hparams['expt_dir'] = \
        get_output_dirs(hparams)
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])

    # check to see if experiment already exists
    if experiment_exists(hparams):
        print('Experiment exists! Aborting fit')
        return

    exp = Experiment(
        name=hparams['experiment_name'],
        debug=False,
        save_dir=hparams['results_dir'])
    exp.save()

    ###########################
    ### LOAD DATA GENERATOR ###
    ###########################

    print('building data generator')

    hparams, signals, transforms, load_kwargs = get_data_generator_inputs(hparams)
    ids = {
        'lab': hparams['lab'],
        'expt': hparams['expt'],
        'animal': hparams['animal'],
        'session': hparams['session']}
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], ids,
        signals=signals, transforms=transforms, load_kwargs=load_kwargs,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])

    hparams['ae_model_path'] = os.path.join(os.path.dirname(data_generator.datasets[0].paths['ae']))
    hparams['training_completed'] = False
    
    ## Get all latents in list
    trial_idxs = {}
    latents={}
    for data_type in ['train','val','test']:
        if data_type == 'train' and hparams['train_percent']<1:
           n_batches = np.floor(hparams['train_percent']*len(data_generator.batch_indxs[0][data_type]))
           trial_idxs[data_type] = data_generator.batch_indxs[0][data_type][:int(n_batches)]
        else:
           trial_idxs[data_type] = data_generator.batch_indxs[0][data_type] 
        latents[data_type] = [data_generator.datasets[0][i_trial]['ae'][:].cpu().detach().numpy() for i_trial in trial_idxs[data_type]]

    hparams['total_train_length'] = len(trial_idxs['train'])*data_generator.datasets[0][0]['images'].shape[0]
    export_hparams(hparams, exp)

    #################
    ### FIT ARHMM ###
    #################

    if hparams['noise_type'] =='gaussian':
        obv_type = 'ar'
    elif hparams['noise_type'] == 'studentst':
        obv_type = 'robust_ar'
    else:
        raise ValueError(hparams['noise_type']+' not a valid noise type')

    if hparams['kappa'] == 0:
        print('No stickiness')
        hmm = ssm.HMM(hparams['n_arhmm_states'], hparams['n_ae_latents'], 
                      observations=obv_type, observation_kwargs=dict(lags=hparams['n_lags']))
    else:
        hmm = ssm.HMM(hparams['n_arhmm_states'], hparams['n_ae_latents'], 
                      observations=obv_type, observation_kwargs=dict(lags=hparams['n_lags']),
                      transitions="sticky", transition_kwargs=dict(kappa=hparams['kappa']))

    hmm.initialize(latents['train'])
    hmm.observations.initialize(latents['train'], localize=False)
    train_ll = hmm.fit(latents['train'], method="em", num_em_iters=hparams['n_iters'],initialize=False)

    # Reconfigure model/states by usage
    zs = [hmm.most_likely_states(x) for x in latents['train']]
    usage = np.bincount(np.concatenate(zs), minlength=hmm.K)
    perm = np.argsort(usage)[::-1]
    hmm.permute(perm)

    # Save model
    filepath = os.path.join(
        hparams['results_dir'], 'test_tube_data',
        hparams['experiment_name'],
        'version_%i' % exp.version,
        'best_val_model.pt')

    with open(filepath, "wb") as f:
        pickle.dump(hmm, f)   

    ######################
    ### EVALUATE ARHMM ###
    ######################

    # Evaluate log likelihood of validation data
    validation_ll = hmm.log_likelihood(latents['val'])

    exp.log({'train_ll': train_ll, 'val_ll': validation_ll})
    exp.save()

    ## Export states
    if hparams['export_states']:
        export_states(hparams, exp, data_generator, hmm)

    ## ARHMM figures/videos
    if hparams['make_plots']:
        make_ind_arhmm_figures(hparams, exp, hmm, latents, trial_idxs, data_generator)

    hparams['training_completed'] = True
    export_hparams(hparams, exp)

def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--lab_example', type=str)  # musall, steinmetz, markowitz
    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument('--model_class', default='arhmm', choices=['arhmm'], type=str)

    # arguments for computing resources (nb_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)
    #parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    #parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    #parser.add_argument('--signals', default=None, type=str)
    #parser.add_argument('--transforms', default=None)
    #parser.add_argument('--load_kwargs', default=None)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=True)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)

    get_arhmm_params(namespace, parser)

    return parser.parse_args()


def get_arhmm_params(namespace, parser):

    # add data arguments
    if namespace.search_type == 'grid_search':
        parser.add_argument('--experiment_name', '-en', default='diff_init_grid_search', type=str) #'grid_search'
    



    parser.add_argument('--ae_experiment_name', default='test_pt',type=str)
    parser.add_argument('--ae_version', default='best')
    parser.add_argument('--ae_model_type', default='conv')
    parser.add_argument('--n_ae_latents', default=12, type=int)
    parser.opt_list('--n_arhmm_states', default=14, options=[4,8,16,32], type=int, tunable=True) 
    parser.opt_list('--train_percent', default=1, options=[.2, .4, .6, .8, 1], tunable=False) 
    parser.opt_list('--kappa', default=0, options=[1e2, 1e4, 1e6, 1e8, 1e10],type=int, tunable=False) 
    parser.opt_list('--noise_type', default='gaussian', options = ['gaussian','studentst'], type=str, tunable=False)
    parser.add_argument('--n_lags', default=1, type=int)
    parser.add_argument('--n_iters', default=150, type=int)

    # Plotting params
    parser.add_argument('--export_states', action='store_true', default=True)
    parser.add_argument('--make_plots', action='store_true', default=True)
    parser.add_argument('--plot_n_frames', default=400, type=int) # Number of frames in videos
    parser.add_argument('--plot_frame_rate', default=7) # Frame rate for plotting videos, if 'orig': use data frame rates


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
            nb_trials=hyperparams.tt_nb_gpu_trials,
            nb_workers=len(gpu_ids))
    elif hyperparams.device == 'cpu':
        hyperparams.optimize_parallel_cpu(
            main,
            nb_trials=hyperparams.tt_nb_cpu_trials,
            nb_workers=hyperparams.tt_nb_cpu_workers)

    if hyperparams.make_plots:
        make_overview_arhmm_figures(hyperparams)
        
    print('Total fit time: {}'.format(time.time() - t))

