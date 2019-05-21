import os
import time
import numpy as np
import random
import torch
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import Decoder
from behavenet.training import fit
from fitting.utils import export_predictions_best
from fitting.utils import experiment_exists
from fitting.utils import export_hparams
from fitting.utils import get_data_generator_inputs
from fitting.utils import get_output_dirs
from fitting.utils import add_lab_defaults_to_parser
from data.data_generator import ConcatSessionsGenerator


def main(hparams):

    # TODO: log files

    # turn matlab-style struct into dict
    hparams = vars(hparams)
    print(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(0, 5))

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

    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

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
    hparams['input_size'] = data_generator.datasets[0].dims[hparams['input_signal']][2]
    print('Data generator loaded')


    hparams['ae_model_path'] = os.path.join(os.path.dirname(data_generator.datasets[0].paths['ae']))
    hparams['ae_predictions_model_path'] = os.path.join(os.path.dirname(data_generator.datasets[0].paths['ae_predictions']))
    hparams['arhmm_predictions_model_path'] = os.path.join(os.path.dirname(data_generator.datasets[0].paths['arhmm_predictions']))
    hparams['arhmm_model_path'] = os.path.join(os.path.dirname(data_generator.datasets[0].paths['arhmm']))

    # Check ARHMM used for predictons is same
    meta_tags = pickle.load(open(os.path.join(hparams['arhmm_predictions_model_path'],'meta_tags.pkl'),'rb'))
    if hparams['arhmm_model_path'] != meta_tags['arhmm_model_path']:
        raise ValueError('ARHMMs do not match')

    hparams['training_completed'] = False
    
    ## Get all latents/predictions in list
    trial_idxs = {}
    latents={}
    latent_predictions={}
    state_log_predictions={}
    states={}
    for data_type in ['train','val','test']:
        if data_type == 'train' and hparams['train_percent']<1:
           n_batches = np.floor(hparams['train_percent']*len(data_generator.batch_indxs[0][data_type]))
           trial_idxs[data_type] = data_generator.batch_indxs[0][data_type][:int(n_batches)]
        else:
           trial_idxs[data_type] = data_generator.batch_indxs[0][data_type] 

        latents[data_type] = [data_generator.datasets[0][i_trial]['ae'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        latent_predictions[data_type] = [data_generator.datasets[0][i_trial]['ae_predictions'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        state_log_predictions[data_type] = [ F.log_softmax(torch.tensor(data_generator.datasets[0][i_trial]['arhmm_predictions'][:]).float(),dim=1).cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        states[data_type] = [data_generator.datasets[0][i_trial]['arhmm'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]

    hparams['total_train_length'] = len(trial_idxs['train'])*data_generator.datasets[0][0]['images'].shape[0]
    export_hparams(hparams, exp)

    print('Model loaded')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    no_nan_predicts = np.concatenate([lat for lat in latent_predictions['train'] ],axis=0)
    no_nan_latents =  np.concatenate([lat for lat in latents['train'] ],axis=0)
    x_covs = np.cov((no_nan_predicts - no_nan_latents).T)
    ver
    #fit(hparams, model, data_generator, exp, method='nll')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--search_type', type=str)  # grid_search, test
    parser.add_argument('--lab_example', type=str)  # musall, steinmetz, markowitz
    parser.add_argument('--lib', default='pt', type=str, choices=['pt', 'tf'])
    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--model_class', default='arhmm-decoding', type=str)

    # arguments for computing resources (nb_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)
    #parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    #parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--signals', default=None, type=str)
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=True)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add fitting arguments
    parser.add_argument('--val_check_interval', default=1)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)

    get_arhmm_decoding_params(namespace, parser)

    return parser.parse_args()


def get_arhmm_decoding_params(namespace, parser):

    # add neural arguments (others are dataset-specific)
    parser.add_argument('--n_arhmm_states', default=32, type=int)
    parser.add_argument('--kappa', default=1e+06, type=float)
    parser.add_argument('--noise_type', default='gaussian', type=str)
    parser.add_argument('--n_max_lags', default=8) 

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
    print('Total fit time: {}'.format(time.time() - t))
    #if hyperparams.export_predictions_best:
    #    export_predictions_best(vars(hyperparams))
