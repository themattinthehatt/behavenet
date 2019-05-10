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

    # ####################
    # ### CREATE MODEL ###
    # ####################

    if hparams['lib'] == 'tf':
        raise ValueError('TF decoders not currently supported')

    torch_rnd_seed = torch.get_rng_state()
    hparams['model_build_rnd_seed'] = torch_rnd_seed

    model = Decoder(hparams)
    model.to(hparams['device'])

    torch_rnd_seed = torch.get_rng_state()
    hparams['training_rnd_seed'] = torch_rnd_seed

    # save out hparams as csv and dict for easy reloading
    hparams['training_completed'] = False
    export_hparams(hparams, exp)

    print('Model loaded')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # t = time.time()
    # for i in range(20):
    #     batch, dataset = data_generator.next_batch('train')
    #     print('Trial {}'.format(batch['batch_indx']))
    #     print(batch['neural'].shape)
    #     print(batch['ae'].shape)
    # print('Epoch processed!')
    # print('Time elapsed: {}'.format(time.time() - t))

    fit(hparams, model, data_generator, exp, method='nll')

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
    parser.add_argument('--model_type', default='ff', choices=['ff', 'linear', 'lstm'], type=str)
    parser.add_argument('--model_class', default='neural-ae', choices=['neural-ae', 'neural-arhmm'], type=str)

    # arguments for computing resources (nb_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)
    parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--signals', default=None, type=str)
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=False)
    parser.add_argument('--batch_load', action='store_true', default=False)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add fitting arguments
    parser.add_argument('--val_check_interval', default=1)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)

    get_decoding_params(namespace, parser)

    return parser.parse_args()


def get_decoding_params(namespace, parser):

    # add neural arguments (others are dataset-specific)
    parser.add_argument('--neural_thresh', default=1.0, help='minimum firing rate for spikes (Hz)', type=float)
    parser.add_argument('--neural_region', default='all', choices=['all', 'single', 'loo'])

    # add data arguments
    if namespace.model_class == 'neural-ae':
        # ae arguments
        parser.add_argument('--ae_experiment_name', type=str)
        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--ae_version', default='best')
        parser.add_argument('--ae_model_type', default='conv')
    elif namespace.model_class == 'neural-arhmm':
        # ae arguments
        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--ae_model_type', default='conv')
        # arhmm arguments
        parser.add_argument('--arhmm_experiment_name', type=str)
        parser.add_argument('--n_arhmm_states', default=12, type=int)
        parser.add_argument('--arhmm_version', default='best')
    elif namespace.model_class == 'neural-dlc':
        raise NotImplementedError

    parser.add_argument('--enable_early_stop', action='store_true', default=True)
    parser.add_argument('--early_stop_history', default=10, type=float)
    parser.add_argument('--min_nb_epochs', default=1, type=int)
    parser.add_argument('--max_nb_epochs', default=500, type=int)
    parser.add_argument('--activation', default='relu', choices=['linear', 'relu', 'lrelu', 'sigmoid', 'tanh'])

    if namespace.search_type == 'test':

        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--n_lags', default=4, type=int)
        parser.add_argument('--l2_reg', default=1e-3, type=float)
        parser.add_argument('--n_max_lags', default=8)  # should match largest value in --n_lags options
        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=False, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', '-en', default='test', type=str)

        if namespace.model_type == 'linear':
            parser.add_argument('--n_hid_layers', default=0, type=int)
        elif namespace.model_type == 'ff':
            parser.add_argument('--n_hid_layers', default=1, type=int)
            parser.add_argument('--n_final_units', default=16, type=int)
            parser.add_argument('--n_int_units', default=64, type=int)
        elif namespace.model_type == 'lstm':
            raise NotImplementedError

    elif namespace.search_type == 'grid_search':

        parser.opt_list('--learning_rate', default=1e-3, options=[1e-2, 1e-3, 1e-4], type=float, tunable=True)
        parser.opt_list('--n_lags', default=0, options=[0, 1, 2, 4, 8], type=int, tunable=True)
        parser.opt_list('--l2_reg', default=0, options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], type=float, tunable=True)
        parser.add_argument('--n_max_lags', default=8)  # should match largest n_lags value
        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=True, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', '-en', default='grid_search', type=str)

        if namespace.model_type == 'linear':
            parser.add_argument('--n_hid_layers', default=0, type=int)
        elif namespace.model_type == 'ff':
            parser.opt_list('--n_hid_layers', default=1, options=[1, 2, 3, 4], type=int, tunable=True)
            parser.opt_list('--n_final_units', default=64, options=[64], type=int, tunable=True)
            parser.add_argument('--n_int_units', default=64, type=int)
        elif namespace.model_type == 'lstm':
            raise NotImplementedError

    else:
        raise Exception


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
    if hyperparams.export_predictions_best:
        export_predictions_best(vars(hyperparams))
