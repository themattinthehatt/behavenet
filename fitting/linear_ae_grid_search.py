import os
import time
import numpy as np
from test_tube import HyperOptArgumentParser, Experiment
from fitting.utils import export_latents_best
from fitting.utils import experiment_exists
from fitting.utils import export_hparams
from fitting.utils import get_data_generator_inputs
from fitting.utils import get_output_dirs
from fitting.utils import add_lab_defaults_to_parser
from data.data_generator import ConcatSessionsGenerator
import random


def main(hparams):

    # TODO: log files
    # TODO: train/eval -> export_best_latents can be eval only mode

    hparams = vars(hparams)
    print(hparams)

    if hparams['lib'] == 'pytorch':
        from behavenet.models import AE as AE
        from behavenet.training import fit as fit
        import torch
    elif hparams['lib'] == 'tf':
        from behavenet.models_tf import AE
        from behavenet.training_tf import fit
    else:
        raise ValueError('"%s" is an invalid lib' % hparams['lib'])

    # Blend outer hparams with architecture hparams
    # hparams = {**hparams, **hparams['architecture_params']}

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(10))

    # #########################
    # ### Create Experiment ###
    # #########################

    # get session_dir, results_dir (session_dir + ae details), expt_dir (
    # results_dir + experiment details)
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
    print('Data generator loaded')

    # ####################
    # ### CREATE MODEL ###
    # ####################

    if hparams['lib'] == 'pytorch':
        torch_rnd_seed = torch.get_rng_state()
        hparams['model_build_rnd_seed'] = torch_rnd_seed

    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)

    model = AE(hparams)
    model.to(hparams['device'])

    if hparams['lib'] == 'pytorch':
        torch_rnd_seed = torch.get_rng_state()
        hparams['training_rnd_seed'] = torch_rnd_seed

    print('Model loaded')

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    fit(hparams, model, data_generator, exp, method='ae')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    parser.add_argument('--search_type', type=str) # latent_search, test
    parser.add_argument('--lab_example', type=str) # musall, steinmetz, markowitz
    parser.add_argument('--lib', default='tf', type=str, choices=['pytorch', 'tf'])

    namespace, extra = parser.parse_known_args()

    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--data_dir', '-d', type=str)

    if namespace.search_type == 'test':

        parser.add_argument('--n_ae_latents', help='number of latents', type=int)

        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=50, type=int)
        parser.add_argument('--experiment_name', '-en', default='test', type=str)
        parser.add_argument('--export_latents', action='store_true', default=False)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    elif namespace.search_type == 'latent_search':

        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=50, type=int)
        parser.add_argument('--experiment_name', '-en', default='best', type=str)
        parser.add_argument('--export_latents', action='store_true', default=True)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    parser.add_argument('--mem_limit_gb', default=8.0, type=float)

    add_lab_defaults_to_parser(parser, namespace.lab_example)

    parser.add_argument('--model_class', '-m', default='ae', type=str) # ae vs vae
    parser.add_argument('--model_type', default='linear', type=str)

    namespace, extra = parser.parse_known_args()

    # add testtube arguments (nb_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)

    # add data generator arguments
    parser.add_argument('--signals', default='images', type=str)
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)  # dict...:(
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=False)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)

    parser.add_argument('--l2_reg', default=0)
    parser.add_argument('--val_check_interval', default=1)

    # add saving arguments
    parser.add_argument('--gpus_viz', default='0', type=str)  # add multiple as 0;1;4 etc

    # Set numpy random seed so it's not the same every call

    # Load in file of architectures
    if namespace.search_type == 'test':
        parser.add_argument('--learning_rate', default=1e-3, type=float)
    elif namespace.search_type == 'latent_search':
        parser.opt_list('--learning_rate', options=[1e-4, 1e-3], type=float, tunable=True)
        parser.opt_list('--n_ae_latents', options=[4, 8, 12, 16, 24, 32, 64], help='number of latents', type=int, tunable=True) # warning: over 64, may need to change max_latents in architecture generator

    return parser.parse_args()


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
    if hyperparams.export_latents_best:
        print('Exporting latents from current best model in experiment')
        export_latents_best(vars(hyperparams))
