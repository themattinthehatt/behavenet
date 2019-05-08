import os
import time
import numpy as np
import pickle
from test_tube import HyperOptArgumentParser, Experiment
from fitting.utils import export_latents_best
from fitting.utils import experiment_exists
from fitting.utils import export_hparams
from fitting.utils import get_data_generator_inputs
from fitting.utils import get_output_dirs
from fitting.utils import get_best_model_version
from fitting.utils import add_lab_defaults_to_parser
from fitting.ae_model_architecture_generator import draw_archs
from fitting.ae_model_architecture_generator import draw_handcrafted_archs
from data.data_generator import ConcatSessionsGenerator
import random


def main(hparams):

    # TODO: log files
    # TODO: train/eval -> export_best_latents can be eval only mode

    hparams = vars(hparams)
    if hparams['model_type'] == 'conv':
        # blend outer hparams with architecture hparams
        hparams = {**hparams, **hparams['architecture_params']}
        # get index of architecture in list
        if hparams['search_type'] == 'initial':
            list_of_archs = pickle.load(open(hparams['arch_file_name'], 'rb'))
            hparams['list_index'] = list_of_archs.index(hparams['architecture_params'])
        elif hparams['search_type'] == 'latent_search':
            hparams['architecture_params']['n_ae_latents'] = hparams[
                'n_ae_latents']
            hparams['architecture_params'].pop('learning_rate', None)
    print(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(1))

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
        from behavenet.models import AE as AE
        from behavenet.training import fit as fit
        import torch
        torch_rnd_seed = torch.get_rng_state()
        hparams['model_build_rnd_seed'] = torch_rnd_seed
        model = AE(hparams)
        model.to(hparams['device'])
        torch_rnd_seed = torch.get_rng_state()
        hparams['training_rnd_seed'] = torch_rnd_seed
    elif hparams['lib'] == 'tf':
        from behavenet.models_tf import AE
        from behavenet.training_tf import fit
        model = AE(hparams)
    else:
        raise ValueError('"%s" is an invalid lib' % hparams['lib'])

    # save out hparams as csv and dict
    hparams['training_completed'] = False
    export_hparams(hparams, exp)

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

    # most important arguments
    parser.add_argument('--search_type', type=str) # latent_search, test
    parser.add_argument('--lab_example', type=str) # musall, steinmetz, markowitz
    parser.add_argument('--lib', default='tf', type=str, choices=['pytorch', 'tf'])
    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--model_type', type=str, choices=['conv', 'linear'])

    # arguments for computing resources
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)
    parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0', type=str)  # add multiple as '0;1;4' etc

    # add data generator arguments
    parser.add_argument('--signals', default='images', type=str)
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=False)
    parser.add_argument('--batch_load', action='store_true', default=True)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add fitting arguments
    parser.add_argument('--model_class', '-m', default='ae', type=str)  # ae vs vae
    parser.add_argument('--l2_reg', default=0)
    parser.add_argument('--val_check_interval', default=1)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)

    # get model-type specific arguments
    if namespace.model_type == 'conv':
        get_conv_params(namespace, parser)
    elif namespace.model_type == 'linear':
        get_linear_params(namespace, parser)
    else:
        raise ValueError('"%s" is an invalid model type')

    return parser.parse_args()


def get_linear_params(namespace, parser):

    if namespace.search_type == 'test':

        parser.add_argument('--n_ae_latents', help='number of latents', type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=50, type=int)
        parser.add_argument('--experiment_name', '-en', default='test', type=str)
        parser.add_argument('--export_latents', action='store_true', default=False)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    elif namespace.search_type == 'latent_search':

        parser.opt_list('--n_ae_latents', options=[4, 8, 12, 16, 24, 32, 64], help='number of latents', type=int, tunable=True) # warning: over 64, may need to change max_latents in architecture generator
        parser.opt_list('--learning_rate', options=[1e-4, 1e-3], type=float, tunable=True)

        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=50, type=int)
        parser.add_argument('--experiment_name', '-en', default='best', type=str)
        parser.add_argument('--export_latents', action='store_true', default=True)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)


def get_conv_params(namespace, parser):

    # get experiment-specific arguments
    if namespace.search_type == 'test':

        parser.add_argument('--n_ae_latents', help='number of latents', type=int)

        parser.add_argument('--which_handcrafted_archs', default='0')
        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=100, type=int)
        parser.add_argument('--experiment_name', '-en', default='test', type=str)
        parser.add_argument('--export_latents', action='store_true', default=False)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    elif namespace.search_type == 'initial':

        parser.add_argument('--arch_file_name', type=str) # file name where storing list of architectures (.pkl file), if exists, assumes already contains handcrafted archs!
        parser.add_argument('--n_ae_latents', help='number of latents', type=int)

        parser.add_argument('--which_handcrafted_archs', default='0;1') # empty string if you don't want any
        parser.add_argument('--n_archs', '-n', default=50, help='number of architectures to randomly sample', type=int)
        parser.add_argument('--max_nb_epochs', default=20, type=int)
        parser.add_argument('--experiment_name', '-en', default='initial_grid_search', type=str) # test
        parser.add_argument('--export_latents', action='store_true', default=False)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=False)
        parser.add_argument('--early_stop_history', default=None, type=int)

    elif namespace.search_type == 'top_n':

        parser.add_argument('--saved_initial_archs', default='initial_grid_search', type=str) # experiment name to look for initial architectures in
        parser.add_argument('--n_ae_latents', help='number of latents', type=int)

        parser.add_argument('--n_top_archs', '-n', default=5, help='number of top architectures to run', type=int)
        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=100, type=int)
        parser.add_argument('--experiment_name', '-en', default='top_n_grid_search', type=str)
        parser.add_argument('--export_latents', action='store_true', default=False)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    elif namespace.search_type == 'latent_search':

        parser.add_argument('--source_n_ae_latents', help='number of latents', type=int)

        parser.add_argument('--saved_top_n_archs', default='top_n_grid_search', type=str) # experiment name to look for top n architectures in
        parser.add_argument('--max_nb_epochs', default=500, type=int)
        parser.add_argument('--min_nb_epochs', default=100, type=int)
        parser.add_argument('--experiment_name', '-en', default='best', type=str)
        parser.add_argument('--export_latents', action='store_true', default=True)
        parser.add_argument('--export_latents_best', action='store_true', default=False)
        parser.add_argument('--enable_early_stop', action='store_true', default=True)
        parser.add_argument('--early_stop_history', default=10, type=int)

    # Load in file of architectures
    if namespace.search_type == 'test':

        which_handcrafted_archs = np.asarray(namespace.which_handcrafted_archs.split(';')).astype('int')
        list_of_archs = draw_handcrafted_archs(
            [namespace.n_input_channels, namespace.y_pixels, namespace.x_pixels],
            namespace.n_ae_latents,
            which_handcrafted_archs,
            check_memory=True,
            batch_size=namespace.approx_batch_size,
            mem_limit_gb=namespace.mem_limit_gb)
        parser.opt_list('--architecture_params', options=list_of_archs, tunable=True)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

    elif namespace.search_type == 'initial':

        if os.path.isfile(namespace.arch_file_name):
            print('Using presaved list of architectures (not appending handcrafted architectures)')
            list_of_archs = pickle.load(open(namespace.arch_file_name, 'rb'))
        else:
            print('Creating new list of architectures and saving')
            list_of_archs = draw_archs(
                batch_size=namespace.approx_batch_size,
                input_dim=[namespace.n_input_channels, namespace.y_pixels, namespace.x_pixels],
                n_ae_latents=namespace.n_ae_latents,
                n_archs=namespace.n_archs,
                check_memory=True,
                mem_limit_gb=namespace.mem_limit_gb)
            if namespace.which_handcrafted_archs:
                which_handcrafted_archs = np.asarray(namespace.which_handcrafted_archs.split(';')).astype('int')
                list_of_handcrafted_archs = draw_handcrafted_archs(
                    [namespace.n_input_channels, namespace.y_pixels, namespace.x_pixels],
                    namespace.n_ae_latents,
                    which_handcrafted_archs,
                    check_memory=True,
                    batch_size=namespace.approx_batch_size,
                    mem_limit_gb=namespace.mem_limit_gb)
                list_of_archs = list_of_archs + list_of_handcrafted_archs
            f = open(namespace.arch_file_name, "wb")
            pickle.dump(list_of_archs, f)
            f.close()
        parser.opt_list('--architecture_params', options=list_of_archs, tunable=True)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

    elif namespace.search_type == 'top_n':

        # Get top n architectures in directory
        results_dir = os.path.join(namespace.tt_save_path, namespace.lab, namespace.expt,namespace.animal, namespace.session,namespace.model_class, 'conv')
        best_versions = get_best_model_version(results_dir+'/'+str(namespace.n_ae_latents)+'_latents/test_tube_data/'+namespace.saved_initial_archs,n_best=namespace.n_top_archs)
        print(best_versions)
        list_of_archs=[]
        for version in best_versions:
             filename = results_dir+'/'+str(namespace.n_ae_latents)+'_latents/test_tube_data/'+namespace.saved_initial_archs+'/'+version+'/meta_tags.pkl'
             temp = pickle.load(open(filename, 'rb'))
             temp['architecture_params']['source_architecture'] = filename
             list_of_archs.append(temp['architecture_params'])
        parser.opt_list('--learning_rate', default=1e-3, options=[1e-4,5e-4,1e-3],type=float,tunable=True)
        parser.opt_list('--architecture_params', options=list_of_archs, tunable=True)

    elif namespace.search_type == 'latent_search':

        # Get top 1 architectures in directory
        results_dir = os.path.join(namespace.tt_save_path, namespace.lab, namespace.expt,namespace.animal, namespace.session,namespace.model_class, 'conv')
        best_version = get_best_model_version(results_dir+'/'+str(namespace.source_n_ae_latents)+'_latents/test_tube_data/'+namespace.saved_top_n_archs,n_best=1)[0]

        filename = results_dir+'/'+str(namespace.source_n_ae_latents)+'_latents/test_tube_data/'+namespace.saved_top_n_archs+'/'+best_version+'/meta_tags.pkl'
        arch = pickle.load(open(filename, 'rb'))
        arch['architecture_params']['source_architecture'] = filename
        arch['architecture_params'].pop('n_ae_latents', None)

        arch['architecture_params']['learning_rate'] = arch['learning_rate']

        # parser.add_argument('--learning_rate', default=arch['learning_rate'])
        parser.opt_list('--architecture_params', options=[arch['architecture_params']],type=float,tunable=True) # have to pass in as a list since add_argument doesn't take dict
        parser.opt_list('--n_ae_latents', options=[4,8,12,16,24,32,64], help='number of latents', type=int, tunable=True) # warning: over 64, may need to change max_latents in architecture generator


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
