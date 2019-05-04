import os
import time
import numpy as np
import pickle
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import AE
from behavenet.training import fit
from behavenet.utils import export_latents_best, experiment_exists, get_best_model_version
from fitting.ae_model_architecture_generator import draw_archs
from data.data_generator import ConcatSessionsGenerator
import random


def main(hparams):

    # TODO: log files

    hparams = vars(hparams)

    # Blend outer hparams with architecture hparams
    hparams = {**hparams, **hparams['architecture_params']}

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(10))

    # delete 'architecture_params' key
    list_of_archs = pickle.load(open(hparams['arch_file_name'], 'rb'))
    hparams['list_index'] = list_of_archs.index(hparams['architecture_params'])

    #hparams.pop('architecture_params', None)
    print(hparams)

    # #########################
    # ### Create Experiment ###
    # #########################

    hparams['results_dir'] = os.path.join(
        hparams['tt_save_path'], hparams['lab'], hparams['expt'],
        hparams['animal'], hparams['session'])

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

    ids = {
        'lab': hparams['lab'],
        'expt': hparams['expt'],
        'animal': hparams['animal'],
        'session': hparams['session']}
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], ids, signals=[hparams['signals']],
        transforms=[hparams['transforms']], load_kwargs=[{'format': 'hdf5'}],
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])
    print('Data generator loaded')

    # ####################
    # ### CREATE MODEL ###
    # ####################

    # save out hparams as dict for easy reloading
    meta_file = os.path.join(
        hparams['results_dir'], 'test_tube_data', hparams['experiment_name'],
        'version_%i' % exp.version, 'meta_tags.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(hparams, f)
    # save out hparams as csv file
    exp.tag(hparams)
    exp.save()

    model = AE(hparams)
    model.to(hparams['device'])

    print('Model loaded')
    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # t = time.time()
    # optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    # for i in range(20):
    #     optimizer.zero_grad()
    #     batch, dataset = data_generator.next_batch('train')
    #     y, x = model(batch['images'][0])
    #     loss = torch.mean((y-batch['images'][0])**2)
    #     loss.backward()
    #     optimizer.step()
    # print('Epoch processed!')
    # print('Time elapsed: {}'.format(time.time() - t))

    fit(hparams, model, data_generator, exp, method='ae')


def get_params(strategy):

    # TODO: fix argarse bools

    parser = HyperOptArgumentParser(strategy)

    parser.add_argument('--search_type', type=str) # initial, top_n, latent_search

    namespace, extra = parser.parse_known_args()


    if namespace.search_type == 'initial':
        parser.add_argument('--arch_file_name', type=str) # file name where storing list of architectures (.pkl file)
        parser.add_argument('--n_archs', '-n', default=100, help='number of architectures to randomly sample', type=int)
        parser.add_argument('--max_nb_epochs', default=20, type=int)
        parser.add_argument('--experiment_name', '-en', default='conv_ae_initial_grid_search', type=str)
    elif namespace.search_type == 'top_n':
        parser.add_argument('--saved_initial_archs', type=str) # experiment name to look for initial architectures in
        parser.add_argument('--n_top_archs', '-n', default=5, help='number of top architectures to run', type=int)
        parser.add_argument('--max_nb_epochs', default=200, type=int)
        parser.add_argument('--experiment_name', '-en', default='conv_ae_top_n_grid_search', type=str)
    elif namespace.search_type == 'latent_search':
        parser.add_argument('--saved_top_n_archs', type=str) # experiment name to look for top n architectures in
        parser.add_argument('--max_nb_epochs', default=200, type=int)
        parser.add_argument('--experiment_name', '-en', default='conv_ae_latent_search', type=str)

    parser.add_argument('--n_input_channels', '-i', default=2, help='list of n_channels', type=int)
    parser.add_argument('--x_pixels', '-x', default=128,help='number of pixels in x dimension', type=int)
    parser.add_argument('--y_pixels', '-y', default=128,help='number of pixels in y dimension', type=int)
    parser.add_argument('--mem_limit_gb', default=5.0, type=float)
    parser.add_argument('--n_latents', '-nl', help='number of latents', type=int)
    
    parser.add_argument('--lab', '-l', default='musall', type=str)
    parser.add_argument('--expt', '-e', default='vistrained', type=str)
    parser.add_argument('--animal', '-a', default='mSM30', type=str)
    parser.add_argument('--session', '-s', default='10-Oct-2017', type=str)

    parser.add_argument('--tt_save_path', '-t', type=str)

    namespace, extra = parser.parse_known_args()

    # add testtube arguments (nb_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_nb_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_nb_cpu_workers', default=5, type=int)

    # add data generator arguments
    if os.uname().nodename == 'white-noise':
        data_dir = '/home/mattw/data/'
    elif os.uname().nodename[:3] == 'gpu':
        data_dir = '/labs/abbott/behavenet/data/'
    else:
        data_dir = ''
    parser.add_argument('--data_dir', '-d', default=data_dir, type=str)

    parser.add_argument('--signals', default='images', type=str)
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)  # dict...:(
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', default=False, type=bool)
    parser.add_argument('--batch_load', default=True, type=bool)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add training arguments
    parser.add_argument('--l2_reg', default=0, type=float)
    parser.add_argument('--val_check_interval', default=1)
    parser.add_argument('--enable_early_stop', default=False, type=bool)
    parser.add_argument('--early_stop_fraction', default=None, type=float)
    parser.add_argument('--early_stop_patience', default=None, type=float)
    
    parser.add_argument('--export_latents', default=True, type=bool)

    # add architecture arguments
    parser.add_argument('--batch_size', '-b', default=200, help='batch_size', type=int)

    # add saving arguments
    parser.add_argument('--model_type', '-m', default='ae', type=str) # ae vs vae
    
    
    parser.add_argument('--gpus_viz', default='0;1', type=str)


    # Set numpy random seed so it's not the same every call
    np.random.seed(random.randint(0, 1000))
    
    # Load in file of architectures
    if namespace.search_type == 'initial':

        if os.path.isfile(namespace.arch_file_name):
            print('Using presaved list of architectures')
            list_of_archs = pickle.load(open(namespace.arch_file_name, 'rb'))
            
        else:
            print('Creating new list of architectures and saving')
            list_of_archs = draw_archs(
                batch_size=namespace.batch_size,
                input_dim=[namespace.n_input_channels, namespace.x_pixels, namespace.y_pixels],
                n_latents=namespace.n_latents,
                n_archs=namespace.n_archs,
                check_memory=True,
                mem_limit_gb=namespace.mem_limit_gb)
            f = open(namespace.arch_file_name, "wb")
            pickle.dump(list_of_archs, f)
            f.close()

        parser.opt_list('--architecture_params', options=list_of_archs, tunable=True)

    elif namespace.search_type == 'top_n':
        # Get top n architectures in directory
        results_dir = os.path.join(namespace.tt_save_path, namespace.lab, namespace.expt,namespace.animal, namespace.session)
        best_versions = get_best_model_version(results_dir+'/test_tube_data/'+namespace.saved_initial_archs,n_best=namespace.n_top_archs)

        list_of_archs=[]
        for i_version in best_versions:
             filename = results_dir+'/test_tube_data/'+namespace.saved_initial_archs+'/version_'+str(i_version)+'/meta_tags.pkl'
             temp = pickle.load(open(namespace.arch_file_name, 'rb'))
             list_of_archs.append(temp['architecture_params'])
        parser.opt_list('--learning_rate', default=1e-3, options=[1e-4,1e-3,1e-2],type=float,tunable=True)
        parser.opt_list('--architecture_params', options=list_of_archs, tunable=True)

    elif namespace.search_type == 'latent_search':
        # Get top 1 architectures in directory

        parser.opt_list('--n_latents', '-nl', default=[4,8,12,16,20,24,28,32], help='number of latents', type=int)
    
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
        export_latents_best(vars(hyperparams))
