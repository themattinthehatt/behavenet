import os
import time
import numpy as np
import pickle
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import AE
from fitting.ae_model_architecture_generator import draw_archs
from data.data_generator import ConcatSessionsGenerator
import random

def main(hparams):

    hparams = vars(hparams)
    # Blend outer hparams with architecture hparams
    hparams = {**hparams,**hparams['architecture_params']}

    # delete 'architecture_params' key
    list_of_archs = pickle.load(open(hparams['arch_file_name'],'rb'))
    hparams['list_index'] = list_of_archs.index(hparams['architecture_params'])

    hparams.pop('architecture_params', None)
    print(hparams)

    # Set numpy random seed so it's not the same every call
    np.random.seed(random.randint(0,1000))
    # Start at random times (so test tube creates separate folders)
    time.sleep(np.random.randint(10))

    # #########################
    # ### Create Experiment ###
    # #########################

    exp = Experiment(name=hparams['experiment_name'],
            debug=False,
            save_dir=hparams['tt_save_path'])
    exp.tag(hparams)
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
        transforms=[hparams['transforms']], load_kwargs=[hparams['load_kwargs']],
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])

    print('Data generator loaded')
    # ####################
    # ### CREATE MODEL ###
    # ####################

    model = AE(hparams)
    model.to(hparams['device'])

    print('Model loaded')
    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # fit(hparams,model,data_generator)

def get_params(strategy):
    parser = HyperOptArgumentParser(strategy)

    # add data generator arguments
    if os.uname().nodename == 'white-noise':
        data_dir = '/home/mattw/data/'
    elif os.uname().nodename[:3] == 'gpu':
        data_dir = '/labs/abbott/behavenet/data/'
    else:
        data_dir = ''
    parser.add_argument('--data_dir', '-d', default=data_dir, help='')
    parser.add_argument('--lab', '-l', default='musall', help='')
    parser.add_argument('--expt', '-e', default='vistrained', help='')
    parser.add_argument('--animal', '-a', default='mSM30', help='')
    parser.add_argument('--session', '-s', default='10-Oct-2017', help='')
    parser.add_argument('--signals', default='images')
    parser.add_argument('--transforms', default=None)
    parser.add_argument('--load_kwargs', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--as_numpy', default=False)
    parser.add_argument('--batch_load', default=True)
    parser.add_argument('--rng_seed', default=0)


    # add training arguments

    # parser.add_argument('--arch_file_name', type=str)
    parser.add_argument('--file_name', '-f', help='file for where to save list of architectures (without extension)')
    parser.add_argument('--n_archs', '-n', help='number of architectures to randomly sample',type=int)
    parser.add_argument('--input_channels', '-i', help='list of n_channels', type=int)
    parser.add_argument('--x_pixels', '-x', help='number of pixels in x dimension', type=int)
    parser.add_argument('--y_pixels', '-y', help='number of pixels in y dimension', type=int)
    parser.add_argument('--n_latents', '-nl', help='number of latents', type=int)
    parser.add_argument('--batch_size', '-b', help='batch_size', type=int)
    parser.add_argument('--arch_file_name', type=str) # file name where storing list of architectures (.pkl file)
    namespace, extra = parser.parse_known_args()

    # Saving arguments
    parser.add_argument('--model_type', '-m', help='ae', type=int) # ae vs vae

    parser.add_argument('--tt_save_path','-t',type=str)
    parser.add_argument('--experiment_name','-m',default='conv_ae_grid_search',type=str)
    parser.add_argument('--gpus_viz', default='0;1', type=str)
    
    # Load in file of architectures

    if os.path.isfile(namespace.arch_file_name):
        print('Using presaved list of architectures')
        list_of_archs = pickle.load(open(namespace.arch_file_name,'rb'))
        
    else:
        print('Creating new list of architectures and saving')
        list_of_archs = draw_archs(batch_size=namespace.batch_size,input_dim=[namespace.input_channels,namespace.x_pixels,namespace.y_pixels], n_latents=namespace.n_latents, n_archs=namespace.n_archs, check_memory=False)
        f = open(namespace.arch_file_name,"wb")
        pickle.dump(list_of_archs,f)
        f.close()

    parser.opt_list('--architecture_params', options=list_of_archs,tunable=True)
    return parser.parse_args()

if __name__ == '__main__':
    hyperparams = get_params('grid_search')

    if hyperparams.device=='cuda':
        hyperparams.optimize_parallel_gpu(
                main,
                gpu_ids=hyperparams.gpus_viz.split(';'),
                nb_trials=500,
                nb_workers=100
            )
    elif hyperparams.device=='cpu':
        hyperparams.optimize_parallel_cpu(
                main,
                nb_trials=500,
                nb_workers=10
            )