import os
import time
import numpy as np
import random
import pickle
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import NN, LSTM
from behavenet.training import fit
from behavenet.utils import export_predictions_best, experiment_exists
from data.data_generator import ConcatSessionsGenerator
from data.transforms import Threshold


def main(hparams):
    # TODO: log files

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(0, 10))

    # turn matlab-style struct into dict
    hparams = vars(hparams)
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
    exp.tag(hparams)
    exp.save()


    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

    if hparams['neural_thresh'] > 0 and hparams['neural_type'] == 'spikes':
        neural_transforms = Threshold(
            threshold=hparams['neural_thresh'],
            bin_size=hparams['neural_bin_size'])
    else:
        neural_transforms = None  # neural_region
    neural_kwargs = None

    # get model-specific signals/transforms/load_kwargs
    if hparams['model_name'] == 'neural-ae':
        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'ae'

        signals = ['neural', 'ae']

        ae_transforms = None

        ae_dir = os.path.join(
            hparams['results_dir'], 'test_tube_data',
            hparams['ae_experiment_name'])

        ae_kwargs = {  # TODO: base_dir + ids (here or in data generator?)
            'model_dir': ae_dir,
            'model_version': hparams['ae_version']}

        transforms = [neural_transforms, ae_transforms]
        load_kwargs = [neural_kwargs, ae_kwargs]

        hparams['output_size'] = hparams['n_ae_latents']

    elif hparams['model_name'] == 'neural-arhmm':
        hparams['input_signal'] = 'neural'
        hparams['output_signal'] = 'arhmm'

        signals = ['neural', 'arhmm']

        arhmm_transforms = None

        arhmm_dir = os.path.join(
            hparams['results_dir'], 'test_tube_data',
            hparams['arhmm_experiment_name'])
        arhmm_kwargs = {  # TODO: base_dir + ids (here or in data generator?)
            'model_dir': arhmm_dir,
            'model_version': hparams['arhmm_version']}

        transforms = [neural_transforms, arhmm_transforms]
        load_kwargs = [neural_kwargs, arhmm_kwargs]

        hparams['output_size'] = hparams['n_arhmm_latents']

    else:
        raise ValueError('"%s" is an invalid model_name' % hparams['model_name'])

    print('building data generator')
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

    if hparams['model_name'] == 'neural-ae':
        hparams['noise_dist'] = 'gaussian'
    elif hparams['model_name'] == 'neural-arhmm':
        hparams['noise_dist'] = 'categorical'
    else:
        raise ValueError('"%s" is an invalid model_name' % hparams['model_name'])

    # save out hparams as dict for easy reloading
    meta_file = os.path.join(
        hparams['results_dir'], 'test_tube_data', hparams['experiment_name'],
        'version_%i' % exp.version, 'meta_tags.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(hparams, f)

    if hparams['model_type'] == 'ff' or hparams['model_type'] == 'linear':
        model = NN(hparams)
    elif hparams['model_type'] == 'lstm':
        model = LSTM(hparams)
    else:
        raise ValueError('"%s" is an invalid model_type' % hparams['model_type'])

    model.to(hparams['device'])

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

    # batch, dataset = data_generator.next_batch('train')
    # x = model(batch['neural'][0])

    fit(hparams, model, data_generator, exp, method='nll')


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    parser.opt_list('--model_name', default='neural-ae', options=['neural-ae', 'neural-arhmm'], type=str, tunable=False)
    parser.opt_list('--model_type', default='ff', options=['ff', 'linear', 'lstm'], type=str)
    namespace, extra = parser.parse_known_args()
    model_name = namespace.model_name
    model_type = namespace.model_type

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
    parser.add_argument('--lab', '-l', default='musall', type=str)
    parser.add_argument('--expt', '-e', default='vistrained', type=str)
    parser.add_argument('--animal', '-a', default='mSM30', type=str)
    parser.add_argument('--session', '-s', default='10-Oct-2017', type=str)

    # data generator arguments
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', default=False, type=bool)
    parser.add_argument('--batch_load', default=False, type=bool)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add training arguments
    parser.add_argument('--val_check_interval', default=1)
    parser.add_argument('--enable_early_stop', default=True, type=bool)
    parser.add_argument('--early_stop_history', default=10, type=float)
    parser.add_argument('--min_nb_epochs', default=1, type=int)
    parser.add_argument('--max_nb_epochs', default=100, type=int)
    # parser.add_argument('--export_latents', default=False, type=bool)

    # add saving arguments
    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--experiment_name', '-en', default='decoder_grid_search', type=str)
    parser.add_argument('--gpus_viz', default='0;1', type=str)
    parser.add_argument('--export_predictions', default=False, type=bool, help='export predictions for each decoder')
    parser.add_argument('--export_predictions_best', default=True, type=bool, help='export predictions best decoder in experiment')

    # add model hyperparameters
    parser.opt_list('--learning_rate', default=1e-3, options=[1e-2, 1e-3, 1e-4], type=float, tunable=True)
    # parser.opt_list('--n_lags', default=0, options=[0, 1, 2, 4, 8, 16], type=int, tunable=True)
    # parser.opt_list('--l2_reg', default=0, options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], type=float, tunable=True)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--n_lags', default=4, type=int)
    parser.add_argument('--l2_reg', default=1e-3, type=float)
    parser.add_argument('--n_max_lags', default=16)  # should match largest value in --n_lags options
    parser.opt_list('--activation', default='relu', options=['linear', 'relu', 'lrelu', 'sigmoid', 'tanh'], tunable=False)
    if model_type == 'linear':
        parser.add_argument('--n_hid_layers', default=0, type=int, tunable=False)
    elif model_type == 'ff':
        # parser.opt_list('--n_hid_layers', default=1, options=[1, 2], type=int, tunable=True)
        # parser.opt_list('--n_final_units', default=16, options=[16, 32, 64], type=int, tunable=True)
        parser.add_argument('--n_hid_layers', default=1, type=int)
        parser.add_argument('--n_final_units', default=16, type=int)
        parser.add_argument('--n_int_units', default=64, type=int)
    elif model_type == 'lstm':
        raise NotImplementedError

    # add neural arguments
    parser.add_argument('--neural_thresh', default=1.0, help='minimum firing rate for spikes (Hz)', type=float)
    parser.add_argument('--neural_bin_size', default=None, help='ms')
    parser.opt_list('--neural_type', default='spikes', options=['spikes', 'wf'])
    parser.opt_list('--neural_region', default='all', options=['all', 'single', 'loo'])

    # add data arguments
    if model_name == 'neural-ae':
        # ae arguments
        #parser.opt_list('--ae_view', default='both', options=['both', 'face', 'body', 'full'])
        parser.add_argument('--ae_experiment_name', type=str)
        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--ae_version', default='best')
    elif model_name == 'neural-arhmm':
        # ae arguments
        parser.opt_list('--ae_view', default='both', options=['both', 'face', 'body', 'full'])
        parser.add_argument('--n_ae_latents', default=12, type=int)
        # arhmm arguments
        parser.add_argument('--arhmm_experiment_name', type=str)
        parser.add_argument('--n_arhmm_latents', default=12, type=int)
        parser.add_argument('--arhmm_version', default='best')
    elif model_name == 'neural-dlc':
        raise NotImplementedError

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
    if hyperparams.export_predictions_best:
        export_predictions_best(vars(hyperparams))
