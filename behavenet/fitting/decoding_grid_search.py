import os
import time
import numpy as np
import random
import torch
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.data.data_generator import ConcatSessionsGenerator
from behavenet.data.utils import get_data_generator_inputs
from behavenet.data.utils import get_region_list
from behavenet.fitting.eval import export_predictions_best
from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import export_hparams
from behavenet.fitting.utils import export_session_info_to_csv
from behavenet.fitting.utils import get_output_dirs
from behavenet.fitting.utils import get_output_session_dir
from behavenet.models import Decoder
from behavenet.training import fit


def main(hparams):

    # turn matlab-style struct into dict
    hparams = vars(hparams)
    print(hparams)

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0, 1000))
    time.sleep(np.random.uniform(0, 5))

    # #########################
    # ### Create Experiment ###
    # #########################

    # get session_dir
    hparams['session_dir'], sess_ids = get_output_session_dir(hparams)
    if not os.path.isdir(hparams['session_dir']):
        os.makedirs(hparams['session_dir'])
        export_session_info_to_csv(hparams['session_dir'], sess_ids)
    # get results_dir(session_dir + ae details),
    # expt_dir(results_dir + tt expt details)
    hparams['results_dir'], hparams['expt_dir'] = get_output_dirs(hparams)
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])
    print('')

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

    print('using data from following sessions:')
    for ids in sess_ids:
        print('%s' % os.path.join(
            hparams['tt_save_path'], ids['lab'], ids['expt'], ids['animal'],
            ids['session']))
    hparams, signals, transforms, paths = get_data_generator_inputs(
        hparams, sess_ids)
    print(hparams)
    print()
    print(paths)
    print('constructing data generator...', end='')
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], sess_ids,
        signals_list=signals, transforms_list=transforms, paths_list=paths,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])
    hparams['input_size'] = data_generator.datasets[0].dims[hparams['input_signal']][2]
    # csv order will reflect dataset order in data generator
    export_session_info_to_csv(os.path.join(
        hparams['expt_dir'], str('version_%i' % exp.version)), sess_ids)
    print('done')
    print(data_generator)

    if hparams['model_class'] == 'neural-arhmm':
         hparams['arhmm_model_path'] = os.path.join(
             os.path.dirname(data_generator.datasets[0].paths['arhmm']))
  
    # ####################
    # ### CREATE MODEL ###
    # ####################

    print('constructing model...', end='')
    torch_rnd_seed = torch.get_rng_state()
    hparams['model_build_rnd_seed'] = torch_rnd_seed
    model = Decoder(hparams)
    model.to(hparams['device'])
    model.version = exp.version
    torch_rnd_seed = torch.get_rng_state()
    hparams['training_rnd_seed'] = torch_rnd_seed

    # save out hparams as csv and dict for easy reloading
    hparams['training_completed'] = False
    export_hparams(hparams, exp)
    print('done')

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
    parser.add_argument('--tt_save_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_type', default='ff', choices=['ff', 'ff-mv', 'linear', 'linear-mv', 'lstm'], type=str)
    parser.add_argument('--model_class', default='neural-ae', choices=['neural-ae', 'neural-arhmm'], type=str)
    parser.add_argument('--sessions_csv', default='', type=str)  # specify multiple sessions

    # arguments for computing resources (infer n_gpu_workers from visible gpus)
    parser.add_argument('--tt_n_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_trials', default=100000, type=int)
    parser.add_argument('--tt_n_cpu_workers', default=5, type=int)
    parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0;1', type=str)

    # add data generator arguments
    parser.add_argument('--reg_list', default='none', type=str, choices=['none', 'arg', 'all'])
    parser.add_argument('--subsample_regions', default='none', choices=['none', 'single', 'loo'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--as_numpy', action='store_true', default=False)
    parser.add_argument('--batch_load', action='store_true', default=False)
    parser.add_argument('--rng_seed', default=0, type=int)

    # add fitting arguments
    parser.add_argument('--val_check_interval', default=1)

    # get lab-specific arguments
    namespace, extra = parser.parse_known_args()
    add_lab_defaults_to_parser(parser, namespace.lab_example)
    namespace, extra = parser.parse_known_args()  # ugly

    # add regions to opt_list if desired

    if namespace.reg_list == 'all':
        parser.opt_list('--region', options=get_region_list(namespace), type=str, tunable=True)
    elif namespace.reg_list == 'arg':
        parser.add_argument('--region', default='all', type=str)
    elif namespace.reg_list == 'none':
        parser.add_argument('--region', value='all', type=str)
    else:
        raise ValueError(
            '"%s" is not a valid region_list' % namespace.region_list)

    get_decoding_params(namespace, parser)

    return parser.parse_args()


def get_decoding_params(namespace, parser):

    # add neural arguments (others are dataset-specific)
    parser.add_argument('--neural_thresh', default=1.0, help='minimum firing rate for spikes (Hz)', type=float)

    # add data arguments
    if namespace.model_class == 'neural-ae':
        # ae arguments
        parser.add_argument('--ae_experiment_name', type=str)
        parser.add_argument('--n_ae_latents', type=int)
        parser.add_argument('--ae_version', default='best')
        parser.add_argument('--ae_model_type', default='conv')
    elif namespace.model_class == 'neural-arhmm':
        # ae arguments
        parser.add_argument('--n_ae_latents', default=12, type=int)
        parser.add_argument('--ae_model_type', default='conv')
        # arhmm arguments
        # TODO: add ae_experiment_name/model_type/model_class to arhmm ids?
        parser.add_argument('--arhmm_experiment_name', type=str)
        parser.add_argument('--n_arhmm_states', default=12, type=int)
        parser.add_argument('--kappa', default=1e+06, type=float)
        parser.add_argument('--noise_type', default='gaussian', type=str)
        parser.add_argument('--arhmm_version', default='best')
    elif namespace.model_class == 'neural-dlc':
        raise NotImplementedError
    else:
        raise ValueError('"%s" is an invalid model class' % namespace.model_class)

    parser.add_argument('--enable_early_stop', action='store_true', default=True)
    parser.add_argument('--early_stop_history', default=10, type=float)
    parser.add_argument('--min_n_epochs', default=1, type=int)
    parser.add_argument('--max_n_epochs', default=500, type=int)
    parser.add_argument('--activation', default='relu', choices=['linear', 'relu', 'lrelu', 'sigmoid', 'tanh'])

    if namespace.search_type == 'best':

        import pickle
        from behavenet.fitting.utils import get_best_model_version

        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=True, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', '-en', default='best', type=str)
        parser.add_argument('--decoder_experiment_name', default='grid_search', type=str)

        # load best model params
        namespace, extra = parser.parse_known_args()
        hparams_tmp = vars(namespace)
        hparams_tmp['experiment_name'] = hparams_tmp['decoder_experiment_name']
        _, _, expt_dir = get_output_dirs(hparams_tmp)
        best_version = get_best_model_version(expt_dir)[0]
        best_file = os.path.join(expt_dir, best_version, 'meta_tags.pkl')
        print('Loading best discrete decoder from %s' % best_file)
        with open(best_file, 'rb') as f:
            hparams_best = pickle.load(f)
        # get model params
        learning_rate = hparams_best['learning_rate']
        n_lags = hparams_best['n_lags']
        l2_reg = hparams_best['l2_reg']
        n_max_lags = hparams_best['n_max_lags']
        n_final_units = hparams_best['n_final_units']
        n_int_units = hparams_best['n_int_units']
        n_hid_layers = hparams_best['n_hid_layers']

        parser.add_argument('--learning_rate', default=learning_rate, type=float)
        parser.add_argument('--n_lags', default=n_lags, type=int)
        parser.add_argument('--l2_reg', default=l2_reg, type=float)
        parser.add_argument('--n_max_lags', default=n_max_lags)  # should match largest value in --n_lags options
        parser.add_argument('--n_hid_layers', default=n_hid_layers, type=int)
        parser.add_argument('--n_final_units', default=n_final_units, type=int)
        parser.add_argument('--n_int_units', default=n_int_units, type=int)
        
    elif namespace.search_type == 'test':

        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--n_lags', default=4, type=int)
        parser.add_argument('--l2_reg', default=1e-3, type=float)
        parser.add_argument('--n_max_lags', default=8)  # should match largest value in --n_lags options
        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=False, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', default='test', type=str)

        if namespace.model_type == 'linear' or namespace.model_type == 'linear-mv':
            parser.add_argument('--n_hid_layers', default=0, type=int)
        elif namespace.model_type == 'ff' or namespace.model_type == 'ff-mv':
            parser.add_argument('--n_hid_layers', default=1, type=int)
            parser.add_argument('--n_final_units', default=64, type=int)
            parser.add_argument('--n_int_units', default=64, type=int)
        elif namespace.model_type == 'lstm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is an invalid model type' % namespace.model_type)

    elif namespace.search_type == 'shuffle':
        # shuffle discrete labels as baseline

        import pickle
        from behavenet.fitting.utils import get_best_model_version

        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=False, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', '-en', default='shuffle', type=str)
        parser.add_argument('--decoder_experiment_name', default='grid_search', type=str)
        parser.add_argument('--n_shuffles', type=int)

        try:
            # load best model params
            namespace, extra = parser.parse_known_args()
            hparams_tmp = vars(namespace)
            hparams_tmp['experiment_name'] = hparams_tmp['decoder_experiment_name']
            _, _, expt_dir = get_output_dirs(hparams_tmp)
            best_version = get_best_model_version(expt_dir)[0]
            best_file = os.path.join(expt_dir, best_version, 'meta_tags.pkl')
            print('Loading best discrete decoder from %s' % best_file)
            with open(best_file, 'rb') as f:
                hparams_best = pickle.load(f)
            # get model params
            learning_rate = hparams_best['learning_rate']
            n_lags = hparams_best['n_lags']
            l2_reg = hparams_best['l2_reg']
            n_max_lags = hparams_best['n_max_lags']
            n_final_units = hparams_best['n_final_units']
            n_int_units = hparams_best['n_int_units']
            n_hid_layers = hparams_best['n_hid_layers']
        except:
            print('Could not load best model; reverting to defaults')
            # choose reasonable defaults
            learning_rate = 1e-3
            n_lags = 4
            l2_reg = 1e-3
            n_max_lags = 8
            n_final_units = 64
            n_int_units = 64
            if namespace.model_type == 'linear':
                n_hid_layers = 0
            else:
                n_hid_layers = 1

        parser.add_argument('--learning_rate', default=learning_rate, type=float)
        parser.add_argument('--n_lags', default=n_lags, type=int)
        parser.add_argument('--l2_reg', default=l2_reg, type=float)
        parser.add_argument('--n_max_lags', default=n_max_lags)  # should match largest value in --n_lags options
        parser.add_argument('--n_hid_layers', default=n_hid_layers, type=int)
        parser.add_argument('--n_final_units', default=n_final_units, type=int)
        parser.add_argument('--n_int_units', default=n_int_units, type=int)
        parser.opt_list('--shuffle_rng_seed', default=0, options=list(np.arange(namespace.n_shuffles)), type=int, tunable=True)

    elif namespace.search_type == 'grid_search':

        # parser.opt_list('--learning_rate', default=1e-3, options=[1e-3], type=float, tunable=True)
        # parser.opt_list('--n_lags', default=0, options=[0, 1, 2, 4, 8], type=int, tunable=True)
        # parser.opt_list('--l2_reg', default=0, options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], type=float, tunable=True)
        # regular for decoding ae latents
        parser.opt_list('--learning_rate', default=1e-3, options=[1e-2, 1e-3, 1e-4], type=float, tunable=True)
        parser.opt_list('--n_lags', default=0, options=[0, 1, 2, 4, 8], type=int, tunable=True)
        parser.opt_list('--l2_reg', default=0, options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], type=float, tunable=True)
        parser.add_argument('--n_max_lags', default=8)  # should match largest n_lags value
        parser.add_argument('--export_predictions', action='store_true', default=False, help='export predictions for each decoder')
        parser.add_argument('--export_predictions_best', action='store_true', default=True, help='export predictions best decoder in experiment')
        parser.add_argument('--experiment_name', '-en', default='grid_search', type=str)

        if namespace.model_type == 'linear' or namespace.model_type == 'linear-mv':
            parser.add_argument('--n_hid_layers', default=0, type=int)
        elif namespace.model_type == 'ff' or namespace.model_type == 'ff-mv':
            parser.opt_list('--n_hid_layers', default=1, options=[1, 2, 3, 4], type=int, tunable=True)
            parser.opt_list('--n_final_units', default=64, options=[64], type=int, tunable=True)
            parser.add_argument('--n_int_units', default=64, type=int)
        elif namespace.model_type == 'lstm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is an invalid model type' % namespace.model_type)

    else:
        raise ValueError('"%s" is not a valid search type' % namespace.search_type)


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
    print('Total fit time: {}'.format(time.time() - t))
    if hyperparams.export_predictions_best \
            and hyperparams.subsample_regions == 'none':
        export_predictions_best(vars(hyperparams))
