import os
import time
import numpy as np
import random
import pickle
import torch
import torch.nn.functional as F
from tqdm.auto import trange
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from test_tube import HyperOptArgumentParser, Experiment
from behavenet.models import NeuralNetDecoderLaggedSLDS
from behavenet.training import fit
from behavenet.fitting.eval import export_predictions_best
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import export_hparams
from behavenet.fitting.utils import get_data_generator_inputs
from behavenet.fitting.utils import get_output_dirs
from behavenet.fitting.utils import add_lab_defaults_to_parser
from behavenet.fitting.utils import get_output_session_dir
from behavenet.fitting.utils import export_session_info_to_csv
from behavenet.data.data_generator import ConcatSessionsGenerator


def whiten_all(data_dict, center=True, mu=None, L=None):
    if mu is None:
        if L is None:
            non_nan = lambda x: x[~np.isnan(np.reshape(x, (x.shape[0], -1))).any(1)]
            meancov = lambda x: (x.mean(0), np.cov(x, rowvar=False, bias=1))
           
            mu, Sigma = meancov(np.concatenate(list(map(non_nan, data_dict))))
            L = np.linalg.cholesky(Sigma)
            
    contig = partial(np.require, dtype=np.float64, requirements='C')
    offset = 0. if center else mu
    apply_whitening = lambda x:  np.linalg.solve(L, (x-mu).T).T + offset

    return [contig(apply_whitening(v)) for v in data_dict], mu, L


def get_sample(slds, latent_predictions, state_log_predictions, x_covs, x_scale, z_scale, parallel_inputs):
    idx = parallel_inputs[0]
    iter = parallel_inputs[1]
    T_smpl = latent_predictions[idx].shape[0]
    slds.add_data(latent_predictions[idx], x_covs, state_log_predictions[idx], x_scale=x_scale, z_scale=z_scale)
    states = slds.states_list.pop()
    z_smpls = []
    x_smpls = []
    ll_smpls = []
    for itr in trange(1000):
        states.resample()
        z_smpls.append(states.stateseq.copy())
        x_smpls.append(states.gaussian_states.copy())
        ll_smpls.append(states.log_likelihood())
    return [z_smpls, x_smpls, ll_smpls, idx, iter]


def get_last_sample(slds, latent_predictions, state_log_predictions, x_covs, x_scale, z_scale, parallel_inputs):
    idx = parallel_inputs[0]
    iter = parallel_inputs[1]
    T_smpl = latent_predictions[idx].shape[0]
    slds.add_data(latent_predictions[idx], x_covs, state_log_predictions[idx], x_scale=x_scale, z_scale=z_scale)
    states = slds.states_list.pop()
    z_smpls = []
    x_smpls = []
    ll_smpls = []
    for itr in trange(1000):
        states.resample()
        z_smpls.append(states.stateseq.copy())
        x_smpls.append(states.gaussian_states.copy())
        ll_smpls.append(states.log_likelihood())
    states.E_step()
    return [z_smpls[-1], x_smpls[-1], ll_smpls, idx, iter, states.expected_states]


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

    # check to see if experiment already exists
    # if experiment_exists(hparams):
    #     print('Experiment exists! Aborting fit')
    #     return

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

    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], sess_ids,
        signals=signals, transforms=transforms, load_kwargs=load_kwargs,
        device=hparams['device'], as_numpy=hparams['as_numpy'],
        batch_load=hparams['batch_load'], rng_seed=hparams['rng_seed'])
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

        trial_idxs[data_type] = data_generator.batch_indxs[0][data_type] 

        latents[data_type] = [data_generator.datasets[0][i_trial]['ae'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        latent_predictions[data_type] = [data_generator.datasets[0][i_trial]['ae_predictions'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        state_log_predictions[data_type] = [ F.log_softmax(torch.tensor(data_generator.datasets[0][i_trial]['arhmm_predictions'][:]).float(),dim=1).cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]
        states[data_type] = [data_generator.datasets[0][i_trial]['arhmm'][:].cpu().detach().numpy()[hparams['n_max_lags']:-hparams['n_max_lags']] for i_trial in trial_idxs[data_type]]

    hparams['total_train_length'] = len(trial_idxs['train'])*data_generator.datasets[0][0]['images'].shape[0]
    export_hparams(hparams, exp)

    print('Model loaded')
    print(trial_idxs['val'])

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    no_nan_predicts = np.concatenate([lat for lat in latent_predictions['train'] ],axis=0)
    no_nan_latents =  np.concatenate([lat for lat in latents['train'] ],axis=0)
    x_covs = np.cov((no_nan_predicts - no_nan_latents).T)

    # Load arhmm model
    arhmm = pickle.load(open(os.path.join(hparams['arhmm_model_path'],'best_val_model.pt'),'rb'))

    slds = NeuralNetDecoderLaggedSLDS(arhmm)

    if hparams['search_type']=='grid_search':
        ## Get val samples
        st = time.time()
        func = partial(get_sample, slds, latent_predictions['val'],state_log_predictions['val'], x_covs, hparams['x_scale'], hparams['z_scale'])
        num_cores = multiprocessing.cpu_count()
        print(num_cores)
        if hparams['lab']=='musall':
            n_val=10
        else:
            n_val=2
        n_inputs = len(latent_predictions['val'][0:n_val])
        listed_inputs = [[i,j] for i in range(n_inputs) for j in range(hparams['n_samples'])]
        parallel_outputs = Parallel(n_jobs=11*6)(delayed(func)(i) for i in listed_inputs)
        et = time.time()
        print((et-st)/60)

        # Get validation samples/MSE
        which_trials = np.asarray([parallel_outputs[i][3] for i in range(len(listed_inputs))])
        val_mse=0
        for val_idx in range(n_inputs):
            x_smpls = parallel_outputs[np.where(which_trials==val_idx)[0][0]][1][-500:]
            x_smpls = [x[:, :hparams['n_ae_latents']] for x in x_smpls]
            #mean_sample = np.mean(np.stack([parallel_outputs[i][1][:,:hparams['n_ae_latents']] for i in np.where(which_trials==val_idx)[0]]),axis=0)
            val_mse +=  np.mean((latents['val'][val_idx] - np.mean(x_smpls[-500:],axis=0))**2)
        val_mse /=n_inputs

        exp.log({'val_loss': val_mse})
        exp.save()

    elif hparams['search_type']=='best':
        print(trial_idxs['test'])
        st = time.time()
        if hparams['n_samples']>1:
            func = partial(get_last_sample, slds, latent_predictions['test'],state_log_predictions['test'], x_covs, hparams['x_scale'], hparams['z_scale'])
        elif hparams['n_samples']==1:
            func = partial(get_sample, slds, latent_predictions['test'],state_log_predictions['test'], x_covs, hparams['x_scale'], hparams['z_scale'])
        num_cores = multiprocessing.cpu_count()
        print(num_cores)
        n_inputs = len(latent_predictions['test'])
        listed_inputs = [[i,j] for i in range(n_inputs) for j in range(hparams['n_samples'])]
        parallel_outputs = Parallel(n_jobs=11*6)(delayed(func)(i) for i in listed_inputs)
        et = time.time()
        print((et-st)/60)

        # Get test samples/MSE
        # whitened_latents={}
        
        # _, mu, L = whiten_all(latents['train'])
        # wl, _, _ = whiten_all(latents['test'],mu=mu, L=L)
        # whitened_latents['test'] = wl

        which_trials = np.asarray([parallel_outputs[i][3] for i in range(len(listed_inputs))])
        test_mse=np.zeros((n_inputs,))
        test_mse_nonlinear = np.zeros((n_inputs,))
        test_mse_baseline = np.zeros((n_inputs,))
        test_samples = [None]*n_inputs
        expected_z_samples = [None]*n_inputs
        for test_idx in range(n_inputs):

            if hparams['n_samples']>1:
                x_smpls = np.stack([parallel_outputs[i][1][:,:hparams['n_ae_latents']] for i in np.where(which_trials==test_idx)[0]])
                expected_z_smpls = np.stack([parallel_outputs[i][5] for i in np.where(which_trials==test_idx)[0]])
                expected_z_samples[test_idx] = expected_z_smpls
            else:
                x_smpls = parallel_outputs[np.where(which_trials==test_idx)[0][0]][1][-500:]
                x_smpls = [x[:, :hparams['n_ae_latents']] for x in x_smpls]

            test_samples[test_idx] = x_smpls    
            mean_sample = np.mean(test_samples[test_idx],axis=0)
            
            # whitened_sample, _, _ = whiten_all([mean_sample],mu=mu, L=L)
            # whitened_pred, _, _ = whiten_all([latent_predictions['test'][test_idx]],mu=mu, L=L)
            # whitened_baseline, _, _ = whiten_all([no_nan_latents.mean(0)],mu=mu, L=L)
            # test_mse[test_idx] =  np.mean((whitened_latents['test'][test_idx] - whitened_sample)**2)
            # test_mse_nonlinear[test_idx] = np.mean((whitened_latents['test'][test_idx] - whitened_pred)**2)
            # test_mse_baseline[test_idx] = np.mean((whitened_latents['test'][test_idx] - whitened_baseline)**2)

            test_mse[test_idx] =  np.mean((latents['test'][test_idx] - np.mean(x_smpls,axis=0))**2)
            test_mse_nonlinear[test_idx] = np.mean((latents['test'][test_idx] - latent_predictions['test'][test_idx])**2)
            test_mse_baseline[test_idx] = np.mean((latents['test'][test_idx] - no_nan_latents.mean(0))**2)

        # prob_preds= np.exp(np.concatenate(state_log_predictions['test']))
        # preds= np.argmax(np.exp(np.concatenate(state_log_predictions['test'])),axis=1)
        # acts = np.concatenate(states['test'])

        # state_prediction_acc = np.sum(acts==preds)/acts.shape[0]
        # state_prediction_baseline_acc = np.sum(acts==0)/acts.shape[0]
        filepath = os.path.join(
            hparams['results_dir'], 'test_tube_data',
            hparams['experiment_name'],
            'version_%i' % exp.version,
            'test_outputs')
        if hparams['n_samples']>1:
            np.savez(filepath, arhmm_dec_samples=test_samples, expected_z_samples=expected_z_samples, latents = latents['test'], nonlinear_dec=latent_predictions['test'], test_mse_nonlinear=test_mse_nonlinear, test_mse=test_mse, test_mse_baseline=test_mse_baseline, states=states['test'], state_log_predictions=state_log_predictions['test'])
        elif hparams['n_samples']==1:
            np.savez(filepath, arhmm_dec_samples=test_samples, latents = latents['test'], nonlinear_dec=latent_predictions['test'], test_mse_nonlinear=test_mse_nonlinear, test_mse=test_mse, test_mse_baseline=test_mse_baseline, states=states['test'], state_log_predictions=state_log_predictions['test'])

        # exp.log({'val_loss': val_mse})
    
    exp.save()
    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams, exp)


def get_params(strategy):

    parser = HyperOptArgumentParser(strategy)

    # most important arguments
    parser.add_argument('--search_type', type=str)  # grid_search, test
    parser.add_argument('--lab_example', type=str)  # musall, steinmetz, markowitz
    parser.add_argument('--tt_save_path', '-t', type=str)
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--model_class', default='arhmm-decoding', type=str)
    parser.add_argument('--model_type', default=None, type=str)

    # arguments for computing resources (n_gpu_workers inferred from visible gpus)
    parser.add_argument('--tt_n_gpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_trials', default=1000, type=int)
    parser.add_argument('--tt_n_cpu_workers', default=5, type=int)
    #parser.add_argument('--mem_limit_gb', default=8.0, type=float)
    parser.add_argument('--gpus_viz', default='0;1', type=str)

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
    parser.add_argument('--ae_experiment_name', default='test_pt',type=str)
    parser.add_argument('--ae_version', default='best')

    parser.add_argument('--neural_ae_experiment_name', default='grid_search')
    parser.add_argument('--neural_ae_model_type', default='ff')
    parser.add_argument('--neural_ae_version', default='best')

    parser.add_argument('--neural_arhmm_experiment_name', default='grid_search')
    parser.add_argument('--neural_arhmm_model_type', default='ff')
    parser.add_argument('--neural_arhmm_version', default='best')


    parser.add_argument('--arhmm_experiment_name', default='diff_init_grid_search')
    parser.add_argument('--arhmm_version', default='best')

    parser.add_argument('--ae_model_type', default='conv')
    parser.add_argument('--n_ae_latents', default=12, type=int)
    parser.add_argument('--n_arhmm_states', default=32, type=int)
    parser.add_argument('--kappa', default=0, type=float)
    parser.add_argument('--noise_type', default='gaussian', type=str)
    parser.add_argument('--n_max_lags', default=8) 

    # add neural arguments (others are dataset-specific)
    if namespace.search_type=='grid_search':
        parser.add_argument('--experiment_name', '-en', default='grid_search', type=str)
        parser.opt_list('--x_scale', default=1, options=[.1,1,10,20], type=int, tunable=True) 
        parser.opt_list('--z_scale', default=1, options=[.1,1,10,20], type=int, tunable=True) 
        parser.add_argument('--n_samples',  default=1, type=int)
    elif namespace.search_type=='best':

        from fitting.utils import get_best_model_version
        namespace, extra = parser.parse_known_args()
        hparams_tmp = vars(namespace)
        hparams_tmp['experiment_name'] = 'grid_search'
        _, _, expt_dir = get_output_dirs(hparams_tmp)
        best_version = get_best_model_version(expt_dir)[0]
        best_file = os.path.join(expt_dir, best_version, 'meta_tags.pkl')
        print('Loading best discrete decoder from %s' % best_file)
        with open(best_file, 'rb') as f:
            hparams_best = pickle.load(f)
        parser.add_argument('--x_scale', default=hparams_best['x_scale']) 
        parser.add_argument('--z_scale', default=hparams_best['z_scale']) 
        parser.add_argument('--n_samples',  default=25, type=int)
        parser.add_argument('--best_version',  default=best_version, type=str)
        parser.add_argument('--experiment_name', '-en', default='best', type=str)


if __name__ == '__main__':

    hyperparams = get_params('grid_search')

    t = time.time()
    # if hyperparams.device == 'cuda' or hyperparams.device == 'gpu':
    #     if hyperparams.device == 'gpu':
    #         hyperparams.device = 'cuda'
    #     gpu_ids = hyperparams.gpus_viz.split(';')
    #     hyperparams.optimize_parallel_gpu(
    #         main,
    #         gpu_ids=gpu_ids,
    #         nb_trials=hyperparams.tt_n_gpu_trials,
    #         nb_workers=len(gpu_ids))
    # elif hyperparams.device == 'cpu':
    #     hyperparams.optimize_parallel_cpu(
    #         main,
    #         nb_trials=hyperparams.tt_n_cpu_trials,
    #         nb_workers=hyperparams.tt_n_cpu_workers)
    for hyperparam_trial in hyperparams.trials(100):
        main(hyperparam_trial)
    print('Total fit time: {}'.format(time.time() - t))
    #if hyperparams.export_predictions_best:
    #    export_predictions_best(vars(hyperparams))
