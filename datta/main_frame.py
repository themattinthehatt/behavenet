import os
import time
import numpy as np
from test_tube import HyperOptArgumentParser, Experiment
from datta.data_generator.data_generator import ConcatSessionsGenerator
from datta.data_generator.transforms import *
from models import ARHMM, VAE, SLDS, AE
from variational import VanillaRecognitionNetwork
from training import fit
import random

def main(hparams):

    # Start at random times (so test tube creates separate folders)
    np.random.seed(random.randint(0,1000))
    time.sleep(np.random.randint(hparams.max_start_time))

    # Get all sessions present in directory
    if hparams.session_list == 'all':
        ignored = ['preprocess_log.txt']
        hparams.session_list = [x for x in os.listdir(hparams.data_dir) if x not in ignored] 

    #########################
    ### Create Experiment ###
    #########################

    exp = Experiment(name=hparams.model_name,
            debug=False,
            save_dir=hparams.tt_save_path)
    exp.add_argparse_meta(hparams)
    exp.save()

    ###########################
    ### LOAD DATA GENERATOR ###
    ###########################

    data_generator = ConcatSessionsGenerator(hparams.data_dir, hparams.session_list,hparams.signals_list,transform=hparams.transforms_list,batch_size=hparams.batch_size,pad_amount=hparams.pad_amount,max_pad_amount=hparams.max_pad_amount,device=hparams.device)


    ####################
    ### CREATE MODEL ###
    ####################

    if hparams.model_name == 'ARHMM':
        model = ARHMM(hparams)
        model.initialize('lr',data_generator)
        method='em'
    elif hparams.model_name == 'VAE':
        model = VAE(hparams)
        method='vae'
    elif hparams.model_name == 'AE':
        model = AE(hparams)
        method='ae'
    elif hparams.model_name == 'SLDS':
        variational_posterior = VanillaRecognitionNetwork(hparams)
        model = SLDS(hparams)
        model.initialize('lr',data_generator)
        model.VAE_encoder_model = None
        method='svi'

    model.batch_inds = data_generator.batch_inds # save train/val/test info

    model.to(hparams.device)

    ####################
    ### TRAIN MODEL ###
    ####################

    data_generator.reset_iterators('all') 
    if method == 'svi':
        fit(hparams,model,data_generator,exp,method=method,variational_posterior=variational_posterior)
    else:
        fit(hparams,model,data_generator,exp,method=method)
    
def get_params(strategy):
    parser = HyperOptArgumentParser(strategy=strategy)

    parser.add_opt_argument_list('--model_name', default='ARHMM', options=['VAE','ARHMM','SLDS','AE'],type=str,tunable=False)
    namespace, extra = parser.parse_known_args()
    model_name = namespace.model_name

    # Test tube
    parser.add_argument('--tt_save_path', default='test_tube', type=str)

    # Computing information
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--gpus_viz', default='1', type=str)
    parser.add_argument('--max_start_time', default=200, type=int) # seconds, code will run at some random integer between 0 and max_start_time seconds (for unlocking/test_tube silliness)

    # Data generation information
    parser.add_argument('--data_dir', default='/media/gssdb/DattaData_clean/',type=str)
    parser.add_argument('--session_list', default='all',nargs='+')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--pad_amount', default=0, type=int)
    parser.add_argument('--max_pad_amount', default=60, type=int)
    parser.add_argument('--pixel_size', default=80, type=int)

    # Training information
    parser.add_opt_argument_list('--learning_rate', default=0.0001, options=[0.0001,0.001,0.01],type=float,tunable=False)
    parser.add_argument('--val_check_interval', default=1, type=float)
    parser.add_argument('--max_nb_epochs', default=300, type=int)
    parser.add_argument('--min_nb_epochs', default=1, type=int)
    parser.add_argument('--disable_early_stop', dest='enable_early_stop', action='store_false')
    parser.add_argument('--early_stop_patience', default=6, type=int, help='number of val checks until stop')
    parser.add_argument('--early_stop_fraction', default=0.999, type=int, help='number of val checks until stop')

    # Model hyperparameters
    if model_name == 'ARHMM':
        parser.add_argument('--nlags', default=3, type=int) # number of lags to use for autoregressive component
        parser.add_argument('--n_discrete_states', default=100, type=int) # number of discrete states
        parser.add_argument('--latent_dim_size_h', default=10, type=int) # dim of continuous latent variables (for example, # of pca components)
        parser.add_argument('--transition_init', default=0.99, type=float) # used to initialize transition matrix, sets weighting of diagonal
        parser.add_argument('--alpha', default=200, type=int) # dirichlet prior hyperparam
        parser.add_opt_argument_list('--kappa', default=1e8, options=[1e6,1e8],type=int,tunable=False)
        #parser.add_argument('--kappa', default=1e8, type=int) # dirichlet prior hyperparam
        parser.add_argument('--low_d_type', default='vae', type=str) 
        parser.add_opt_argument_list('--whiten_vae', default=0, options=[0,1],type=int,tunable=False)
        parser.add_argument('--vae_model_path', default=None) 
        parser.add_argument('--signals_list', nargs='+', default=['depth'])
        parser.add_argument('--transforms_list',default=[ClipNormalize()])
    elif model_name == 'VAE':
        parser.add_opt_argument_list('--vae_type', default='linear', options=['linear','conv'],type=str,tunable=False)
        parser.add_opt_argument_list('--latent_dim_size_h', default=10,options=[4,7,10], type=int,tunable=False)
        parser.add_opt_argument_list('--y_var_value',default=1e-4,options=[1e-4,1e-3],type=float,tunable=False)
        parser.add_argument('--signals_list', nargs='+', default=['depth'])
        parser.add_argument('--transforms_list',default=[ClipNormalize()])
        parser.add_argument('--use_KL', default=1, type=int)
        parser.add_argument('--bn', default=1, type=int) # 1 to use batch norm, 0 not to
        parser.add_opt_argument_list('--y_var_parameter', default=1,options=[0,1], type=int,tunable=False) # 0 if you want one y_var (equal to y_var_value) which is unchangeable, 1 if you want y_var to be 80x80 parameter
    elif model_name == 'AE':
        parser.add_opt_argument_list('--ae_type', default='linear', options=['linear','conv'],type=str,tunable=False)
        parser.add_opt_argument_list('--loss_type', default='mse', options=['mse','nll'],type=str,tunable=True)
        parser.add_argument('--latent_dim_size_h', default=10, type=int)
        parser.add_argument('--signals_list', nargs='+', default=['depth'])
        parser.add_argument('--transforms_list',default=[ClipNormalize()])
        parser.add_argument('--bn', default=1, type=int) # 1 to use batch norm, 0 not to
        parser.add_opt_argument_list('--y_var_parameter', default=0,options=[0,1], type=int,tunable=True) # 0 if you want one y_var (equal to y_var_value) which is unchangeable, 1 if you want y_var to be 80x80 parameter
        parser.add_opt_argument_list('--y_var_value',default=1e-4,options=[1e-4,1e-3],type=float,tunable=False)
    elif model_name == 'SLDS':
        parser.add_argument('--nlags', default=3, type=int) # number of lags to use for autoregressive component
        parser.add_argument('--n_discrete_states', default=100, type=int) # number of discrete states
        parser.add_argument('--latent_dim_size_h', default=10, type=int) # dim of continuous latent variables (for example, # of pca components)
        parser.add_argument('--transition_init', default=0.99, type=float) # used to initialize transition matrix, sets weighting of diagonal
        parser.add_argument('--alpha', default=200, type=int) # dirichlet prior hyperparam
        parser.add_argument('--kappa', default=1e8, type=int) # dirichlet prior hyperparam
        #parser.add_argument('--vae_type', default='linear',type=str) 
        parser.add_argument('--init_vae_model_path', default=None) 
        parser.add_argument('--signals_list', nargs='+', default=['depth'])
        parser.add_argument('--transforms_list',default=[ClipNormalize()])
        #parser.add_argument('--y_var_value',default=1e-4,type=float)
        parser.add_argument('--low_d_type', default='vae', type=str) 

    return parser.parse_args()
    
if __name__ == '__main__':
    hyperparams = get_params('grid_search')
#    main(hyperparams)

    if hyperparams.device=='cuda':
        hyperparams.optimize_parallel_gpu_cuda(
            main,
            gpu_ids = hyperparams.gpus_viz.split(';'),
            nb_trials=100,
            nb_workers=100
        )
    if hyperparams.device=='cpu':
        main(hyperparams)
    
