
from models import ARHMM
#from test_tube import HyperOptArgumentParser, Experiment
import h5py
import numpy as np
import torch

# TEMP
import argparse
import copy
from training import fit_em

def data_generator(pca_file, uuids, batch_size,n_mice):
    
    total_batches=0
    n_batches = [None]*n_mice

    for i_mouse in range(n_mice):
        length_mouse = pca_file[uuids[i_mouse]].shape[0]
        n_batches[i_mouse] = np.floor(length_mouse/batch_size)

    total_batches = int(np.sum(n_batches))

    batch_inds = np.zeros((int(total_batches),2))
    i_pos=0
    
    for i_mouse in range(n_mice):
        for i_batch in range(int(n_batches[i_mouse])):
            batch_inds[i_pos,0] = i_mouse
            batch_inds[i_pos,1] = i_batch
            i_pos+=1

    loop_vec = np.arange(total_batches)
    
    
    for i_epoch in range(1000):
        np.random.shuffle(loop_vec)
        for ii in loop_vec:

            i_mouse = int(batch_inds[ii,0])
            which_batch = int(batch_inds[ii,1])

            yield pca_file[uuids[i_mouse]][which_batch*batch_size:(which_batch+1)*batch_size] #, i_mouse, which_batch #, behavioral_labels[i_mouse][which_bucket][which_batch*batch_size:(which_batch+1)*batch_size],depth[i_mouse][which_bucket][which_batch*batch_size:(which_batch+1)*batch_size],i_mouse,which_bucket,which_batch

def run(hparams):

    n_mice=17
    h5_temp = h5py.File('whitened_clean_pca.h5','r')        

    ################################
    ## Get rid of mouse with nans ##
    ################################

    uuids_all = list(h5_temp.keys())
    uuids = copy.deepcopy(uuids_all)
    delete_inds=[]
    for i in range(len(uuids_all)):
       # print(np.sum(np.isnan(h5_temp[uuids_all[i]][:])))
        if np.sum(np.isnan(h5_temp[uuids_all[i]][:]))>0:
            delete_inds.append(i)
    for ii in sorted(delete_inds,reverse=True):
        del uuids[ii]
        
    for i in range(n_mice):
       # print(uuids[i])
        if np.sum(np.isnan(h5_temp[uuids[i]][:])) > 0:
            print('ERROR: NANS STILL PRESENT')
            ver

    # Get number of training batches
    pca_file = h5_temp
    total_batches=0
    n_batches = [None]*n_mice
    for i_mouse in range(n_mice):
        length_mouse = pca_file[uuids[i_mouse]].shape[0]
        n_batches[i_mouse] = np.floor(length_mouse/hparams.batch_size)
    total_batches = int(np.sum(n_batches))
    nb_tng_batches = total_batches

    # Create data generator

    data_gen = data_generator(h5_temp,uuids,hparams.batch_size,n_mice)

    # Build and initialize model
    model = ARHMM(hparams)
    model.initialize('lr',data_gen,nb_tng_batches)

    # Fit model
    fit_em(model,data_gen,nb_tng_batches,num_epochs=3)


def get_params(strategy):
    parser = argparse.ArgumentParser()

    #parser = HyperOptArgumentParser(strategy=strategy)

    # Computing information
    parser.add_argument('--device', default='cpu', type=str)

    # Data information
    parser.add_argument('--batch_size', default=1000, type=int)

    # Training information
    parser.add_argument('--learning_rate', default=0.001, type=float)

    # Model hyperparameters
    parser.add_argument('--nlags', default=3, type=int) # number of lags to use for autoregressive component
    parser.add_argument('--n_discrete_states', default=100, type=int) # number of discrete states
    parser.add_argument('--latent_dim_size_h', default=10, type=int) # dim of continuous latent variables (for example, # of pca components)
    parser.add_argument('--transition_init', default=0.99, type=float) # used to initialize transition matrix, sets weighting of diagonal
    parser.add_argument('--alpha', default=200, type=int) # dirichlet prior hyperparam
    parser.add_argument('--kappa', default=1e8, type=int) # dirichlet prior hyperparam

    return parser.parse_args()
if __name__ == '__main__':
    hyperparams = get_params('grid_search')
    run(hyperparams)