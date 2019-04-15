import h5py
import numpy as np
import os
import argparse
from glob import glob
import scipy.interpolate
from distutils.dir_util import copy_tree

"""
This script creates a new .h5 datastore with imputed and droped frames.
The processing does (per session):
1. Interpolates nans when there are 1-2 in a row
2. Drops the SAME frames across all signals to maintain the temporal order if those frames occur at start or end of session
Outputs a log file with all the changes done to the original data and a new file with the clean data
"""


def remove_nans(data_dir,dest_dir):

    log = []
    log.append('\n\n'+'-' * 100+'\n NAN REMOVAL \n '+ '_'*100+'\n\n')

    # If using new directory, copy everything
    if dest_dir is not None:
        print('copying original data from {} to {}...'.format(data_dir, dest_dir))
        copy_tree(data_dir,dest_dir)
    else:
        dest_dir = data_dir


    files = [y for x in os.walk(dest_dir) for y in glob(os.path.join(x[0], '*.h5'))]

    for filename in files:

        log.append('\nfilename: {}'.format(filename))

        original_data_chunk = h5py.File(filename, 'r')

        # impute single nan frames with the neighbors
        all_remaining_nan_idxs, edge_nan_idxs, dataset = impute_single_nans(log, original_data_chunk)

        
        # drop the same frames across all signals if they occur at start or end
        dataset, empty_data = drop_frame_across_signals(edge_nan_idxs, dataset, log)

        # check if any remaining nans
        all_remaining_nan_idxs = np.setdiff1d(all_remaining_nan_idxs,edge_nan_idxs)

        if all_remaining_nan_idxs.size:
            print('NaNs remaining after interpolation and edge removal!')
            print(filename)
            #ver
            # TO DO: throw error here (not for now as Win working on getting rid of more nans)

        # Save dataset
        original_data_chunk.close()

        tmp_filename = filename.split('.h5')[0] + '_tmp.h5'
        h5_tmp = h5py.File(tmp_filename, 'w')

        # Save temporary file
        for signal in list(dataset.keys()):
            h5_tmp.create_dataset(signal,
                                  data=dataset[signal]) 
        h5_tmp.close()
        
        # Delete non temp
        os.remove(filename)

        # Rename tmp to original
        os.rename(tmp_filename, filename)

        # Remove file completely if empty
        if empty_data:
             os.remove(filename)

    # write logs
    log_path = dest_dir+'/preprocess_log.txt'
    log_file = open(log_path, 'a')
    log_file.write("\n".join(log))

def drop_frame_across_signals(edge_nan_idxs, dataset, log):
    """
    Some frames at the start/end of session need to be droped across all signals because they'll shift the time component
    if they're dropped from that signal only
    :param block_nan_idxs_to_remove:
    :param dataset:
    :param log:
    :param h5py_group:
    :return:
    """

    signals = list(dataset.keys())

    for signal in signals:

        # load data
        data = dataset[signal]

        # use numpy mask to remove the nan indexes
        mask = np.ones(len(data), np.bool)
        mask[list(edge_nan_idxs)] = 0
        dataset[signal] = data[mask]

        if len(dataset[signal]) == 0:
            empty_data = True
        else:
            empty_data=False
    log.append('Removed {} frames'.format(edge_nan_idxs.shape[0]))
    return dataset, empty_data

def construct_nan_indices(a,max_consecutive_nans=25):
    '''
    Return true/false array where true indicates nans (but only when 1-2 are consecutive and not on the edges)
    a: array, should be Tx1
    '''
    
    # Get where all nans occur
    nan_indices = np.isnan(a)
    ranges=None
    if np.sum(nan_indices): # if any nans

        # Get ranges of consecutive nans
        isnan = np.concatenate(([0], np.isnan(a).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isnan))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        
        # Check if any ranges are over 2, make these elements false
        large_ranges = np.where(np.diff(ranges)>max_consecutive_nans)[0]
        for i_range in large_ranges:
            nan_indices[ranges[i_range][0]:ranges[i_range][1]]=False
        
        # Check if any nans are on the edges, make these elements false
        if ranges[0,0]==0 or ranges[0,0]==1:
            nan_indices[ranges[0][0]:ranges[0][1]]=False
        if ranges[-1,1]==a.shape[0]:
            nan_indices[ranges[-1][0]:ranges[-1][1]]=False
    
    return nan_indices, ranges

def impute_single_nans(log, data_chunk):
    """
    Sets a nan frame as the average of the 2 neighbors if it has non-nan left, right neighbor
    :param log:
    :param data_ptr:
    :param filename:
    :return:
    """
    all_remaining_nan_idxs = np.empty(0,)
    edge_nan_idxs = np.empty(0,)
    dataset = {}

    signals = list(data_chunk.keys())

    for signal in signals:

        # load data
        data = data_chunk[signal][:]

        if data.shape[0] == 1: # some of the signals (ex angle) were originally 1xT
            data = np.transpose(data)

        # Check if any time points are nans for some columns and not others (problem for code)
        nan_amounts = np.unique(np.sum(np.isnan(data.reshape((-1,1))),1))
        if np.any(~np.isin(nan_amounts,np.asarray([0,data.reshape((-1,1)).shape[1]]))):
            print('ERROR: time points have nans in some columns, values in others')
            sys.exit(0)

        # Replace nans with linear interpolation (only if 1-2 nans consecutively)

        if data.ndim == 1:
            data_cat = data
        else:
            data_cat = np.sum(data.reshape((data.shape[0],-1)),axis=1)

        nan_indices, ranges = construct_nan_indices(data_cat)

        xp = np.where(~nan_indices)[0]
        fp = data[~nan_indices]
        x  = np.where(nan_indices)[0]

        f = scipy.interpolate.interp1d(xp, fp,axis=0)

        data[nan_indices] = f(x)

        # Log information
        log.append('{}: replaced {} nans'.format(signal, np.sum(nan_indices)))

        # find all remaining rows with nans (edges/3 or more consective nans)
        if data.ndim == 1:
            data_cat = data
        else:
            data_cat = np.sum(data.reshape((data.shape[0],-1)),axis=1)
        x_nan_idxs = np.argwhere(np.isnan(data_cat)).flatten()

        # track so we can remove across full dataset
        all_remaining_nan_idxs = np.append(all_remaining_nan_idxs,x_nan_idxs)
        if ranges is not None:
            if ranges[0,0]==0 or ranges[0,0]==1:
                edge_nan_idxs = np.append(edge_nan_idxs,np.arange(ranges[0][0],ranges[0][1]))
            if ranges[-1,1]==data_cat.shape[0]:
                edge_nan_idxs = np.append(edge_nan_idxs,np.arange(ranges[-1][0],ranges[-1][1]))
        dataset[signal] = data
        # if ranges is not None:
        #     ranges = np.delete(ranges,np.where(np.diff(ranges)<=9)[0],axis=0)
        # print(data_cat.shape[0])
        # print(ranges)
        # if ranges is not None:
        #     print(np.max(np.diff(ranges)))
    all_remaining_nan_idxs = np.unique(all_remaining_nan_idxs)
    edge_nan_idxs = np.unique(edge_nan_idxs)
    return all_remaining_nan_idxs,  edge_nan_idxs.astype('int'), dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir','-d', help='data folder to process')
    parser.add_argument('--dest_dir', '-n', default=None, help='destination path, None if you just want to override current data')
    args = parser.parse_args()

    remove_nans(args.data_dir,args.dest_dir)


