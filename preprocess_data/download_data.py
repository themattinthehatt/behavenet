import os
import subprocess
import argparse
import h5py
import numpy as np
import shutil

# To set up gsutil:
# Don't need first two lines if you're already using python 2.7 (gsutil requires it)
# conda create -n env2.7 python=2.7
# source activate env2.7
# pip install gsutil
# pip install h5py
# pip install argparse
# pip install numpy
# gsutil config
# follow instructions

def download_data(data_dir, data_type):
    '''
    data_dir: head directory in which to download data
    data_type: 'clean', 'cabled'
    '''

    #######################
    ### DOWNLOAD CHUNKS ###
    #######################

    ## Check last item of data dir is / otherwise, add it
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'

    ## Check if directory exists, otherwise create it
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        print('WARNING: DIRECTORY EXISTS AND CODE ASSUMES NO EXTRA FILES ARE SAVED')
      #  sys.exit(0)
      
    ## Get directory from which to download data
    if data_type == 'clean':
        download_data_dir = 'gs://datta-data-shared/synthetic-mouse/chunked-data-2/'
    elif data_type == 'cabled':
        download_data_dir = 'gs://datta-data-shared/training-final/'


    if data_type == 'clean':
        download_data_dir = download_data_dir+'*'
    elif data_type == 'cabled':
        download_data_dir = download_data_dir+'*c-filt*' # there are two types of data, we only want most recent


    ## Download all data from google console
    return_code = subprocess.call('gsutil cp -r  ' +download_data_dir +' '+ data_dir + '', shell=True)


    ##########################
    ### CONCATENATE CHUNKS ###
    ##########################
    # we want one large dataset for each session

    dir_names = os.listdir(data_dir) # each directory is a session

    for dir_name in dir_names:

        files = os.listdir(data_dir+dir_name)
        files.sort(key=lambda x: int(x.split('.h5')[0][-3:]))

        dataset={}

        n_frames=0
        for i_file in range(len(files)):

            file_name = data_dir+dir_name+'/'+files[i_file]
            print(file_name)
            if i_file == 0: # first file
                
                h5_file = h5py.File(file_name,'r')
                signals = list(h5_file.keys())

                for signal in signals:

                    data = h5_file[signal][:]
                    if data.shape[0] == 1: # some of the signals (ex angle) were originally 1xT
                        data = np.transpose(data)

                    dataset[signal] = data
                h5_file.close()

                # Save this chunk separately (small data for local use)
                shutil.copyfile(file_name, data_dir+dir_name+'/first_chunk.h5')

                n_frames += data.shape[0]
            else:
                
                h5_file = h5py.File(file_name,'r')
 
                for signal in signals:

                    data = h5_file[signal][:]
                    if data.shape[0] == 1: # some of the signals (ex angle) were originally 1xT
                        data = np.transpose(data)

                    dataset[signal] = np.concatenate((dataset[signal],data),axis=0)
                h5_file.close()
                n_frames += data.shape[0]

        ## Save new file/sanity check that each signal has same number of frames
        large_file_filename = data_dir+dir_name+'/concatenated_data.h5'
        small_file_filename = data_dir+dir_name+'/partial_concatenated_data.h5'
        h5py_dataset = h5py.File(large_file_filename, 'w')
        h5py_dataset_small = h5py.File(small_file_filename, 'w')
        for signal in signals:

            if dataset[signal].shape[0] !=n_frames:
                print('ERROR: # OF FRAMES OFF')
                sys.exit(0)

            h5py_dataset.create_dataset(signal,
                                  data=dataset[signal])

            if signal !='depth':
                h5py_dataset_small.create_dataset(signal,
                      data=dataset[signal])

        h5py_dataset.close()
        h5py_dataset_small.close()


    ############################
    ### DELETE ORIGINAL DATA ###
    ############################

    for dir_name in dir_names:

        no_remove = set()
        no_remove.add('concatenated_data.h5')
        no_remove.add('partial_concatenated_data.h5')
        no_remove.add('first_chunk.h5')

        for f in os.listdir(data_dir+dir_name):
            if f not in no_remove:
                os.remove(data_dir+dir_name+'/'+f)

    #######################
    ### CREATE LOG FILE ###
    #######################
    log = ['downloaded from '+download_data_dir+'','concatenated into large dataset']
    log_path = os.path.join(data_dir+'preprocess_log.txt')
    log_file = open(log_path, 'a')
    log_file.write('\n'.join(log))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir','-d')
    parser.add_argument('--data_type','-t',choices=['clean','cabled'])
    params = parser.parse_args()
    download_data(params.data_dir,params.data_type)

    