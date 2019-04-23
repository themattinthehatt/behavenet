import os
import time

from data.data_generator import *
from data.transforms import Resize

# choose dataset(s)
ids = {
    'lab': 'musall',
    'expt': 'vistrained',
    # 'animal': ['mSM30', 'mSM34'],
    # 'session': []}
    'animal': 'mSM30',
    'session': '10-Oct-2017'}

if os.uname().nodename == 'white-noise':
    data_dir = '/home/mattw/data/'
else:
    data_dir = '/labs/abbott/behavenet/data/'

# build data generator
# data_generator = ConcatSessionsGenerator(
#     data_dir, ids, ['images'], [Resize(size=(128, 128), order=1)],
#     format='hdf5', device='cpu')
data_generator = ConcatSessionsGenerator(
    data_dir, ids, ['images'], [None],
    format='hdf5', device='cpu')

# iterate through all datasets
for epoch in range(1):
    print('Starting epoch {}'.format(epoch))
    t = time.time()
    for batch in range(data_generator.num_tot_batches['train']):
        # get next minibatch
        data, dataset = data_generator.next_batch('train')
        # data['images'], data['batch_indx']
        trial = data['batch_indx']

        print('Dataset: {}, Trial: {}'.format(dataset, trial))
    print('Elapsed time: {} sec'.format(time.time() - t))
