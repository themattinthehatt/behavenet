import torch
from torch.utils import data
import h5py
from collections import OrderedDict
from torchvision import transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    """Samples elements from a given list of indices (no randomness)

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

def get_train_val_test_batches(n_frames, max_pad_amount, batch_size, chunk_n_frames=[12600,2700,2700],gap_size=300):

    np.random.seed(123) # same random seed for reproducibility
    chunk_order = np.empty(0,)
    for i in range(int(np.ceil(n_frames/chunk_n_frames[0]))):
        chunk_order = np.append(chunk_order,np.random.permutation([0,1,2])).astype('int')
    
    batch_inds = [np.empty(0,).astype('int')]*3
    current_frame = max_pad_amount
    for i_chunk in chunk_order:

        max_frame = min(current_frame+chunk_n_frames[i_chunk],n_frames-max_pad_amount-batch_size)
        batch_inds[i_chunk] = np.append(batch_inds[i_chunk],np.arange(current_frame, max_frame, batch_size),axis=0).astype('int')
        if current_frame+chunk_n_frames[i_chunk]>n_frames:
            break
        current_frame += chunk_n_frames[i_chunk]
        current_frame += gap_size
    
    return batch_inds

class SingleSessionDataset(data.Dataset):
    
    def __init__(self, data_dir, session_name, signals_list, transform, batch_size, pad_amount,device):
        
        # Load data 
        self.batch_size = batch_size
        self.pad_amount = pad_amount
        self.session_name = session_name
        self.signals_list = signals_list
        self.transform = transform
        self.data_dir = data_dir
        self.h5_pointer = h5py.File(self.data_dir+session_name+'/concatenated_data.h5', 'r') 
        self.device = device
          
    def __len__(self):
         return len(self.h5_pointer[list(self.h5_pointer.keys())[0]])

    def __getitem__(self, idx):

        sample = OrderedDict()
        
        for i, signal in enumerate(self.signals_list):
            
            # Index correct section
            sample[signal] = self.h5_pointer[signal][idx-self.pad_amount:idx+self.batch_size+self.pad_amount]
        
            # Apply transforms
            if self.transform[i]:

                if signal == 'loglikes': # need depth info for mask
                    sample[signal] = self.transform[i](sample[signal],self.h5_pointer['depth'][idx-self.pad_amount:idx+self.batch_size+self.pad_amount])
                else:
                    sample[signal] = self.transform[i](sample[signal])
                

            # Transform into tensor
            sample[signal] = torch.from_numpy(sample[signal]).to(self.device).float()
            sample['batch_idx'] = idx
        return sample

class ConcatSessionsGenerator():
    
    def __init__(self,data_dir, session_list, signals_list, transform, batch_size, pad_amount, max_pad_amount, device):

        self.session_list = session_list

        # Gather all datasets
        self.datasets=[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            self.datasets[i] = SingleSessionDataset(data_dir, dataset,signals_list,transform=transform,batch_size=batch_size,pad_amount=pad_amount,device=device)
        
        # Cut to shortest length, give warning if cutting off batches
        min_data_len = np.min([self.datasets[i].__len__() for i, session in enumerate(session_list)])
        max_data_len = np.max([self.datasets[i].__len__() for i, session in enumerate(session_list)])
        if (max_data_len-min_data_len)>batch_size:
            print('WARNING: CUTTING OFF DATA')

        # Get train/val/test batch indices 
        self.batch_inds = [None]*len(session_list)
        self.n_batches = [None]*len(session_list)
        for i, dataset in enumerate(session_list):
            self.n_batches[i] = [None]*3
            self.batch_inds[i] = get_train_val_test_batches(min_data_len, max_pad_amount, batch_size) #get_train_val_test_batches(self.datasets[i].__len__(), max_pad_amount, batch_size)
            for i_type in range(3):
                self.n_batches[i][i_type] = len(self.batch_inds[i][i_type])

        self.n_max_train_batches = np.max([item[0] for item in self.n_batches])
        self.n_max_val_batches = np.max([item[1] for item in self.n_batches])
        self.n_max_test_batches = np.max([item[2] for item in self.n_batches])

        # Gather all train data loaders
        self.train_dataset_loaders =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            train_sampler = SubsetRandomSampler(self.batch_inds[i][0])
            self.train_dataset_loaders[i] = torch.utils.data.DataLoader(self.datasets[i], batch_size=1,sampler=train_sampler)
        
        # Create all train iterators
        self.train_dataset_iter =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            self.train_dataset_iter[i] = iter(self.train_dataset_loaders[i])
        
        # Gather all val data loaders
        self.val_dataset_loaders =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            val_sampler = SubsetSampler(self.batch_inds[i][1])
            self.val_dataset_loaders[i] = torch.utils.data.DataLoader(self.datasets[i], batch_size=1,sampler=val_sampler)
        
        # Create all val iterators
        self.val_dataset_iter =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            self.val_dataset_iter[i] = iter(self.val_dataset_loaders[i])
        
        # Gather all test data loaders
        self.test_dataset_loaders =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            test_sampler = SubsetSampler(self.batch_inds[i][2])
            self.test_dataset_loaders[i] = torch.utils.data.DataLoader(self.datasets[i], batch_size=1,sampler=test_sampler)
        
        # Create all test iterators
        self.test_dataset_iter =[None]*len(session_list)
        for i, dataset in enumerate(session_list):
            self.test_dataset_iter[i] = iter(self.test_dataset_loaders[i])
             
    def reset_iterators(self,which_type):
        
        if which_type == 'train' or which_type=='all':
            for i, dataset in enumerate(self.session_list):
                self.train_dataset_iter[i] = iter(self.train_dataset_loaders[i])
            
        if which_type == 'val' or which_type=='all':
            for i, dataset in enumerate(self.session_list):
                self.val_dataset_iter[i] = iter(self.val_dataset_loaders[i])
            
        if which_type == 'test' or which_type=='all':
            for i, dataset in enumerate(self.session_list):
                self.test_dataset_iter[i] = iter(self.test_dataset_loaders[i])
            
    def next_train_batch(self):
        
        concat_sample = OrderedDict()
        for i, dataset in enumerate(self.session_list):
            
            # Get this session data
            try:
                sample = next(self.train_dataset_iter[i])
            except StopIteration:
                self.train_dataset_iter[i] = iter(self.train_dataset_loaders[i])
                sample = next(self.train_dataset_iter[i])
            
            # Concat across sessions
            for k, v in sample.items(): 
                if k in concat_sample:
                    concat_sample[k] = torch.cat((concat_sample[k],v),dim=0)
                else:
                    concat_sample[k] = v
    
        return concat_sample

    def next_val_batch(self):
        
        concat_sample = OrderedDict()
        for i, dataset in enumerate(self.session_list):
            
            # Get this session data
            try:
                sample = next(self.val_dataset_iter[i])
            except StopIteration:
                self.val_dataset_iter[i] = iter(self.val_dataset_loaders[i])
                sample = next(self.val_dataset_iter[i])
            
            # Concat across sessions
            for k, v in sample.items(): 
                if k in concat_sample:
                    concat_sample[k] = torch.cat((concat_sample[k],v),dim=0)
                else:
                    concat_sample[k] = v
    
        return concat_sample
    
    def next_test_batch(self):
        
        concat_sample = OrderedDict()
        for i, dataset in enumerate(self.session_list):
            
            # Get this session data
            try:
                sample = next(self.test_dataset_iter[i])
            except StopIteration:
                self.test_dataset_iter[i] = iter(self.test_dataset_loaders[i])
                sample = next(self.test_dataset_iter[i])
            
            # Concat across sessions
            for k, v in sample.items(): 
                if k in concat_sample:
                    concat_sample[k] = torch.cat((concat_sample[k],v),dim=0)
                else:
                    concat_sample[k] = v
    
        return concat_sample