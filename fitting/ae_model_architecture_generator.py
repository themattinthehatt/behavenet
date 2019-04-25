import os
import argparse
import numpy as np
import pickle
from behavenet.models import AE
from behavenet.utils import estimate_model_footprint
import copy


def calculate_conv2d_maxpool2d_output_dim(input_dim,kernel,stride,padding):
    output_dim = (input_dim+2*padding-kernel)/stride + 1
    return int(np.floor(output_dim))


def calculate_convtranspose2d_output_dim(input_dim,kernel,stride,padding,target_output_dim):
    output_dim = (input_dim-1)*stride-2*padding+kernel
    output_padding = target_output_dim-output_dim
    return int(output_dim+output_padding), output_padding


def get_encoding_conv_block(input_dim, n_latents):
    # input dims should be n channels by x pix by y pix
    
    # Set possible params
    possible_kernel_sizes = np.asarray([3,5,7,9,11,15])
    possible_strides = np.asarray([2]) # stride will be 1 if using max pooling layers
    possible_max_pool_sizes = np.asarray([2])
    possible_n_channels = np.asarray([16,32,64,128,256,512])
    prob_stopping = np.arange(0,1,.05)
    
    
    encoding_block={}
    encoding_block['input_dim'] = input_dim
    encoding_block['ae_conv_vs_linear'] = 'conv' 
    encoding_block['n_latents'] = n_latents
    
    # First decide if strides only (0) or max pooling (1)
    network_types = ['strides_only','max_pooling']
    which_network_type = network_types[np.random.randint(2)]
    encoding_block['ae_encoding_network_type'] = which_network_type
    
    # Then decide if padding is 0 (0) or same (1) for all layers
    padding_types = ['zero','same']
    which_padding_type = padding_types[np.random.randint(2)]
    encoding_block['ae_encoding_padding_type'] = which_padding_type
    
    last_dims = input_dim[0]*input_dim[1]*input_dim[2]
    smallest_pix = min(input_dim[1],input_dim[2])
    
    encoding_block['ae_encoding_x_dim'] = []
    encoding_block['ae_encoding_y_dim'] = []
    
    encoding_block['ae_encoding_n_channels'] = []
    encoding_block['ae_encoding_kernel_size'] = []
    encoding_block['ae_encoding_stride_size'] = []
    encoding_block['ae_encoding_padding_size'] = []
    encoding_block['ae_encoding_layer_type'] = []
        
    i_layer=0
    global_layer=0
    while last_dims > encoding_block['n_latents'] and smallest_pix>1: 

        # Get conv2d layer
        kernel_size = np.random.choice(possible_kernel_sizes)
        stride_size = np.random.choice(possible_strides) if which_network_type == 'strides_only' else 1
        padding_size = 0 if which_padding_type == 'zero' else (kernel_size-1)/2

        if i_layer == 0: # use input dimensions
            input_dim_x, input_dim_y = input_dim[1], input_dim[2]
        else:
            input_dim_x, input_dim_y = encoding_block['ae_encoding_x_dim'][i_layer-1], encoding_block['ae_encoding_y_dim'][i_layer-1]

        output_dim_x = calculate_conv2d_maxpool2d_output_dim(input_dim_x,kernel_size,stride_size,padding_size)
        output_dim_y = calculate_conv2d_maxpool2d_output_dim(input_dim_y,kernel_size,stride_size,padding_size)

        if i_layer == 0:
            remaining_channels = possible_n_channels[possible_n_channels>=input_dim[0]]
        else:
            remaining_channels=possible_n_channels[possible_n_channels>=encoding_block['ae_encoding_n_channels'][i_layer-1]]
        
        if len(remaining_channels)>1:
            prob_channels = [.75]+[.25/(len(remaining_channels)-1) for i in range(len(remaining_channels)-1)] 
        else:
            prob_channels = [1]


        n_channels = np.random.choice(remaining_channels,p=prob_channels)


        if np.prod(n_channels*output_dim_x*output_dim_y)> encoding_block['n_latents'] and np.min([output_dim_x,output_dim_y])>=1:
            # Choices ahead of time
            encoding_block['ae_encoding_n_channels'].append(n_channels)
            encoding_block['ae_encoding_kernel_size'].append(kernel_size)
            encoding_block['ae_encoding_stride_size'].append(stride_size)
            
            # Automatically calculated
            encoding_block['ae_encoding_x_dim'].append(output_dim_x)
            encoding_block['ae_encoding_y_dim'].append(output_dim_y)
            encoding_block['ae_encoding_padding_size'].append(int(padding_size))
            
            encoding_block['ae_encoding_layer_type'].append('conv')
            i_layer+=1
        else:
            break


        # Get max pool layer if applicable      
        if which_network_type == 'max_pooling':
            kernel_size = np.random.choice(possible_max_pool_sizes)

            if which_padding_type == 'zero':
                padding_size = 0
            elif which_padding_type == 'same':
                padding_size = (kernel_size-1)/2
            else:
                raise ValueError('Not implemented')   


            output_dim_x = calculate_conv2d_maxpool2d_output_dim(encoding_block['ae_encoding_x_dim'][i_layer-1],kernel_size,kernel_size,padding_size)
            output_dim_y = calculate_conv2d_maxpool2d_output_dim(encoding_block['ae_encoding_x_dim'][i_layer-1],kernel_size,kernel_size,padding_size)

            if np.prod(n_channels*output_dim_x*output_dim_y)> encoding_block['n_latents'] and np.min([output_dim_x,output_dim_y])>=1:
                
                encoding_block['ae_encoding_n_channels'].append(n_channels)
                encoding_block['ae_encoding_kernel_size'].append(kernel_size)
                encoding_block['ae_encoding_stride_size'].append(kernel_size) # for max pool layers have stride as kernel size
                encoding_block['ae_encoding_padding_size'].append(int(padding_size))
                
                encoding_block['ae_encoding_x_dim'].append(output_dim_x)
                encoding_block['ae_encoding_y_dim'].append(output_dim_y)
                encoding_block['ae_encoding_layer_type'].append('maxpool')
                
                i_layer+=1
            else:
                # Delete previous conv layer
                encoding_block['ae_encoding_n_channels'] = encoding_block['ae_encoding_n_channels'][:-1]
                encoding_block['ae_encoding_kernel_size'] = encoding_block['ae_encoding_kernel_size'][:-1]
                encoding_block['ae_encoding_stride_size'] = encoding_block['ae_encoding_stride_size'][:-1]
                encoding_block['ae_encoding_x_dim'] = encoding_block['ae_encoding_x_dim'][:-1]
                encoding_block['ae_encoding_y_dim'] = encoding_block['ae_encoding_y_dim'][:-1]
                encoding_block['ae_encoding_layer_type'] = encoding_block['ae_encoding_layer_type'][:-1]
                break
    
        stop_this_layer = np.random.choice([0,1],p=[1-prob_stopping[global_layer],prob_stopping[global_layer]])

        if stop_this_layer:
            break
            
        global_layer+=1   
        
    return encoding_block

def get_decoding_conv_block(input_dim, encoding_block):
    
    decoding_block={}
    decoding_block['ae_decoding_x_dim'] = []
    decoding_block['ae_decoding_y_dim'] = []
    decoding_block['ae_decoding_x_output_padding'] = []
    decoding_block['ae_decoding_y_output_padding'] = []
    
    decoding_block['ae_decoding_n_channels'] = []
    decoding_block['ae_decoding_kernel_size'] = []
    decoding_block['ae_decoding_stride_size'] = []
    decoding_block['ae_decoding_padding_size'] = []
    decoding_block['ae_decoding_layer_type'] = []
    
    starting_dim = [encoding_block['ae_encoding_n_channels'][-1],encoding_block['ae_encoding_x_dim'][-1],encoding_block['ae_encoding_y_dim'][-1]]

    encoding_layer_num_vec = np.arange(len(encoding_block['ae_encoding_n_channels'])-1,-1,-1)
    
    i_layer=0
    for which_encoding_layer in encoding_layer_num_vec:       
        
        if which_encoding_layer==0:
            decoding_block['ae_decoding_n_channels'].append(input_dim[0])
        else:
            decoding_block['ae_decoding_n_channels'].append(encoding_block['ae_encoding_n_channels'][which_encoding_layer-1])
        
        decoding_block['ae_decoding_kernel_size'].append(encoding_block['ae_encoding_kernel_size'][which_encoding_layer])
        decoding_block['ae_decoding_stride_size'].append(encoding_block['ae_encoding_stride_size'][which_encoding_layer])
        decoding_block['ae_decoding_padding_size'].append(encoding_block['ae_encoding_padding_size'][which_encoding_layer])

        if i_layer==0:
            input_dim_x, input_dim_y = starting_dim[1:3]
        else:
            input_dim_x, input_dim_y = decoding_block['ae_decoding_x_dim'][i_layer-1], decoding_block['ae_decoding_y_dim'][i_layer-1]                
                       
        target_output_dim_x = encoding_block['ae_encoding_x_dim'][which_encoding_layer-1] if which_encoding_layer>0 else input_dim[1]
        target_output_dim_y = encoding_block['ae_encoding_y_dim'][which_encoding_layer-1] if which_encoding_layer>0 else input_dim[2]

        if encoding_block['ae_encoding_layer_type'][which_encoding_layer]=='maxpool':
            
            decoding_block['ae_decoding_layer_type'].append('unpool')
            output_dim_x =  target_output_dim_x  
            output_dim_y =  target_output_dim_y   
            output_padding_x=0
            output_padding_y=0
                                       
        elif encoding_block['ae_encoding_layer_type'][which_encoding_layer]=='conv': # if conv layer
            
            decoding_block['ae_decoding_layer_type'].append('convtranspose')
           
            output_dim_x, output_padding_x = calculate_convtranspose2d_output_dim(input_dim_x,encoding_block['ae_encoding_kernel_size'][which_encoding_layer],encoding_block['ae_encoding_stride_size'][which_encoding_layer],encoding_block['ae_encoding_padding_size'][which_encoding_layer],target_output_dim_x)
            output_dim_y, output_padding_y = calculate_convtranspose2d_output_dim(input_dim_y,encoding_block['ae_encoding_kernel_size'][which_encoding_layer],encoding_block['ae_encoding_stride_size'][which_encoding_layer],encoding_block['ae_encoding_padding_size'][which_encoding_layer],target_output_dim_y)

                                       
        decoding_block['ae_decoding_x_dim'].append(output_dim_x)
        decoding_block['ae_decoding_y_dim'].append(output_dim_y)
        decoding_block['ae_decoding_x_output_padding'].append(output_padding_x)
        decoding_block['ae_decoding_y_output_padding'].append(output_padding_y) 
        i_layer+=1
                               
    return decoding_block


def get_possible_arch(input_dim,n_latents):
    encoding_block = get_encoding_conv_block(input_dim, n_latents)
    decoding_block = get_decoding_conv_block(input_dim,encoding_block)
    arch_params = {**encoding_block,**decoding_block}
    return arch_params


def draw_archs(batch_size, input_dim,n_latents,n_archs=100,check_memory=True):
    all_archs=[]

    while len(all_archs)<n_archs:

        new_arch = get_possible_arch(input_dim, n_latents)

        # Check max memory, keep if smaller than 10 GB, print if rejecting
        if check_memory:
            mem_limit_gb = 5.0
            copied_arch = copy.deepcopy(new_arch)
            copied_arch['model_type'] = 'ae'
            model = AE(copied_arch)
            mem_size = estimate_model_footprint(
                model, tuple([batch_size] + input_dim))
            mem_size_gb = mem_size / 1000000000
            print(mem_size_gb)
            if mem_size_gb > mem_limit_gb:  # GB
                print(
                    'Model size of %02.3f GB is larger than limit of %1.3f GB;'
                    ' skipping model' % (mem_size_gb, mem_limit_gb))
                continue
            new_arch['mem_size_gb'] = mem_size_gb

        # Check against all previous arches
        matching=0
        for prev_arch in all_archs:
            if prev_arch == new_arch:
                matching=1
                break   

        if matching==0:
            all_archs.append(new_arch)

    return all_archs
