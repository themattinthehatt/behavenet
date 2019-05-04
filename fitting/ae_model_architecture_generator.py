import os
import argparse
import numpy as np
import pickle
from behavenet.models import AE
from fitting.utils import estimate_model_footprint
import copy

def get_possible_arch(input_dim,n_ae_latents):
    ## Here is where you can set options/probabilities etc

    # Possible options for the architecture
    opts={}
    opts['possible_kernel_sizes'] = np.asarray([3,5,7,9,11,15])
    opts['possible_strides'] = np.asarray([2]) # stride will be 1 if using max pooling layers
    opts['possible_max_pool_sizes'] = np.asarray([2]) ### MAX POOL SIZE > 2 NOT IMPLEMENTED YET - NEED TO FIGURE OUT HOW TO COMBINE PADDING/CEIL MODE
    opts['possible_n_channels'] = np.asarray([16,32,64,128,256,512])
    opts['prob_stopping'] = np.arange(0,1,.05)
    #opts['FF_layer_prob'] = .2 # probability of having FF layer at end of decoding model (in past seemed to help weirdly)

    arch={}
    arch['ae_input_dim'] = input_dim
    arch['model_type'] = 'conv' 
    arch['ae_n_ae_latents'] = n_ae_latents
    arch['ae_decoding_last_FF_layer'] = 0 #np.random.choice(np.asarray([0,1]),p=np.asarray([1-FF_layer_prob, FF_layer_prob]))
    arch['ae_batch_norm'] = 0 

    # First decide if strides only or max pooling
    network_types = ['strides_only'] # ['strides_only','max_pooling']
    arch['ae_network_type'] = 'strides_only' #network_types[np.random.randint()]
    
    # Then decide if padding is 0 (0) or same (1) for all layers
    padding_types = ['valid','same']
    arch['ae_padding_type'] = padding_types[np.random.randint(2)]

    arch = get_encoding_conv_block(arch, opts)
    arch = get_decoding_conv_block(arch)

    return arch

def calculate_output_dim(input_dim,kernel,stride,padding_type, layer_type):
    # inspired by: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
    # https://github.com/pytorch/pytorch/issues/3867
    
    if layer_type == 'conv':

        if padding_type=='same':
            output_dim = (input_dim+stride-1)//stride
            total_padding_needed = max(0,(output_dim-1)*stride+kernel-input_dim)
            before_pad = total_padding_needed//2
            after_pad = total_padding_needed-before_pad
        elif padding_type=='valid':
            output_dim = int(np.floor((input_dim-kernel)/stride+1))
            before_pad = 0 
            after_pad = 0

    elif layer_type == 'maxpool':
       
        if kernel!=2:
            raise NotImplementedError

        if padding_type=='same':
            output_dim = int(np.ceil((input_dim-kernel)/stride+1))
            before_pad=0
            after_pad=0
        elif padding_type=='valid':
            output_dim = int(np.floor((input_dim-kernel)/stride+1))
            before_pad = 0
            after_pad = 0   

    return output_dim, before_pad, after_pad


def get_encoding_conv_block(arch,opts):
    # input dims should be n channels by y pix by x pix
    
    last_dims = arch['ae_input_dim'][0]*arch['ae_input_dim'][1]*arch['ae_input_dim'][2]
    smallest_pix = min(arch['ae_input_dim'][1],arch['ae_input_dim'][2])
    
    arch['ae_encoding_x_dim'] = []
    arch['ae_encoding_y_dim'] = []
    
    arch['ae_encoding_n_channels'] = []
    arch['ae_encoding_kernel_size'] = []
    arch['ae_encoding_stride_size'] = []
    arch['ae_encoding_x_padding'] = []
    arch['ae_encoding_y_padding'] = []
    arch['ae_encoding_layer_type'] = []
        
    i_layer=0
    global_layer=0
    while last_dims > arch['ae_n_ae_latents'] and smallest_pix>=1: 

        # Get conv2d layer
        kernel_size = np.random.choice(opts['possible_kernel_sizes'])
        stride_size = np.random.choice(opts['possible_strides']) if arch['ae_network_type'] == 'strides_only' else 1

        if i_layer == 0: # use input dimensions
            input_dim_y, input_dim_x = arch['ae_input_dim'][1], arch['ae_input_dim'][2]
        else:
            input_dim_y, input_dim_x = arch['ae_encoding_y_dim'][i_layer-1], arch['ae_encoding_x_dim'][i_layer-1]

        output_dim_y, y_before_pad, y_after_pad  = calculate_output_dim(input_dim_y,kernel_size,stride_size,padding_type=arch['ae_padding_type'],layer_type='conv')
        output_dim_x, x_before_pad, x_after_pad= calculate_output_dim(input_dim_x,kernel_size,stride_size,padding_type=arch['ae_padding_type'],layer_type='conv')

        if i_layer == 0:
            remaining_channels = opts['possible_n_channels'][opts['possible_n_channels']>=arch['ae_input_dim'][0]]
        else:
            remaining_channels=opts['possible_n_channels'][opts['possible_n_channels']>=arch['ae_encoding_n_channels'][i_layer-1]]
        
        if len(remaining_channels)>1:
            prob_channels = [.75]+[.25/(len(remaining_channels)-1) for i in range(len(remaining_channels)-1)] 
        else:
            prob_channels = [1]


        n_channels = np.random.choice(remaining_channels,p=prob_channels)

        if np.prod(n_channels*output_dim_x*output_dim_y)> arch['ae_n_ae_latents'] and np.min([output_dim_x,output_dim_y])>=1:
            # Choices ahead of time
            arch['ae_encoding_n_channels'].append(n_channels)
            arch['ae_encoding_kernel_size'].append(kernel_size)
            arch['ae_encoding_stride_size'].append(stride_size)
            
            # Automatically calculated
            arch['ae_encoding_x_dim'].append(output_dim_x)
            arch['ae_encoding_y_dim'].append(output_dim_y)
            arch['ae_encoding_x_padding'].append((x_before_pad,x_after_pad))
            arch['ae_encoding_y_padding'].append((y_before_pad,y_after_pad))
            arch['ae_encoding_layer_type'].append('conv')
            i_layer+=1
        else:
            break


        # Get max pool layer if applicable      
        if arch['ae_network_type'] == 'max_pooling':
            kernel_size = np.random.choice(opts['possible_max_pool_sizes'])

            output_dim_y, y_before_pad, y_after_pad = calculate_output_dim(arch['ae_encoding_y_dim'][i_layer-1],kernel_size,kernel_size,padding_type=arch['ae_padding_type'],layer_type='maxpool')
            output_dim_x, x_before_pad, x_after_pad = calculate_output_dim(arch['ae_encoding_x_dim'][i_layer-1],kernel_size,kernel_size,padding_type=arch['ae_padding_type'],layer_type='maxpool')


            if np.prod(n_channels*output_dim_x*output_dim_y)> arch['ae_n_ae_latents'] and np.min([output_dim_x,output_dim_y])>=1:
                
                arch['ae_encoding_n_channels'].append(n_channels)
                arch['ae_encoding_kernel_size'].append(kernel_size)
                arch['ae_encoding_stride_size'].append(kernel_size) # for max pool layers have stride as kernel size
                arch['ae_encoding_x_padding'].append((x_before_pad,x_after_pad))
                arch['ae_encoding_y_padding'].append((y_before_pad,y_after_pad))
                arch['ae_encoding_x_dim'].append(output_dim_x)
                arch['ae_encoding_y_dim'].append(output_dim_y)
                arch['ae_encoding_layer_type'].append('maxpool')
                
                i_layer+=1
            else:
                # Delete previous conv layer
                arch['ae_encoding_n_channels'] = arch['ae_encoding_n_channels'][:-1]
                arch['ae_encoding_kernel_size'] = arch['ae_encoding_kernel_size'][:-1]
                arch['ae_encoding_stride_size'] = arch['ae_encoding_stride_size'][:-1]
                arch['ae_encoding_x_padding'] = arch['ae_encoding_x_padding'][:-1]
                arch['ae_encoding_y_padding'] = arch['ae_encoding_y_padding'][:-1]
                arch['ae_encoding_x_dim'] = arch['ae_encoding_x_dim'][:-1]
                arch['ae_encoding_y_dim'] = arch['ae_encoding_y_dim'][:-1]
                arch['ae_encoding_layer_type'] = arch['ae_encoding_layer_type'][:-1]
                break
    
        last_dims = arch['ae_encoding_n_channels'][-1]*arch['ae_encoding_y_dim'][-1]*arch['ae_encoding_x_dim'][-1]
        smallest_pix= min(arch['ae_encoding_y_dim'][-1],arch['ae_encoding_x_dim'][-1])
        stop_this_layer = np.random.choice([0,1],p=[1-opts['prob_stopping'][global_layer],opts['prob_stopping'][global_layer]])

        if stop_this_layer:
            break
            
        global_layer+=1   
        
    return arch

def get_decoding_conv_block(arch):
    
    arch['ae_decoding_x_dim'] = []
    arch['ae_decoding_y_dim'] = []
    arch['ae_decoding_x_padding'] = []
    arch['ae_decoding_y_padding'] = []

    arch['ae_decoding_n_channels'] = []
    arch['ae_decoding_kernel_size'] = []
    arch['ae_decoding_stride_size'] = []
 
    arch['ae_decoding_layer_type'] = []
    
    arch['ae_decoding_starting_dim'] = [arch['ae_encoding_n_channels'][-1],arch['ae_encoding_y_dim'][-1],arch['ae_encoding_x_dim'][-1]]

    encoding_layer_num_vec = np.arange(len(arch['ae_encoding_n_channels'])-1,-1,-1)
    
    i_layer=0
    for which_encoding_layer in encoding_layer_num_vec:       
        
        if which_encoding_layer==0:
            arch['ae_decoding_n_channels'].append(arch['ae_input_dim'][0])
        else:
            arch['ae_decoding_n_channels'].append(arch['ae_encoding_n_channels'][which_encoding_layer-1])
        
        arch['ae_decoding_kernel_size'].append(arch['ae_encoding_kernel_size'][which_encoding_layer])
        arch['ae_decoding_stride_size'].append(arch['ae_encoding_stride_size'][which_encoding_layer])
        arch['ae_decoding_x_padding'].append(arch['ae_encoding_x_padding'][which_encoding_layer])
        arch['ae_decoding_y_padding'].append(arch['ae_encoding_y_padding'][which_encoding_layer])

        if i_layer==0:
            input_dim_y, input_dim_x = arch['ae_decoding_starting_dim'][1:3]
        else:
            input_dim_y, input_dim_x = arch['ae_decoding_y_dim'][i_layer-1], arch['ae_decoding_x_dim'][i_layer-1]                
        
        output_dim_y = arch['ae_encoding_y_dim'][which_encoding_layer-1] if which_encoding_layer>0 else arch['ae_input_dim'][1]    
        output_dim_x = arch['ae_encoding_x_dim'][which_encoding_layer-1] if which_encoding_layer>0 else arch['ae_input_dim'][2]
        arch['ae_decoding_y_dim'].append(output_dim_y)
        arch['ae_decoding_x_dim'].append(output_dim_x)


        if arch['ae_encoding_layer_type'][which_encoding_layer]=='maxpool':
            arch['ae_decoding_layer_type'].append('unpool')
        elif arch['ae_encoding_layer_type'][which_encoding_layer]=='conv': # if conv layer
            arch['ae_decoding_layer_type'].append('convtranspose')
                                    
        i_layer+=1

    if arch['ae_decoding_last_FF_layer']: # if ff layer at end, use 16 channels for final conv layer
        arch['ae_decoding_n_channels'][-1] = 16              
    return arch


def draw_archs(
        batch_size, input_dim, n_ae_latents, n_archs=100, check_memory=True,
        mem_limit_gb=5.0):
    # input dim is [n_channels, y dim, x dim]

    all_archs=[]

    while len(all_archs)<n_archs:

        new_arch = get_possible_arch(input_dim, n_ae_latents)

        # Check max memory, keep if smaller than 10 GB, print if rejecting
        if check_memory:
            copied_arch = copy.deepcopy(new_arch)
            copied_arch['model_class'] = 'ae'
            model = AE(copied_arch)
            mem_size = estimate_model_footprint(
                model, tuple([batch_size] + input_dim))
            mem_size_gb = mem_size / 1e9
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
