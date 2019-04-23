import os
import time
import numpy as np
from test_tube import HyperOptArgumentParser, Experiment
from models import AE
import pickle

def main(hparams):

    hparams = vars(hparams)
    # Blend outer hparams with architecture hparams
    hparams = {**hparams,**hparams['architecture_params']}

    # delete 'architecture_params' key

    print(hparams)
    # # Start at random times (so test tube creates separate folders)
    # np.random.seed(random.randint(0,1000))
    # time.sleep(np.random.randint(hparams.max_start_time))

    # # Get all sessions present in directory
    # if hparams.session_list == 'all':
    #     ignored = ['preprocess_log.txt']
    #     hparams.session_list = [x for x in os.listdir(hparams.data_dir) if x not in ignored] 

    # #########################
    # ### Create Experiment ###
    # #########################

    # exp = Experiment(name=hparams.model_name,
    #         debug=False,
    #         save_dir=hparams.tt_save_path)
    # exp.add_argparse_meta(hparams)
    # exp.save()

    # ###########################
    # ### LOAD DATA GENERATOR ###
    # ###########################

    # data_generator = ConcatSessionsGenerator(hparams.data_dir, hparams.session_list,hparams.signals_list,transform=hparams.transforms_list,batch_size=hparams.batch_size,pad_amount=hparams.pad_amount,max_pad_amount=hparams.max_pad_amount,device=hparams.device)


    # ####################
    # ### CREATE MODEL ###
    # ####################

    model = AE(hparams)
    # model.to(hparams.device)

    # ####################
    # ### TRAIN MODEL ###
    # ####################

    # fit(hparams,model,data_generator)


def get_params(strategy):
    parser = HyperOptArgumentParser(strategy)

    # add data generator arguments

    # add training arguments

    
    # parser.add_argument('--arch_file_name', type=str)
    parser.add_argument('--file_name', '-f', help='file for where to save list of architectures (without extension)')
    parser.add_argument('--n_archs', '-n', help='number of architectures to randomly sample',type=int)
    parser.add_argument('--input_channels', '-i', help='list of n_channels', type=int)
    parser.add_argument('--x_pixels', '-x', help='number of pixels in x dimension', type=int)
    parser.add_argument('--y_pixels', '-y', help='number of pixels in y dimension', type=int)
    parser.add_argument('--n_latents', '-nl', help='number of latents',type=int)

    namespace, extra = parser.parse_known_args()

    parser.add_argument('--model_type',default='ae', type=str)

    # Load in file of architectures
    #list_of_archs = pickle.load(open(namespace.arch_file_name,'rb'))
    parser.add_argument('--arch_file_name', type=str)

    if file doesn't exist:
        print('Creating new list of architectures')
        list_archs = draw_archs()
        # save list of archtictures
    if it does exist:
        print('Using presaved list of architectures')
        load_file


    parser.opt_list('--architecture_params', options=list_of_archs,tunable=True)
    return parser.parse_args()

if __name__ == '__main__':
    hyperparams = get_params('grid_search')

    hyperparams.optimize_parallel_cpu(main,nb_trials=100)





