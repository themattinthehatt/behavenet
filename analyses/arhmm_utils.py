import pickle
import os
import h5py
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import time
from behavenet.models import AE as AE
import pandas as pd

def make_overview_arhmm_figures(hparams):

    if type(hparams) is not dict:
        hparams = vars(hparams)

    filepath = os.path.join(
            hparams['tt_save_path'], hparams['lab'], hparams['expt'],
            hparams['animal'], hparams['session'], 'arhmm', str(hparams['n_ae_latents']).zfill(2)+'_latents')

    # Search over K/kappa/noise distribution
    # TO DO: make this less hacky (use best feature?)
    results={}
    K_vec = []
    kappa_vec = []
    K_dirs = [filename for filename in os.listdir(filepath) if os.path.isdir(os.path.join(filepath,filename))]
    for K_dir in K_dirs:
        kappa_dirs = [filename for filename in os.listdir(os.path.join(filepath, K_dir)) if os.path.isdir(os.path.join(os.path.join(filepath, K_dir),filename))]
        for kappa_dir in kappa_dirs:
            noise_dirs = [filename for filename in os.listdir(os.path.join(filepath, K_dir,kappa_dir)) if os.path.isdir(os.path.join(os.path.join(filepath, K_dir,kappa_dir),filename))]
            for noise_dir in noise_dirs:
                ver_dirs = [filename for filename in os.listdir(os.path.join(filepath, K_dir,kappa_dir,noise_dir,'test_tube_data',hparams['experiment_name'])) if os.path.isdir(os.path.join(os.path.join(filepath, K_dir,kappa_dir, noise_dir, 'test_tube_data',hparams['experiment_name']),filename))]
                for ver_dir in ver_dirs:
                    try:
                      filename = os.path.join(filepath, K_dir, kappa_dir, noise_dir,'test_tube_data',hparams['experiment_name'], ver_dir)
                      arch_file = pickle.load(open(os.path.join(filename,'meta_tags.pkl'),'rb'))
                      metrics_file = pd.read_csv(os.path.join(filename,'metrics.csv'))
                      if arch_file['training_completed']:
                          val_ll = metrics_file['val_ll'][0]
                          median_dur = metrics_file['median_dur'][1]
                          results[arch_file['n_arhmm_states'],arch_file['kappa'],arch_file['noise_type']] = dict(val_ll=val_ll,median_dur=median_dur)
                          K_vec.append(arch_file['n_arhmm_states'])
                          kappa_vec.append(arch_file['kappa'])
                    except:
                          pass
    K_vec = np.unique(np.asarray(K_vec))
    kappa_vec = np.unique(np.asarray(kappa_vec))

    filepath = os.path.join(
            hparams['tt_save_path'], hparams['lab'], hparams['expt'],
            hparams['animal'], hparams['session'], 'arhmm')

    plt.figure(figsize=(4, 4))
    for K in K_vec:
        plt.plot([results[(K, kappa, 'gaussian')]['median_dur'] for kappa in kappa_vec], '-o', label='K='+str(K)+', gaussian')
        plt.plot([results[(K, kappa, 'studentst')]['median_dur'] for kappa in kappa_vec], '--o', label='K='+str(K)+', studentst')
    plt.xticks(np.arange(len(kappa_vec)),[format(k,'.0e') for k in kappa_vec])
    plt.xlabel("Kappa ")
    plt.ylabel("Median State Duration (ms) ")
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(filepath+'/'+hparams['lab_example']+'_validation_median_durations.png',bbox_inches='tight',dpi=200)

    ## Generate number of states vs val likelihoos for all kappas

    plt.figure(figsize=(4, 4))
    for kappa in kappa_vec:
        plt.plot(K_vec, [results[K, kappa, 'gaussian']['val_ll'] for K in K_vec], '-o', label='kappa ='+str('{:.2e}'.format(kappa))+', gaussian')
        plt.plot(K_vec, [results[K, kappa, 'studentst']['val_ll'] for K in K_vec], '--o', label='kappa ='+str('{:.2e}'.format(kappa))+', studentst')

    plt.xlabel("Number of States (K)")
    plt.ylabel("Validation LL")
    plt.title('Val Likelihood')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(filepath+'/'+hparams['lab_example']+'_validation_loglikelihood.png',bbox_inches='tight',dpi=200)


def make_ind_arhmm_figures(hparams, exp, hmm, latents, trial_idxs, data_generator):

    if hparams['neural_bin_size']:
        frame_rate = 1000/hparams['neural_bin_size']
    elif not hparams['neural_bin_size']:
        frame_rate=30 # TO DO: fix frame rates

    states={}
    for data_type in ['train','val','test']:
        states[data_type] = [hmm.most_likely_states(x) for x in latents[data_type]]

    # relabeled_states={}
    # relabeled_states['train'], mapping, frame_counts = relabel_states_by_use(states['train'])
    # relabeled_states['val'], _, _ = relabel_states_by_use(states['val'], mapping)
    # relabeled_states['test'], _, _ = relabel_states_by_use(states['test'], mapping)

    filepath = os.path.join(hparams['results_dir'], 'test_tube_data', hparams['experiment_name'],
        'version_%i' % exp.version)

    # Compute state distributions on training data
    train_durations_frames = get_state_durations(latents['train'], hmm)
    train_durations_ms = train_durations_frames/frame_rate*1000

    exp.log({'median_dur': np.median(train_durations_ms)})
    exp.save()

    plt.clf()
    plt.hist(train_durations_ms,100,color='k');
    plt.xlabel('Time (ms)')
    plt.ylabel('Occurrences')
    plt.title('Training Data State Durations \n Kappa = '+str(format(hparams['kappa'],'.0e'))+', # States = ' +str(hparams['n_arhmm_states'])+' \n Noise = '+hparams['noise_type']+', N lags = '+str(hparams['n_lags']))
    plt.savefig(os.path.join(filepath,'duration_hist_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(format(hparams['kappa'],'.0e'))+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.png'),bbox_inches='tight')


    # ## Make figure of frame counts
    _, _, frame_counts = relabel_states_by_use(states['train'])
    plt.clf()
    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(0,len(frame_counts)),100*frame_counts/np.sum(frame_counts),color='k')
    xlab = plt.xlabel('Discrete State')
    plt.ylabel('Percentage of frames')
    plt.title('Training Data ARHMM State Times')
    plt.savefig(os.path.join(filepath,'proportion_times_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(format(hparams['kappa'],'.0e'))+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.png'),bbox_inches='tight')

    ## Make syllable movies
    make_syllable_movies(filepath=filepath, hparams=hparams, latents=latents['val'], states=states['val'], trial_idxs=trial_idxs['val'], data_generator=data_generator)

    ## Make real vs generated movies
    make_real_vs_generated_movies(filepath=filepath, hparams=hparams, hmm = hmm, latents=latents['val'], states=states['val'], data_generator=data_generator)


def get_discrete_chunks(states, include_edges=True):
    '''
    Find occurences of each discrete state

    input:
        states: list of trials, each trial is numpy array containing discrete state for each frame
        include_edges: include states at start and end of chunk

    output:
        indexing_list: list of length discrete states, each list contains all occurences of that discrete state by [chunk number, starting index, ending index]

    '''
    max_state = max([max(x) for x in states])
    indexing_list = [[] for x in range(max_state+1)]

    for i_chunk, chunk in enumerate(states):

        chunk = np.pad(chunk,(1,1),mode='constant',constant_values=-1) # pad either side so we get start and end chunks
        split_indices = np.where(np.ediff1d(chunk)!=0)[0] # Don't add 1 because of start padding, this is now indice in original unpadded data
        split_indices[-1]-=1 # Last index will be 1 higher that it should be due to padding

        for i in range(len(split_indices)-1):

            which_state = chunk[split_indices[i]+1] # get which state this chunk was (+1 because data is still padded)

            if not include_edges: # if not including the edges
                if split_indices[i]!=0 and split_indices[i+1]!=(len(chunk)-2-1):
                    indexing_list[which_state].append([i_chunk, split_indices[i],split_indices[i+1]])
            else:
                indexing_list[which_state].append([i_chunk, split_indices[i],split_indices[i+1]])

    # Convert lists to numpy arrays
    indexing_list = [np.asarray(indexing_list[i_state]) for i_state in range(max_state+1)]

    return indexing_list


def get_state_durations(latents, hmm):

    states = [hmm.most_likely_states(x) for x in latents]
    state_indices = get_discrete_chunks(states, include_edges=False)

    durations = []
    for i_state in range(0,len(state_indices)):
        if len(state_indices[i_state])>0:
             durations = np.append(durations,np.diff(state_indices[i_state][:,1:3],1))

    return durations


def relabel_states_by_use(states,mapping=None):
    '''
    Takes in discrete states and relabels according to mapping or length of time in each.

    input:
        states: list of trials, each trial is numpy array containing discrete state for each frame
        mapping: mapping you want to use if already calculated, format mapping[old_state]=new_state (for example if using training length of times mapping on validation data)

    output:
        relabeled_states: same data structure but with states relabeled by use (state 0 has most frames, etc)
        mapping: mapping of original labels to new labels, format mapping[old_state]=new_state

    '''
    frame_counts=[]
    if mapping is None:

        # Get number of frames for each state
        max_state = max([max(x) for x in states]) # Get maximum state
        bin_edges = np.arange(-.5,max_state+.7)

        frame_counts = np.zeros((max_state+1))
        for chunk in states:
            these_counts, _ = np.histogram(chunk,bin_edges)
            frame_counts += these_counts

        # Define mapping
        mapping = np.asarray(scipy.stats.rankdata(-frame_counts,method='ordinal')-1)

    # Remap states
    relabeled_states = [[]]*len(states)
    for i, chunk in enumerate(states):

        relabeled_states[i] = mapping[chunk]


    return relabeled_states, mapping, np.sort(frame_counts)[::-1]


def make_syllable_movies(filepath, hparams, latents, states, trial_idxs, data_generator, min_threshold=0, n_buffer=5, n_pre_frames=3):

    plot_n_frames = hparams['plot_n_frames']
    if hparams['plot_frame_rate'] == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams['plot_frame_rate']

    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[0][0]['images'].shape

    state_indices = get_discrete_chunks(states, include_edges=True)

    movie_dim1 = n_channels*y_dim
    movie_dim2 = x_dim

    actual_K = len(state_indices)

    # Get all example over threshold
    over_threshold_instances = [[] for _ in range(actual_K)]
    for i_state in range(actual_K):
        if state_indices[i_state].shape[0]>0:
            over_threshold_instances[i_state] = state_indices[i_state][(np.diff(state_indices[i_state][:,1:3],1)>min_threshold)[:,0]]
            np.random.shuffle(over_threshold_instances[i_state]) # Shuffle instances

    dim1 = int(np.floor(np.sqrt(actual_K)))
    dim2 = int(np.ceil(actual_K/dim1))

    # Initialize syllable movie frames
    plt.clf()
    fig_dim_div = movie_dim2*dim2/10 # aiming for dim 1 being 10
    fig, axes = plt.subplots(dim1,dim2,figsize=((movie_dim2*dim2)/fig_dim_div,(movie_dim1*dim1)/fig_dim_div))

    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Syllable '+str(i),fontsize=16)
    fig.tight_layout(pad=0)

    ims=[[] for _ in range(plot_n_frames+bs+200)]

    # Loop through syllables
    for i_syllable in range(actual_K):
        if len(over_threshold_instances[i_syllable])>0:
            i_chunk=0
            i_frame=0

            while i_frame < plot_n_frames:

                if i_chunk>=len(over_threshold_instances[i_syllable]):
                    im = fig.axes[i_syllable].imshow(np.zeros((movie_dim1,movie_dim2)),animated=True,vmin=0,vmax=1,cmap='gray')
                    ims[i_frame].append(im)
                    i_frame+=1
                else:

                    # Get movies/latents
                    which_trial = trial_idxs[over_threshold_instances[i_syllable][i_chunk,0]]
                    movie_chunk = data_generator.datasets[0][which_trial]['images'].cpu().detach().numpy()[max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]]
                    #movie_chunk = images[over_threshold_instances[i_syllable][i_chunk,0]][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]]

                    if hparams['lab']=='musall':
                        movie_chunk = np.transpose(movie_chunk,(0,1,3,2))
                    movie_chunk = np.concatenate([movie_chunk[:,j] for j in range(movie_chunk.shape[1])],axis=1)

                    latents_chunk = latents[over_threshold_instances[i_syllable][i_chunk,0]][over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames:over_threshold_instances[i_syllable][i_chunk,2]]

                    if np.sum(states[over_threshold_instances[i_syllable][i_chunk,0]][over_threshold_instances[i_syllable][i_chunk,1]:over_threshold_instances[i_syllable][i_chunk,2]-1]!=i_syllable)>0:
                        raise ValueError('Misaligned states for syllable segmentation')

                    # Loop over this chunk
                    for i in range(movie_chunk.shape[0]):

                        im = fig.axes[i_syllable].imshow(movie_chunk[i],animated=True,vmin=0,vmax=1,cmap='gray')
                        ims[i_frame].append(im)

                        # Add red box if start of syllable
                        syllable_start = n_pre_frames if over_threshold_instances[i_syllable][i_chunk,1]>=n_pre_frames else over_threshold_instances[i_syllable][i_chunk,1]

                        if i>syllable_start and i<(syllable_start+4):
                            rect =  matplotlib.patches.Rectangle((5,5),10,10,linewidth=1,edgecolor='r',facecolor='r')
                            im = fig.axes[i_syllable].add_patch(rect)
                            ims[i_frame].append(im)

                        i_frame+=1

                    # Add buffer black frames
                    for j in range(n_buffer):
                        im = fig.axes[i_syllable].imshow(np.zeros((movie_dim1,movie_dim2)),animated=True,vmin=0,vmax=1,cmap='gray')
                        ims[i_frame].append(im)
                        i_frame+=1

                    i_chunk+=1

    ani = animation.ArtistAnimation(fig, [ims[i] for i in range(len(ims)) if ims[i]!=[]], interval=20, blit=True, repeat=False)
    writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist='mrw'), bitrate=-1)
    save_file = os.path.join(filepath,'syllable_behavior_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(hparams['kappa'])+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.mp4')
    ani.save(save_file, writer=writer)

def make_real_vs_generated_movies(filepath, hparams, hmm, latents, states, data_generator, n_buffer=5):

    plot_n_frames = hparams['plot_n_frames']
    if hparams['plot_frame_rate'] == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams['plot_frame_rate']
    n_latents = hparams['n_ae_latents']
    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[0][0]['images'].shape


    ## Load in AE decoder
    ae_model_file = os.path.join(hparams['ae_model_path'],'best_val_model.pt')
    ae_arch = pickle.load(open(os.path.join(hparams['ae_model_path'],'meta_tags.pkl'),'rb'))
    ae_model = AE(ae_arch)
    ae_model.load_state_dict(torch.load(ae_model_file, map_location=lambda storage, loc: storage))
    ae_model.eval()


    # Get sampled observations
    sampled_observations = [np.zeros((len(state_seg),n_latents)) for state_seg in states]

    for i_seg, state_seg in enumerate(states):
        for i_t in range(len(state_seg)):
            if i_t>=1:
                sampled_observations[i_seg][i_t] = hmm.observations.sample_x(states[i_seg][i_t],sampled_observations[i_seg][:i_t])
            else:
                sampled_observations[i_seg][i_t] = hmm.observations.sample_x(states[i_seg][i_t],latents[i_seg][0].reshape((1,n_latents)))


    # Make real vs simulated arrays
    which_trials = np.arange(0,len(states)).astype('int')
    np.random.shuffle(which_trials)

    all_recon = np.zeros((0,n_channels*y_dim,x_dim))
    i_trial=0
    while all_recon.shape[0] < plot_n_frames:

        recon = ae_model.decoding(torch.tensor(latents[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
        if hparams['lab']=='musall':
            recon = np.transpose(recon,(0,1,3,2))
        recon = np.concatenate([recon[:,i] for i in range(recon.shape[1])],axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer,n_channels*y_dim,x_dim))

        all_recon = np.concatenate((all_recon,recon,zero_frames),axis=0)
        i_trial+=1


    all_simulated_recon = np.zeros((0,n_channels*y_dim,x_dim))
    i_trial=0
    while all_simulated_recon.shape[0] < plot_n_frames:

        simulated_recon = ae_model.decoding(torch.tensor(sampled_observations[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
        if hparams['lab']=='musall':
            simulated_recon = np.transpose(simulated_recon,(0,1,3,2))
        simulated_recon = np.concatenate([simulated_recon[:,i] for i in range(simulated_recon.shape[1])],axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer,n_channels*y_dim,x_dim))

        all_simulated_recon = np.concatenate((all_simulated_recon,simulated_recon,zero_frames),axis=0)
        i_trial+=1


    ## Make overlaid plot
    spc=3
    which_trial=which_trials[0]
    trial_len = len(states[which_trial])
    fig, axes = plt.subplots(2,1,sharex=True,sharey=True, figsize=(10,10))
    axes[0].imshow(states[which_trial][:trial_len][None,:],
                   aspect="auto",
                   extent=(0, trial_len, -spc-1, spc*n_latents),
                   cmap="jet", alpha=0.5)
    axes[0].plot(latents[which_trial] + spc * np.arange(n_latents), '-k', lw=1)
    axes[1].imshow(states[which_trial][:trial_len][None,:],
                   aspect="auto",
                   extent=(0, trial_len, -spc-1, spc*n_latents),
                   cmap="jet", alpha=0.5)
    axes[1].plot(sampled_observations[which_trial] + spc * np.arange(n_latents), '-k', lw=1)

    axes[0].set_title('Real Latents',fontsize=20)
    axes[1].set_title('Simulated Latents',fontsize=20)
    xlab = fig.text(0.5, -0.01, 'Time (frames)', ha='center',fontsize=20)
    ylab = fig.text(-0.01, 0.5, 'AE Dimensions', va='center', rotation='vertical',fontsize=20)
    for i in range(2):
        axes[i].set_yticks(spc * np.arange(n_latents))
        axes[i].set_yticklabels(np.arange(n_latents),fontsize=16)
    fig.tight_layout()
    save_file = os.path.join(filepath,'real_vs_generated_latents_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(hparams['kappa'])+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.png')
    fig.savefig(save_file,dpi=200)


    # Make videos
    plt.clf()
    fig_dim_div = x_dim*2/10 # aiming for dim 1 being 10
    fig, axes = plt.subplots(1,2,figsize=(x_dim*2/fig_dim_div,y_dim*n_channels/fig_dim_div))

    for j in range(2):
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    axes[0].set_title('Real Reconstructions',fontsize=16)
    axes[1].set_title('Simulated Reconstructions',fontsize=16)
    fig.tight_layout(pad=0)

    ims = []
    for i in range(plot_n_frames):

        ims_curr = []

        im = axes[0].imshow(all_recon[i],cmap='gray',vmin=0,vmax=1,animated=True)
        ims_curr.append(im)

        im = axes[1].imshow(all_simulated_recon[i],cmap='gray',vmin=0,vmax=1,animated=True)
        ims_curr.append(im)

        ims.append(ims_curr)

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat=False)
    writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist='mrw'))
    save_file = os.path.join(filepath,'real_vs_generated_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(hparams['kappa'])+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.mp4')
    ani.save(save_file, writer=writer)


