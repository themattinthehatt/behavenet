import pickle
import os
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from behavenet.analyses import make_dir_if_not_exists
from behavenet.models import AE as AE


def get_model_str(hparams):
    return str('K_%i_kappa_%0.e_noise_%s_nlags_%i' % (
        hparams['n_arhmm_states'], hparams['kappa'], hparams['noise_type'], hparams['n_lags']))


def get_discrete_chunks(states, include_edges=True):
    """
    Find occurences of each discrete state

    Args:
        states: list of trials, each trial is numpy array containing discrete state for each frame
        include_edges: include states at start and end of chunk

    Returns:
        indexing_list: list of length discrete states, each list contains all occurences of that
            discrete state by [chunk number, starting index, ending index]

    """

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
    for i_state in range(0, len(state_indices)):
        if len(state_indices[i_state]) > 0:
             durations = np.append(durations, np.diff(state_indices[i_state][:, 1:3], 1))
    # TODO: bad return type when n_arhmm_states=1

    return durations


def relabel_states_by_use(states, mapping=None):
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


def get_latent_arrays_by_dtype(data_generator, sess_idxs=0):
    """
    Collect data from data generator and put into dictionary with keys `train`,
    `test`, and `val`

    Args:
        data_generator (ConcatSessionsGenerator):
        sess_idxs (int or list): concatenate train/test/val data across
            multiple sessions

    Returns:
        (tuple): latents dict, trial indices dict
    """
    if isinstance(sess_idxs, int):
        sess_idxs = [sess_idxs]
    dtypes = ['train', 'val', 'test']
    latents = {key: [] for key in dtypes}
    trial_idxs = {key: [] for key in dtypes}
    for sess_idx in sess_idxs:
        dataset = data_generator.datasets[sess_idx]
        for data_type in dtypes:
            trial_idxs[data_type] = dataset.batch_indxs[data_type]
            latents[data_type] += [dataset[i_trial]['ae_latents'][:].cpu().detach().numpy()
                                   for i_trial in trial_idxs[data_type]]
    return latents, trial_idxs


def get_model_latents_states(hparams, version, sess_idx=0, generated=False):
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_best_model_version
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_output_session_dir

    hparams['session_dir'], sess_ids = get_output_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    # get version/model
    if version == 'best':
        version = get_best_model_version(hparams['expt_dir'], measure='val_ll', best_def='max')[0]
    else:
        _, version = experiment_exists(hparams, which_version=True)
    if version is None:
        raise FileNotFoundError(
            'Could not find the specified model version in %s' % hparams['expt_dir'])
    # load model
    model_file = os.path.join(hparams['expt_dir'], version, 'best_val_model.pt')
    with open(model_file, 'rb') as f:
        hmm = pickle.load(f)
        # load latents
    _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
    with open(latents_file, 'rb') as f:
        all_latents = pickle.load(f)
    # collect inferred latents/states
    trial_idxs = {}
    latents = {}
    states = {}
    for data_type in ['train', 'val', 'test']:
        trial_idxs[data_type] = all_latents['trials'][data_type]
        latents[data_type] = [all_latents['latents'][i_trial] for i_trial in trial_idxs[data_type]]
        states[data_type] = [hmm.most_likely_states(x) for x in latents[data_type]]
    # collect generative latents/states
    if generated:
        states_gen = []
        latents_gen = []
        np.random.seed(101)
        offset = 200
        for i in range(len(latents['test'])):
            these_states_gen, these_latents_gen = hmm.sample(latents['test'][0].shape[0] + offset)
            latents_gen.append(these_latents_gen[offset:])
            states_gen.append(these_states_gen[offset:])
    else:
        latents_gen = []
        states_gen = []
    return_dict = {
        'model': hmm,
        'latents': latents,
        'states': states,
        'trial_idxs': trial_idxs,
        'latents_gen': latents_gen,
        'states_gen': states_gen,
    }
    return return_dict


def make_syllable_movies(
        filepath, hparams, latents, states, trial_idxs, data_generator, sess_idx=0,
        min_threshold=0, n_buffer=5, n_pre_frames=3, n_rows=None, append_str=None,
        single_syllable=None):
    """
    Present video clips of each individual syllable in separate panels

    Args:
        filepath (str): directory for saving movie
        hparams (dict): for generating save name
        latents (list of np.ndarrays):
        states (list of np.ndarrays): inferred state for each time point
        trial_idxs (array-like): indices into `states` for which trials should be plotted
        data_generator (ConcatSessionsGenerator):
        sess_idx (int): session index into data_generator
        min_threshold (int): minimum number of frames in a syllable run to be considered for movie
        n_buffer (int): number of blank frames between syllable instances
        n_pre_frames (int): number of behavioral frames to precede each syllable instance
        n_rows (int or NoneType): number of rows in output movie
        append_str (str): appended to end of filename before saving
        single_syllable (int or NoneType): choose only a single state for movie

    Returns:
        None; saves movie to `filepath/model_name_append_str.mp4`
    """

    plot_n_frames = hparams['plot_n_frames']
    if hparams['plot_frame_rate'] == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams['plot_frame_rate']

    # find runs of discrete states; state indices is a list, each entry of which is a np array with
    # shape (n_state_instances, 3), where the 3 values are:
    # chunk_idx, chunk_start_idx, chunk_end_idx
    # chunk_idx is in [0, n_chunks], and indexes trial_idxs
    state_indices = get_discrete_chunks(states, include_edges=True)
    actual_K = len(state_indices)

    # get all example over minimum state length threshold
    over_threshold_instances = [[] for _ in range(actual_K)]
    for i_state in range(actual_K):
        if state_indices[i_state].shape[0] > 0:
            state_lens = np.diff(state_indices[i_state][:, 1:3], axis=1)
            over_idxs = state_lens > min_threshold
            over_threshold_instances[i_state] = state_indices[i_state][over_idxs[:, 0]]
            np.random.shuffle(over_threshold_instances[i_state])  # shuffle instances

    # Initialize syllable movie frames
    plt.clf()
    if single_syllable is not None:
        actual_K = 1
        fig_width = 5
        n_rows = 1
    else:
        fig_width = 10  # aiming for dim 1 being 10
    # get video dims
    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[sess_idx][0]['images'].shape
    movie_dim1 = n_channels * y_dim
    movie_dim2 = x_dim
    if n_rows is None:
        n_rows = int(np.floor(np.sqrt(actual_K)))
    n_cols = int(np.ceil(actual_K / n_rows))

    fig_dim_div = movie_dim2 * n_cols / fig_width
    fig_width = (movie_dim2 * n_cols) / fig_dim_div
    fig_height = (movie_dim1 * n_rows) / fig_dim_div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if i >= actual_K:
            ax.set_axis_off()
        elif single_syllable is not None:
            ax.set_title('Syllable %i' % single_syllable, fontsize=16)
        else:
            ax.set_title('Syllable %i' % i, fontsize=16)
    fig.tight_layout(pad=0, h_pad=1.005)

    imshow_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    ims = [[] for _ in range(plot_n_frames + bs + 200)]

    # Loop through syllables
    for i_k, ax in enumerate(fig.axes):

        # skip if no syllable in this axis
        if i_k >= actual_K:
            continue
        print('processing syllable %i/%i' % (i_k + 1, actual_K))
        # skip if no syllables are longer than threshold
        if len(over_threshold_instances[i_k]) == 0:
            continue

        if single_syllable is not None:
            i_k = single_syllable

        i_chunk = 0
        i_frame = 0

        while i_frame < plot_n_frames:

            if i_chunk >= len(over_threshold_instances[i_k]):
                # show blank if out of syllable examples
                im = ax.imshow(np.zeros((movie_dim1, movie_dim2)), **imshow_kwargs)
                ims[i_frame].append(im)
                i_frame += 1
            else:

                # Get movies/latents
                chunk_idx = over_threshold_instances[i_k][i_chunk, 0]
                which_trial = trial_idxs[chunk_idx]
                tr_beg = over_threshold_instances[i_k][i_chunk, 1]
                tr_end = over_threshold_instances[i_k][i_chunk, 2]
                batch = data_generator.datasets[sess_idx][which_trial]['images'].cpu().detach().numpy()
                movie_chunk = batch[max(tr_beg - n_pre_frames, 0):tr_end]

                if hparams['lab'] == 'musall':
                    movie_chunk = np.transpose(movie_chunk, (0, 1, 3, 2))
                movie_chunk = np.concatenate(
                    [movie_chunk[:, j] for j in range(movie_chunk.shape[1])], axis=1)

                if np.sum(states[chunk_idx][tr_beg:tr_end-1] != i_k) > 0:
                    raise ValueError('Misaligned states for syllable segmentation')

                # Loop over this chunk
                for i in range(movie_chunk.shape[0]):

                    im = ax.imshow(movie_chunk[i], **imshow_kwargs)
                    ims[i_frame].append(im)

                    # Add red box if start of syllable
                    syllable_start = n_pre_frames if tr_beg >= n_pre_frames else tr_beg

                    if syllable_start <= i < (syllable_start + 2):
                        rect = matplotlib.patches.Rectangle(
                            (5, 5), 10, 10, linewidth=1, edgecolor='r', facecolor='r')
                        im = ax.add_patch(rect)
                        ims[i_frame].append(im)

                    i_frame += 1

                # Add buffer black frames
                for j in range(n_buffer):
                    im = ax.imshow(np.zeros((movie_dim1, movie_dim2)), **imshow_kwargs)
                    ims[i_frame].append(im)
                    i_frame += 1

                i_chunk += 1

    # put together file name
    if single_syllable is not None:
        state_str = str('_syllable-%02i' % single_syllable)
    else:
        state_str = ''
    filename = str(
        'syllable_behavior_%s%s%s.mp4' % (get_model_str(hparams), state_str, append_str))
    save_file = os.path.join(filepath, filename)
    make_dir_if_not_exists(save_file)

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig,
        [ims[i] for i in range(len(ims)) if ims[i] != []], interval=20, blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % filename, end='')
    writer = FFMpegWriter(fps=max(plot_frame_rate, 10), bitrate=-1)
    ani.save(save_file, writer=writer)
    print('done')


def make_real_vs_generated_movies(
        filepath, hparams, hmm, latents, states, data_generator, sess_idx=0, n_buffer=5, ptype=0,
        xtick_locs=None):

    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_output_session_dir

    plot_n_frames = hparams.get('plot_n_frames', 400)
    if hparams.get('plot_frame_rate', None) == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams.get('plot_frame_rate', 15)
    n_latents = hparams['n_ae_latents']
    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[sess_idx][0]['images'].shape

    # Load in AE decoder
    if hparams.get('ae_model_path', None) is not None:
        ae_model_file = os.path.join(hparams['ae_model_path'], 'best_val_model.pt')
        ae_arch = pickle.load(open(os.path.join(hparams['ae_model_path'], 'meta_tags.pkl'), 'rb'))
    else:
        hparams['session_dir'], sess_ids = get_output_session_dir(hparams)
        hparams['expt_dir'] = get_expt_dir(hparams)
        _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
        # _, version = experiment_exists(hparams, which_version=True)
        ae_model_file = os.path.join(os.path.dirname(latents_file), 'best_val_model.pt')
        ae_arch = pickle.load(open(os.path.join(os.path.dirname(latents_file), 'meta_tags.pkl'), 'rb'))
        print('loading model from %s' % ae_model_file)
    ae_model = AE(ae_arch)
    ae_model.load_state_dict(torch.load(ae_model_file, map_location=lambda storage, loc: storage))
    ae_model.eval()

    # Get sampled observations
    sampled_observations = [np.zeros((len(state_seg), n_latents)) for state_seg in states]
    for i_seg, state_seg in enumerate(states):
        for i_t in range(len(state_seg)):
            if i_t >= 1:
                sampled_observations[i_seg][i_t] = hmm.observations.sample_x(
                    states[i_seg][i_t], sampled_observations[i_seg][:i_t], input=np.zeros(0))
            else:
                sampled_observations[i_seg][i_t] = hmm.observations.sample_x(
                    states[i_seg][i_t], latents[i_seg][0].reshape((1, n_latents)), np.zeros(0))

    # Make real vs simulated arrays
    which_trials = np.arange(0, len(states)).astype('int')
    np.random.shuffle(which_trials)

    # reconstruct observed latents
    all_recon = np.zeros((0, n_channels*y_dim, x_dim))
    i_trial = 0
    while all_recon.shape[0] < plot_n_frames:

        recon = ae_model.decoding(
            torch.tensor(latents[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
        if hparams['lab'] == 'musall':
            recon = np.transpose(recon, (0, 1, 3, 2))
        recon = np.concatenate([recon[:, i] for i in range(recon.shape[1])], axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer, n_channels*y_dim, x_dim))

        all_recon = np.concatenate((all_recon, recon, zero_frames), axis=0)
        i_trial += 1

    # reconstruct sampled latents
    all_simulated_recon = np.zeros((0, n_channels*y_dim, x_dim))
    i_trial = 0
    while all_simulated_recon.shape[0] < plot_n_frames:

        simulated_recon = ae_model.decoding(
            torch.tensor(sampled_observations[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
        if hparams['lab'] == 'musall':
            simulated_recon = np.transpose(simulated_recon, (0, 1, 3, 2))
        simulated_recon = np.concatenate([simulated_recon[:, i] for i in range(simulated_recon.shape[1])], axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer, n_channels*y_dim, x_dim))

        all_simulated_recon = np.concatenate((all_simulated_recon,simulated_recon,zero_frames), axis=0)
        i_trial += 1

    # Make overlaid plot
    which_trial = which_trials[0]
    trial_len = len(states[which_trial])
    if ptype == 0:
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 10))
        spc = 3
        axes[0].imshow(states[which_trial][:trial_len][None,:],
                       aspect="auto",
                       extent=(0, trial_len, -spc-1, spc*n_latents),
                       cmap="jet", alpha=0.5)
        axes[0].plot(latents[which_trial] + spc * np.arange(n_latents), '-k', lw=3)
        axes[1].imshow(states[which_trial][:trial_len][None, :],
                       aspect="auto",
                       extent=(0, trial_len, -spc-1, spc*n_latents),
                       cmap="jet", alpha=0.5)
        axes[1].plot(sampled_observations[which_trial] + spc * np.arange(n_latents), '-k', lw=3)
        axes[0].set_title('Real Latents',fontsize=20)
        axes[1].set_title('Simulated Latents',fontsize=20)
        xlab = fig.text(0.5, -0.01, 'Time (frames)', ha='center',fontsize=20)
        ylab = fig.text(-0.01, 0.5, 'AE Dimensions', va='center', rotation='vertical',fontsize=20)
        for i in range(2):
            axes[i].set_yticks(spc * np.arange(n_latents))
            axes[i].set_yticklabels(np.arange(n_latents),fontsize=16)
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_states = states[which_trial][:trial_len]
        plot_latents = latents[which_trial][:trial_len]
        plot_obs = sampled_observations[which_trial][:trial_len]

        axes[0] = plot_states_overlaid_with_latents(
            plot_latents, plot_states,
            ax=axes[0], xtick_locs=xtick_locs, frame_rate=hparams.get('frame_rate', None))
        axes[0].set_xticks([])
        axes[0].set_xlabel('')
        axes[0].set_title('Inferred latents')

        axes[1] = plot_states_overlaid_with_latents(
            plot_obs, plot_states,
            ax=axes[1], xtick_locs=xtick_locs, frame_rate=hparams.get('frame_rate', None))
        axes[1].set_title('Generated latents')

    save_file = os.path.join(
        filepath, 'real_vs_generated_latents_' + get_model_str(hparams) + '.png')
    fig.savefig(save_file, dpi=200)

    # Make videos
    # plt.clf()
    # fig_dim_div = x_dim*2/10 # aiming for dim 1 being 10
    # fig, axes = plt.subplots(1,2,figsize=(x_dim*2/fig_dim_div,y_dim*n_channels/fig_dim_div))
    #
    # for j in range(2):
    #     axes[j].set_xticks([])
    #     axes[j].set_yticks([])
    #
    # axes[0].set_title('Real Reconstructions', fontsize=16)
    # axes[1].set_title('Generative Reconstructions', fontsize=16)
    # fig.tight_layout(pad=0)
    #
    # im_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'animated': True}
    # ims = []
    # for i in range(plot_n_frames):
    #
    #     ims_curr = []
    #
    #     im = axes[0].imshow(all_recon[i], **im_kwargs)
    #     ims_curr.append(im)
    #
    #     im = axes[1].imshow(all_simulated_recon[i], **im_kwargs)
    #     ims_curr.append(im)
    #
    #     ims.append(ims_curr)
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat=False)
    # writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist='mrw'))
    # save_file = os.path.join(
    #     filepath, 'real_vs_generated_' + get_model_str(hparams) + '.mp4')
    # make_dir_if_not_exists(save_file)
    # ani.save(save_file, writer=writer)


def make_real_vs_nonconditioned_generated_movies(
        filepath, hparams, real_latents, generated_latents, data_generator, trial_idxs, n_buffer=5):

    plot_n_frames = hparams['plot_n_frames']
    if hparams['plot_frame_rate'] == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams['plot_frame_rate']

    n_latents = hparams['n_ae_latents']
    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[0][0]['images'].shape


    ## Load in AE decoder
    ae_model_file = os.path.join(hparams['ae_model_path'],'best_val_model.pt')
    print(ae_model_file)
    ae_arch = pickle.load(open(os.path.join(hparams['ae_model_path'],'meta_tags.pkl'),'rb'))
    ae_model = AE(ae_arch)
    ae_model.load_state_dict(torch.load(ae_model_file, map_location=lambda storage, loc: storage))
    ae_model.eval()


    # Make original videos vs real recons vs simulated recons arrays
    #which_trials = np.arange(0,len(real_latents)).astype('int')
    #np.random.shuffle(which_trials)
    if hparams['lab_example']=='steinmetz-face':
        print('steinmetz-face')
        which_trials = np.asarray([4,9])
    elif hparams['lab_example']=='steinmetz':
        print('steinmetz')
        which_trials = np.asarray([4,9])
    else:
        which_trials = np.arange(0,len(real_latents)).astype('int')
        np.random.shuffle(which_trials)  
    all_orig = np.zeros((0,n_channels*y_dim,x_dim))
    i_trial=0
    while all_orig.shape[0] < plot_n_frames:

        orig = data_generator.datasets[0][trial_idxs[which_trials[i_trial]]]['images'].cpu().detach().numpy()
        #recon = ae_model.decoding(torch.tensor(latents[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
        if hparams['lab']=='musall':
            orig = np.transpose(orig,(0,1,3,2))
        orig = np.concatenate([orig[:,i] for i in range(orig.shape[1])],axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer,n_channels*y_dim,x_dim))

        all_orig = np.concatenate((all_orig,orig,zero_frames),axis=0)
        i_trial+=1


    all_recon = np.zeros((0,n_channels*y_dim,x_dim))
    i_trial=0
    while all_recon.shape[0] < plot_n_frames:

        recon = ae_model.decoding(torch.tensor(real_latents[which_trials[i_trial]]).float(), None, None).cpu().detach().numpy()
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

        simulated_recon = ae_model.decoding(torch.tensor(generated_latents[i_trial]).float(), None, None).cpu().detach().numpy()
        if hparams['lab']=='musall':
            simulated_recon = np.transpose(simulated_recon,(0,1,3,2))
        simulated_recon = np.concatenate([simulated_recon[:,i] for i in range(simulated_recon.shape[1])],axis=1)

        # Add a few black frames
        zero_frames = np.zeros((n_buffer,n_channels*y_dim,x_dim))

        all_simulated_recon = np.concatenate((all_simulated_recon,simulated_recon,zero_frames),axis=0)
        i_trial+=1


    # Make videos
    plt.clf()
    fig_dim_div = x_dim*3/10 # aiming for dim 1 being 10
    fig, axes = plt.subplots(1,3,figsize=(x_dim*3/fig_dim_div,y_dim*n_channels/fig_dim_div))

    for j in range(3):
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    axes[0].set_title('Original Frames',fontsize=16)
    axes[1].set_title('Real Reconstructions',fontsize=16)
    axes[2].set_title('Generative Reconstructions',fontsize=16)
    fig.tight_layout(pad=0)

    ims = []
    for i in range(plot_n_frames):

        ims_curr = []

        im = axes[0].imshow(all_orig[i],cmap='gray',vmin=0,vmax=1,animated=True)
        ims_curr.append(im)

        im = axes[1].imshow(all_recon[i],cmap='gray',vmin=0,vmax=1,animated=True)
        ims_curr.append(im)

        im = axes[2].imshow(all_simulated_recon[i],cmap='gray',vmin=0,vmax=1,animated=True)
        ims_curr.append(im)

        ims.append(ims_curr)

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat=False)
    writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist='mrw'))
    save_file = os.path.join(filepath, hparams['dataset_name']+'_real_vs_nonconditioned_generated_K_'+str(hparams['n_arhmm_states'])+'_kappa_'+str(hparams['kappa'])+'_noise_'+hparams['noise_type']+'_nlags_'+str(hparams['n_lags'])+'.mp4')
    make_dir_if_not_exists(save_file)
    ani.save(save_file, writer=writer)


def plot_segmentations_by_trial(
        states, trial_info_dict=None, xtick_locs=None, frame_rate=None, save_file=None,
        title=None, cmap='tab20b'):

    from matplotlib.lines import Line2D

    colors = ['r', 'k', 'g', 'b']
    if trial_info_dict is not None:
        line_kwargs = {'ymin': 0, 'ymax': 1, 'linewidth': 6, 'clip_on': False, 'alpha': 1}

    n_trials = len(states)

    fig = plt.figure(figsize=(10, n_trials / 4))
    gs_bottom_left = plt.GridSpec(n_trials, 1, top=0.85, right=1)
    for i_trial in range(n_trials):

        axes = plt.subplot(gs_bottom_left[i_trial, 0])
        axes.imshow(
            states[i_trial][None, :], aspect='auto',
            extent=(0, len(states[i_trial]), 0, 1), cmap=cmap) #, alpha=0.9)
        if trial_info_dict is not None:
            i = 0
            for key, val in trial_info_dict.items():
                if val[i_trial] >= 0:
                    axes.axvline(x=val[i_trial], c=colors[i], label=key, **line_kwargs)
                i += 1

        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)

    if trial_info_dict is not None:
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        plt.legend(
            lines, trial_info_dict.keys(), loc='center left', bbox_to_anchor=(1, 15),
            frameon=False)

    if xtick_locs is not None and frame_rate is not None:
        axes.set_xticks(xtick_locs)
        axes.set_xticklabels((np.asarray(xtick_locs) / frame_rate).astype('int'))
        axes.set_xlabel('Time (s)')
    else:
        axes.set_xlabel('Time (bins)')
    axes = plt.subplot(gs_bottom_left[int(np.floor(n_trials / 2)), 0])
    axes.set_ylabel('Trials')

    plt.suptitle(title)
    plt.tight_layout()

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        fig.savefig(save_file, transparent=True, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_states_overlaid_with_latents(
        latents_trial, states_trial, ax, xtick_locs=None, frame_rate=None):
    spc = 1.1 * abs(latents_trial.max())
    n_latents = latents_trial.shape[1]
    plotting_latents = latents_trial + spc * np.arange(n_latents)
    ymin = min(-spc - 1, np.min(plotting_latents))
    ymax = max(spc * n_latents, np.max(plotting_latents))
    ax.imshow(
        states_trial[None, :],
        aspect='auto',
        extent=(0, len(latents_trial), ymin, ymax),
        cmap='tab20b',
        alpha=1.0)
    ax.plot(plotting_latents, '-k', lw=3)
    ax.set_ylim([ymin, ymax])
    #     yticks = spc * np.arange(n_latents)
    #     ax.set_yticks(yticks[::2])
    #     ax.set_yticklabels(np.arange(n_latents)[::2])
    ax.set_yticks([])
    #     ax.set_ylabel('Latent')

    ax.set_xlabel('Time (bins)')
    if xtick_locs is not None:
        ax.set_xticks(xtick_locs)
        if frame_rate is not None:
            ax.set_xticklabels((np.asarray(xtick_locs) / frame_rate).astype('int'))
            ax.set_xlabel('Time (sec)')
    return ax


def plot_state_transition_matrix(model, deridge=False):
    trans = np.copy(model.transitions.transition_matrix)
    if deridge:
        n_states = trans.shape[0]
        for i in range(n_states):
            trans[i, i] = np.nan
        clim = np.nanmax(np.abs(trans))
    else:
        clim = 1
    fig = plt.figure()
    plt.imshow(trans, clim=[-clim, clim], cmap='RdBu_r')
    plt.colorbar()
    plt.ylabel('State (t)')
    plt.xlabel('State (t+1)')
    plt.title('State transition matrix')
    plt.show()
    return fig


def plot_dynamics_matrices(model, deridge=False):
    K = model.K
    D = model.D
    n_lags = model.observations.lags
    if n_lags == 1:
        n_cols = 3
        fac = 1
    elif n_lags == 2:
        n_cols = 3
        fac = 1 / n_lags
    elif n_lags == 3:
        n_cols = 3
        fac = 1.25 / n_lags
    elif n_lags == 4:
        n_cols = 3
        fac = 1.50 / n_lags
    elif n_lags == 5:
        n_cols = 2
        fac = 1.75 / n_lags
    else:
        n_cols = 1
        fac = 1
    n_rows = int(np.ceil(K / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows * fac))

    mats = np.copy(model.observations.As)
    if deridge:
        for k in range(K):
            for d in range(model.D):
                mats[k, d, d] = np.nan
        clim = np.nanmax(np.abs(mats))
    else:
        clim = np.max(np.abs(mats))

    for k in range(K):
        plt.subplot(n_rows, n_cols, k + 1)
        im = plt.imshow(mats[k], cmap='RdBu_r', clim=[-clim, clim])
        for l in range(n_lags - 1):
            plt.axvline((l + 1) * D - 0.5, ymin=0, ymax=K, color=[0, 0, 0])
        plt.xticks([])
        plt.yticks([])
        plt.title('State %i' % k)
    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)
    # plt.suptitle('Dynamics matrices')

    return fig


def plot_obs_biases(model):
    fig = plt.figure(figsize=(6, 4))

    mats = np.copy(model.observations.bs.T)
    clim = np.max(np.abs(mats))
    im = plt.imshow(mats, cmap='RdBu_r', clim=[-clim, clim], aspect='auto')
    plt.xlabel('State')
    plt.yticks([])
    plt.ylabel('Observation dimension')
    plt.tight_layout()
    plt.colorbar()
    plt.title('State biases')
    plt.show()
    return fig


def plot_obs_covariance_matrices(model):
    K = model.K
    n_cols = int(np.sqrt(K))
    n_rows = int(np.ceil(K / n_cols))

    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))

    mats = np.copy(model.observations.Sigmas)
    clim = np.quantile(np.abs(mats), 0.95)

    for k in range(K):
        plt.subplot(n_rows, n_cols, k + 1)
        im = plt.imshow(mats[k], cmap='RdBu_r', clim=[-clim, clim])
        plt.xticks([])
        plt.yticks([])
        plt.title('State %i' % k)
    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    return fig
