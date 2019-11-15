import pickle
import os
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from behavenet.plotting import make_dir_if_not_exists
from behavenet.models import AE as AE


def get_model_str(hparams):
    """Helper function for automatically generating figure names"""
    return str('K_%i_kappa_%0.e_noise_%s_nlags_%i' % (
        hparams['n_arhmm_states'], hparams['kappa'], hparams['noise_type'],
        hparams['n_arhmm_lags']))


def get_discrete_chunks(states, include_edges=True):
    """
    Find occurences of each discrete state

    Args:
        states (list): list of trials; each trial is numpy array containing discrete state for each
            frame
        include_edges (bool): include states at start and end of chunk

    Returns:
        list: list of length discrete states; each list contains all occurences of that discrete
            state by [chunk number, starting index, ending index]
    """

    max_state = max([max(x) for x in states])
    indexing_list = [[] for _ in range(max_state + 1)]

    for i_chunk, chunk in enumerate(states):

        # pad either side so we get start and end chunks
        chunk = np.pad(chunk, (1, 1), mode='constant', constant_values=-1)
        # don't add 1 because of start padding, this is now indice in original unpadded data
        split_indices = np.where(np.ediff1d(chunk) != 0)[0]
        # last index will be 1 higher that it should be due to padding
        split_indices[-1] -= 1

        for i in range(len(split_indices)-1):
            # get which state this chunk was (+1 because data is still padded)
            which_state = chunk[split_indices[i]+1]
            if not include_edges:
                if split_indices[i] != 0 and split_indices[i+1] != (len(chunk)-2-1):
                    indexing_list[which_state].append(
                        [i_chunk, split_indices[i], split_indices[i+1]])
            else:
                indexing_list[which_state].append(
                    [i_chunk, split_indices[i], split_indices[i + 1]])

    # convert lists to numpy arrays
    indexing_list = [np.asarray(indexing_list[i_state]) for i_state in range(max_state + 1)]

    return indexing_list


def get_state_durations(latents, hmm):
    """Calculate frame count for each state"""
    # TODO: bad return type when n_arhmm_states = 1
    states = [hmm.most_likely_states(x) for x in latents]
    state_indices = get_discrete_chunks(states, include_edges=False)
    durations = []
    for i_state in range(0, len(state_indices)):
        if len(state_indices[i_state]) > 0:
            durations = np.append(durations, np.diff(state_indices[i_state][:, 1:3], 1))
    return durations


def relabel_states_by_use(states, mapping=None):
    """
    Takes in discrete states and relabels according to mapping or length of time in each.

    Args:
        states (list): list of trials; each trial is numpy array containing discrete state for each
            frame
        mapping (array-like, optional): format is mapping[old_state] = new_state; for example if
            using training length of times mapping on validation data

    Returns:
        (tuple): (relabeled states, mapping, frame counts
            relabeled_states: same data structure but with states relabeled by use (state 0 has
                most frames, etc)
            mapping: mapping of original labels to new labels; mapping[old_state] = new_state
            frame counts: updated frame counts for relabeled states
    """
    frame_counts = []
    if mapping is None:
        # Get number of frames for each state
        max_state = max([max(x) for x in states])  # Get maximum state
        bin_edges = np.arange(-.5, max_state + .7)
        frame_counts = np.zeros((max_state + 1))
        for chunk in states:
            these_counts, _ = np.histogram(chunk, bin_edges)
            frame_counts += these_counts
        # define mapping
        mapping = np.asarray(scipy.stats.rankdata(-frame_counts,method='ordinal')-1)
    # remap states
    relabeled_states = [[]]*len(states)
    for i, chunk in enumerate(states):
        relabeled_states[i] = mapping[chunk]
    return relabeled_states, mapping, np.sort(frame_counts)[::-1]


def get_latent_arrays_by_dtype(data_generator, sess_idxs=0):
    """
    Collect data from data generator and put into dictionary with keys `train`, `test`, and `val`

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
            curr_idxs = dataset.batch_indxs[data_type]
            trial_idxs[data_type] += list(curr_idxs)
            latents[data_type] += [
                dataset[i_trial]['ae_latents'][:].cpu().detach().numpy() for i_trial in curr_idxs]
    return latents, trial_idxs


def get_model_latents_states(hparams, version, sess_idx=0, return_samples=0, cond_sampling=False):
    """
    Return arhmm defined in `hparams` with associated latents and states. Can also return generated
    latents and states.

    Args:
        hparams (dict):
        version (str or int):
        sess_idx (int, optional): session index into data generator
        return_samples (int, optional): number of trials to sample from model
        cond_sampling (bool, optional): if True return samples conditioned on discrete states from
            test trials; if False return unconditioned samples

    Returns:
        dict:
            'model': ssm object
            'latents': dict with latents from train, val and test trials
            'states': dict with states from train, val and test trials
            'trial_idxs': dict with trial indices from train, val and test trials
            'latents_gen': list of generated latents
            'states_gen': list of generated states
    """
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_best_model_version
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

    hparams['session_dir'], sess_ids = get_session_dir(hparams)
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
    if return_samples > 0:
        if cond_sampling:
            # TODO: for i in range(len(latents['test'])):
            raise NotImplementedError
        else:
            states_gen = []
            latents_gen = []
            np.random.seed(101)
            offset = 200
            for i in range(return_samples):
                these_states_gen, these_latents_gen = hmm.sample(
                    latents['test'][0].shape[0] + offset)
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


def make_syllable_movies_wrapper(
        hparams, save_file, sess_idx=0, dtype='test', max_frames=400, frame_rate=10,
        min_threshold=0, n_buffer=5, n_pre_frames=3, n_rows=None, single_syllable=None):
    """
    Present video clips of each individual syllable in separate panels

    Args:
        hparams (dict):
        save_file (str):
        sess_idx (int, optional): session index into data generator
        dtype (str, optional): types of trials to make video with; 'train', 'val', 'test'
        max_frames (int, optional): maximum number of frames to animate from a trial
        frame_rate (float, optional): frame rate of saved movie
        min_threshold (int, optional): minimum number of frames in a syllable run to be considered
            for movie
        n_buffer (int): number of blank frames between syllable instances
        n_pre_frames (int): number of behavioral frames to precede each syllable instance
        n_rows (int or NoneType): number of rows in output movie
        single_syllable (int or NoneType): choose only a single state for movie
    """
    from behavenet.data.data_generator import ConcatSessionsGenerator
    from behavenet.data.utils import get_data_generator_inputs
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

    # load images, latents, and states
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    hparams['load_videos'] = True
    hparams, signals, transforms, paths = get_data_generator_inputs(hparams, sess_ids)
    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], sess_ids,
        signals_list=[signals[sess_idx]],
        transforms_list=[transforms[sess_idx]],
        paths_list=[paths[sess_idx]],
        device='cpu', as_numpy=True, batch_load=False, rng_seed=hparams['rng_seed_data'])
    ims_orig = data_generator.datasets[sess_idx].data['images']
    del data_generator  # free up memory

    # get tt version number
    _, version = experiment_exists(hparams, which_version=True)
    print('producing syllable videos for arhmm %s' % version)
    # load latents
    _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
    with open(latents_file, 'rb') as f:
        latents = pickle.load(f)
    trial_idxs = latents['trials'][dtype]
    # load model
    model_file = os.path.join(hparams['expt_dir'], version, 'best_val_model.pt')
    with open(model_file, 'rb') as f:
        hmm = pickle.load(f)
    # infer discrete states
    states = [hmm.most_likely_states(latents['latents'][s]) for s in latents['trials'][dtype]]
    if len(states) == 0:
        raise ValueError('No latents for dtype=%s' % dtype)

    # find runs of discrete states; state indices is a list, each entry of which is a np array with
    # shape (n_state_instances, 3), where the 3 values are:
    # chunk_idx, chunk_start_idx, chunk_end_idx
    # chunk_idx is in [0, n_chunks], and indexes trial_idxs
    state_indices = get_discrete_chunks(states, include_edges=True)
    K = len(state_indices)

    # get all example over minimum state length threshold
    over_threshold_instances = [[] for _ in range(K)]
    for i_state in range(K):
        if state_indices[i_state].shape[0] > 0:
            state_lens = np.diff(state_indices[i_state][:, 1:3], axis=1)
            over_idxs = state_lens > min_threshold
            over_threshold_instances[i_state] = state_indices[i_state][over_idxs[:, 0]]
            np.random.shuffle(over_threshold_instances[i_state])  # shuffle instances

    make_syllable_movies(
        ims_orig=ims_orig,
        state_list=over_threshold_instances,
        trial_idxs=trial_idxs,
        save_file=save_file,
        max_frames=max_frames,
        frame_rate=frame_rate,
        n_buffer=n_buffer,
        n_pre_frames=n_pre_frames,
        n_rows=n_rows,
        single_syllable=single_syllable)


def make_syllable_movies(
        ims_orig, state_list, trial_idxs, save_file=None, max_frames=400, frame_rate=10,
        n_buffer=5, n_pre_frames=3, n_rows=None, single_syllable=None):
    """
    Present video clips of each individual syllable in separate panels

    Args:
        ims_orig (list): each entry (one per trial) is an np.ndarray of shape
            (n_frames, n_channels, y_pix, x_pix)
        state_list (list): each entry (one per state) is a
        trial_idxs (array-like): indices into `states` for which trials should be plotted
        save_file (str): directory for saving movie
        max_frames (int, optional): maximum number of frames to animate from a trial
        frame_rate (float, optional): frame rate of saved movie
        n_buffer (int): number of blank frames between syllable instances
        n_pre_frames (int): number of behavioral frames to precede each syllable instance
        n_rows (int or NoneType): number of rows in output movie
        single_syllable (int or NoneType): choose only a single state for movie
    """

    K = len(state_list)

    # Initialize syllable movie frames
    plt.clf()
    if single_syllable is not None:
        K = 1
        fig_width = 5
        n_rows = 1
    else:
        fig_width = 10  # aiming for dim 1 being 10
    # get video dims
    [bs, n_channels, y_dim, x_dim] = ims_orig[0].shape
    movie_dim1 = n_channels * y_dim
    movie_dim2 = x_dim
    if n_rows is None:
        n_rows = int(np.floor(np.sqrt(K)))
    n_cols = int(np.ceil(K / n_rows))

    fig_dim_div = movie_dim2 * n_cols / fig_width
    fig_width = (movie_dim2 * n_cols) / fig_dim_div
    fig_height = (movie_dim1 * n_rows) / fig_dim_div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if i >= K:
            ax.set_axis_off()
        elif single_syllable is not None:
            ax.set_title('Syllable %i' % single_syllable, fontsize=16)
        else:
            ax.set_title('Syllable %i' % i, fontsize=16)
    fig.tight_layout(pad=0, h_pad=1.005)

    imshow_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    ims = [[] for _ in range(max_frames + bs + 200)]

    # Loop through syllables
    for i_k, ax in enumerate(fig.axes):

        # skip if no syllable in this axis
        if i_k >= K:
            continue
        print('processing syllable %i/%i' % (i_k + 1, K))
        # skip if no syllables are longer than threshold
        if len(state_list[i_k]) == 0:
            continue

        if single_syllable is not None:
            i_k = single_syllable

        i_chunk = 0
        i_frame = 0

        while i_frame < max_frames:

            if i_chunk >= len(state_list[i_k]):
                # show blank if out of syllable examples
                im = ax.imshow(np.zeros((movie_dim1, movie_dim2)), **imshow_kwargs)
                ims[i_frame].append(im)
                i_frame += 1
            else:

                # Get movies/latents
                chunk_idx = state_list[i_k][i_chunk, 0]
                which_trial = trial_idxs[chunk_idx]
                tr_beg = state_list[i_k][i_chunk, 1]
                tr_end = state_list[i_k][i_chunk, 2]
                batch = ims_orig[which_trial]
                movie_chunk = batch[max(tr_beg - n_pre_frames, 0):tr_end]

                movie_chunk = np.concatenate(
                    [movie_chunk[:, j] for j in range(movie_chunk.shape[1])], axis=1)

                # if np.sum(states[chunk_idx][tr_beg:tr_end-1] != i_k) > 0:
                #     raise ValueError('Misaligned states for syllable segmentation')

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

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig,
        [ims[i] for i in range(len(ims)) if ims[i] != []], interval=20, blit=True, repeat=False)
    writer = FFMpegWriter(fps=max(frame_rate, 10), bitrate=-1)
    print('done')

    if save_file is not None:
        # put together file name
        if save_file[-3:] == 'mp4':
            save_file = save_file[:-3]
        if single_syllable is not None:
            state_str = str('_syllable-%02i' % single_syllable)
        else:
            state_str = ''
        save_file += state_str
        save_file += '.mp4'
        make_dir_if_not_exists(save_file)
        print('saving video to %s...' % save_file, end='')
        ani.save(save_file, writer=writer)
        print('done')


def real_vs_generated_samples_wrapper(
        output, hparams, save_file, sess_idx, dtype='test', conditional=True, max_frames=400,
        frame_rate=20, n_buffer=5):
    # if output == 'both' or output == 'movie':
    #     if conditional:
    #         make_real_vs_conditionally_generated_movies()
    #     else:
    #         make_real_vs_unconditionally_generated_movies()
    # if output == 'both' or output == 'plot':
    #     if conditional:
    #         plot_real_vs_conditionally_generated_samples()
    #     else:
    #         plot_real_vs_unconditionally_generated_samples()
    pass


def make_real_vs_conditionally_generated_movies(
        filepath, hparams, hmm, latents, states, data_generator, sess_idx=0, n_buffer=5, ptype=0,
        xtick_locs=None):

    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

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
        hparams['session_dir'], sess_ids = get_session_dir(hparams)
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


def plot_real_vs_conditionally_generated_samples():
    pass


def make_real_vs_unconditionally_generated_movies(
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


def plot_real_vs_unconditionally_generated_samples():
    pass


def plot_states_overlaid_with_latents(
        latents, states, save_file=None, ax=None, xtick_locs=None, frame_rate=None, format='png'):
    """
    Plot states for a single trial overlaid with latents

    Args:
        latents (np.ndarray): (n_frames, n_latents)
        states (np.ndarray): (n_frames,)
        save_file (str, optional):
        ax (matplotlib.Axes object, optional):
        xtick_locs (array-like, optional): tick locations in bin values
        frame_rate (float, optional): behavioral video framerate; to properly relabel xticks
        format (str, optional): e.g. 'png' | 'pdf' | 'jpeg'

    Returns:
        matplotlib figure handle if `ax` is None, otherwise updated axis
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.gca()
    else:
        fig = None
    spc = 1.1 * abs(latents.max())
    n_latents = latents.shape[1]
    plotting_latents = latents + spc * np.arange(n_latents)
    ymin = min(-spc - 1, np.min(plotting_latents))
    ymax = max(spc * n_latents, np.max(plotting_latents))
    ax.imshow(
        states[None, :], aspect='auto', extent=(0, len(latents), ymin, ymax), cmap='tab20b',
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

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    if fig is None:
        return ax
    else:
        return fig


def plot_state_transition_matrix(model, deridge=False):
    """
    Plot Markov transition matrix for arhmm

    Args:
        model (ssm object):
        deridge (bool): remove diagonal to more clearly see dynamic range of off-diagonal entries
    """
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
    """
    Plot autoregressive dynamics matrices for arhmm

    Args:
        model (ssm object):
        deridge (bool): remove diagonal to more clearly see dynamic range of off-diagonal entries
    """
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
    """
    Plot observation bias vectors for arhmm

    Args:
        model (ssm object)
    """
    fig = plt.figure(figsize=(6, 4))
    mats = np.copy(model.observations.bs.T)
    clim = np.max(np.abs(mats))
    plt.imshow(mats, cmap='RdBu_r', clim=[-clim, clim], aspect='auto')
    plt.xlabel('State')
    plt.yticks([])
    plt.ylabel('Observation dimension')
    plt.tight_layout()
    plt.colorbar()
    plt.title('State biases')
    plt.show()
    return fig


def plot_obs_covariance_matrices(model):
    """
    Plot observation covariance matrices for arhmm

    Args:
        model (ssm object)
    """
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
