"""Plotting and video making functions for ARHMMs."""

import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from behavenet import make_dir_if_not_exists
from behavenet.models import AE as AE
from behavenet.plotting import save_movie

# to ignore imports for sphix-autoapidoc
__all__ = [
    'get_discrete_chunks', 'get_state_durations', 'get_latent_arrays_by_dtype',
    'get_model_latents_states',
    'make_syllable_movies_wrapper', 'make_syllable_movies',
    'real_vs_sampled_wrapper', 'make_real_vs_sampled_movies', 'plot_real_vs_sampled',
    'plot_states_overlaid_with_latents', 'plot_state_transition_matrix', 'plot_dynamics_matrices',
    'plot_obs_biases', 'plot_obs_covariance_matrices']


def get_discrete_chunks(states, include_edges=True):
    """Find occurences of each discrete state.

    Parameters
    ----------
    states : :obj:`list`
        list of trials; each trial is numpy array containing discrete state for each frame
    include_edges : :obj:`bool`
        include states at start and end of chunk

    Returns
    -------
    :obj:`list`
        list of length discrete states; each list contains all occurences of that discrete state by
        :obj:`[chunk number, starting index, ending index]`

    """

    max_state = max([max(x) for x in states])
    indexing_list = [[] for _ in range(max_state + 1)]

    for i_chunk, chunk in enumerate(states):

        # pad either side so we get start and end chunks
        chunk = np.pad(chunk, (1, 1), mode='constant', constant_values=-1)
        # don't add 1 because of start padding, this is now index in original unpadded data
        split_indices = np.where(np.ediff1d(chunk) != 0)[0]
        # last index will be 1 higher that it should be due to padding
        # split_indices[-1] -= 1

        for i in range(len(split_indices)-1):
            # get which state this chunk was (+1 because data is still padded)
            which_state = chunk[split_indices[i]+1]
            if not include_edges:
                if split_indices[i] != 0 and split_indices[i+1] != (len(chunk)-2):
                    indexing_list[which_state].append(
                        [i_chunk, split_indices[i], split_indices[i+1]])
            else:
                indexing_list[which_state].append(
                    [i_chunk, split_indices[i], split_indices[i+1]])

    # convert lists to numpy arrays
    indexing_list = [np.asarray(indexing_list[i_state]) for i_state in range(max_state + 1)]

    return indexing_list


def get_state_durations(latents, hmm, include_edges=True):
    """Calculate frame count for each state.

    Parameters
    ----------
    latents : :obj:`list` of :obj:`np.ndarray`
        latent states
    hmm : :obj:`ssm.HMM`
        arhmm objecct
    include_edges : :obj:`bool`
        include states at start and end of chunk

    Returns
    -------
    :obj:`list`
        number of frames for each state run; list is empty if single-state model

    """
    if hmm.K == 1:
        return []
    states = [hmm.most_likely_states(x) for x in latents if len(x) > 0]
    state_indices = get_discrete_chunks(states, include_edges=include_edges)
    durations = []
    for i_state in range(0, len(state_indices)):
        if len(state_indices[i_state]) > 0:
            durations.append(np.concatenate(np.diff(state_indices[i_state][:, 1:3], 1)))
        else:
            durations.append(np.array([]))
    return durations


def get_latent_arrays_by_dtype(data_generator, sess_idxs=0, data_key='ae_latents'):
    """Collect data from data generator and put into dictionary with dtypes for keys.

    Parameters
    ----------
    data_generator : :obj:`ConcatSessionsGenerator`
    sess_idxs : :obj:`int` or :obj:`list`
        concatenate train/test/val data across one or more sessions
    data_key : :obj:`str`
        key into data generator object; 'ae_latents' | 'labels'

    Returns
    -------
    :obj:`tuple`
        - latents (:obj:`dict`): with keys 'train', 'val', 'test'
        - trial indices (:obj:`dict`): with keys 'train', 'val', 'test'

    """
    if isinstance(sess_idxs, int):
        sess_idxs = [sess_idxs]
    dtypes = ['train', 'val', 'test']
    latents = {key: [] for key in dtypes}
    trial_idxs = {key: [] for key in dtypes}
    for sess_idx in sess_idxs:
        dataset = data_generator.datasets[sess_idx]
        for data_type in dtypes:
            curr_idxs = dataset.batch_idxs[data_type]
            trial_idxs[data_type] += list(curr_idxs)
            latents[data_type] += [dataset[i_trial][data_key][0][:] for i_trial in curr_idxs]
    return latents, trial_idxs


def get_model_latents_states(
        hparams, version, sess_idx=0, return_samples=0, cond_sampling=False, dtype='test',
        dtypes=['train', 'val', 'test'], rng_seed=0):
    """Return arhmm defined in :obj:`hparams` with associated latents and states.

    Can also return sampled latents and states.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an arhmm
    version : :obj:`str` or :obj:`int`
        test tube model version (can be 'best')
    sess_idx : :obj:`int`, optional
        session index into data generator
    return_samples : :obj:`int`, optional
        number of trials to sample from model
    cond_sampling : :obj:`bool`, optional
        if :obj:`True` return samples conditioned on most likely state sequence; else return
        unconditioned samples
    dtype : :obj:`str`, optional
        trial type to use for conditonal sampling; 'train' | 'val' | 'test'
    dtypes : :obj:`array-like`, optional
        trial types for which to collect latents and states
    rng_seed : :obj:`int`, optional
        random number generator seed to control sampling

    Returns
    -------
    :obj:`dict`
        - 'model' (:obj:`ssm.HMM` object)
        - 'latents' (:obj:`dict`): latents from train, val and test trials
        - 'states' (:obj:`dict`): states from train, val and test trials
        - 'trial_idxs' (:obj:`dict`): trial indices from train, val and test trials
        - 'latents_gen' (:obj:`list`)
        - 'states_gen' (:obj:`list`)

    """
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_best_model_version
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

    hparams['session_dir'], sess_ids = get_session_dir(
        hparams, session_source=hparams.get('all_source', 'save'))
    hparams['expt_dir'] = get_expt_dir(hparams)

    # get version/model
    if version == 'best':
        version = get_best_model_version(
            hparams['expt_dir'], measure='val_loss', best_def='min')[0]
    else:
        _, version = experiment_exists(hparams, which_version=True)
    if version is None:
        raise FileNotFoundError(
            'Could not find the specified model version in %s' % hparams['expt_dir'])

    # load model
    model_file = os.path.join(hparams['expt_dir'], 'version_%i' % version, 'best_val_model.pt')
    with open(model_file, 'rb') as f:
        hmm = pickle.load(f)

    # load latents/labels
    if hparams['model_class'].find('labels') > -1:
        from behavenet.data.utils import load_labels_like_latents
        all_latents = load_labels_like_latents(hparams, sess_ids, sess_idx)
    else:
        _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
        with open(latents_file, 'rb') as f:
            all_latents = pickle.load(f)

    # collect inferred latents/states
    trial_idxs = {}
    latents = {}
    states = {}
    for data_type in dtypes:
        trial_idxs[data_type] = all_latents['trials'][data_type]
        latents[data_type] = [all_latents['latents'][i_trial] for i_trial in trial_idxs[data_type]]
        states[data_type] = [hmm.most_likely_states(x) for x in latents[data_type]]

    # collect sampled latents/states
    if return_samples > 0:
        states_gen = []
        np.random.seed(rng_seed)
        if cond_sampling:
            n_latents = latents[dtype][0].shape[1]
            latents_gen = [np.zeros((len(state_seg), n_latents)) for state_seg in states[dtype]]
            for i_seg, state_seg in enumerate(states[dtype]):
                for i_t in range(len(state_seg)):
                    if i_t >= 1:
                        latents_gen[i_seg][i_t] = hmm.observations.sample_x(
                            states[dtype][i_seg][i_t], latents_gen[i_seg][:i_t], input=np.zeros(0))
                    else:
                        latents_gen[i_seg][i_t] = hmm.observations.sample_x(
                            states[dtype][i_seg][i_t],
                            latents[dtype][i_seg][0].reshape((1, n_latents)), input=np.zeros(0))
        else:
            latents_gen = []
            offset = 200
            for i in range(return_samples):
                these_states_gen, these_latents_gen = hmm.sample(
                    latents[dtype][0].shape[0] + offset)
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
    """Present video clips of each individual syllable in separate panels.

    This is a high-level function that loads the arhmm model described in the hparams dictionary
    and produces the necessary states/video frames.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an arhmm
    save_file : :obj:`str`
        full save file (path and filename)
    sess_idx : :obj:`int`, optional
        session index into data generator
    dtype : :obj:`str`, optional
        types of trials to make video with; 'train' | 'val' | 'test'
    max_frames : :obj:`int`, optional
        maximum number of frames to animate
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    min_threshold : :obj:`int`, optional
        minimum number of frames in a syllable run to be considered for movie
    n_buffer : :obj:`int`
        number of blank frames between syllable instances
    n_pre_frames : :obj:`int`
        number of behavioral frames to precede each syllable instance
    n_rows : :obj:`int` or :obj:`NoneType`
        number of rows in output movie
    single_syllable : :obj:`int` or :obj:`NoneType`
        choose only a single state for movie

    """
    from behavenet.data.data_generator import ConcatSessionsGenerator
    from behavenet.data.utils import get_data_generator_inputs
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import experiment_exists
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

    # load images, latents, and states
    hparams['session_dir'], sess_ids = get_session_dir(
        hparams, session_source=hparams.get('all_source', 'save'))
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
    # load latents/labels
    if hparams['model_class'].find('labels') > -1:
        from behavenet.data.utils import load_labels_like_latents
        latents = load_labels_like_latents(hparams, sess_ids, sess_idx)
    else:
        _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
        with open(latents_file, 'rb') as f:
            latents = pickle.load(f)
    trial_idxs = latents['trials'][dtype]
    # load model
    model_file = os.path.join(hparams['expt_dir'], 'version_%i' % version, 'best_val_model.pt')
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
    """Present video clips of each individual syllable in separate panels

    Parameters
    ----------
    ims_orig : :obj:`np.ndarray`
        shape (n_frames, n_channels, y_pix, x_pix)
    state_list : :obj:`list`
        each entry (one per state) contains all occurences of that discrete state by
        :obj:`[chunk number, starting index, ending index]`
    trial_idxs : :obj:`array-like`
        indices into :obj:`states` for which trials should be plotted
    save_file : :obj:`str`
        full save file (path and filename)
    max_frames : :obj:`int`, optional
        maximum number of frames to animate
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    n_buffer : :obj:`int`, optional
        number of blank frames between syllable instances
    n_pre_frames : :obj:`int`, optional
        number of behavioral frames to precede each syllable instance
    n_rows : :obj:`int` or :obj:`NoneType`, optional
        number of rows in output movie
    single_syllable : :obj:`int` or :obj:`NoneType`, optional
        choose only a single state for movie

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
    bs, n_channels, y_dim, x_dim = ims_orig[0].shape
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
        save_movie(save_file, ani, frame_rate=frame_rate)


def real_vs_sampled_wrapper(
        output_type, hparams, save_file, sess_idx, dtype='test', conditional=True, max_frames=400,
        frame_rate=20, n_buffer=5, xtick_locs=None, frame_rate_beh=None, format='png'):
    """Produce movie with (AE) reconstructed video and sampled video.

    This is a high-level function that loads the model described in the hparams dictionary and
    produces the necessary state sequences/samples. The sampled video can be completely
    unconditional (states and latents are sampled) or conditioned on the most likely state
    sequence.

    Parameters
    ----------
    output_type : :obj:`str`
        'plot' | 'movie' | 'both'
    hparams : :obj:`dict`
        needs to contain enough information to specify an autoencoder
    save_file : :obj:`str`
        full save file (path and filename)
    sess_idx : :obj:`int`, optional
        session index into data generator
    dtype : :obj:`str`, optional
        types of trials to make plot/video with; 'train' | 'val' | 'test'
    conditional : :obj:`bool`
        conditional vs unconditional samples; for creating reconstruction title
    max_frames : :obj:`int`, optional
        maximum number of frames to animate
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    n_buffer : :obj:`int`
        number of blank frames between animated trials if more one are needed to reach
        :obj:`max_frames`
    xtick_locs : :obj:`array-like`, optional
        tick locations in bin values for plot
    frame_rate_beh : :obj:`float`, optional
        behavioral video framerate; to properly relabel xticks
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle if :obj:`output_type='plot'` or :obj:`output_type='both'`, else
        nothing returned (movie is saved)

    """
    from behavenet.data.utils import get_transforms_paths
    from behavenet.fitting.utils import get_expt_dir
    from behavenet.fitting.utils import get_session_dir

    # check input - cannot create sampled movies for arhmm-labels models (no mapping from labels to
    # frames)
    if hparams['model_class'].find('labels') > -1:
        if output_type == 'both' or output_type == 'movie':
            print('warning: cannot create video with "arhmm-labels" model; producing plots')
            output_type = 'plot'

    # load latents and states (observed and sampled)
    model_output = get_model_latents_states(
        hparams, '', sess_idx=sess_idx, return_samples=50, cond_sampling=conditional)

    if output_type == 'both' or output_type == 'movie':

        # load in AE decoder
        if hparams.get('ae_model_path', None) is not None:
            ae_model_file = os.path.join(hparams['ae_model_path'], 'best_val_model.pt')
            ae_arch = pickle.load(
                open(os.path.join(hparams['ae_model_path'], 'meta_tags.pkl'), 'rb'))
        else:
            hparams['session_dir'], sess_ids = get_session_dir(
                hparams, session_source=hparams.get('all_source', 'save'))
            hparams['expt_dir'] = get_expt_dir(hparams)
            _, latents_file = get_transforms_paths('ae_latents', hparams, sess_ids[sess_idx])
            ae_model_file = os.path.join(os.path.dirname(latents_file), 'best_val_model.pt')
            ae_arch = pickle.load(
                open(os.path.join(os.path.dirname(latents_file), 'meta_tags.pkl'), 'rb'))
        print('loading model from %s' % ae_model_file)
        ae_model = AE(ae_arch)
        ae_model.load_state_dict(
            torch.load(ae_model_file, map_location=lambda storage, loc: storage))
        ae_model.eval()

        n_channels = ae_model.hparams['n_input_channels']
        y_pix = ae_model.hparams['y_pixels']
        x_pix = ae_model.hparams['x_pixels']

        # push observed latents through ae decoder
        ims_recon = np.zeros((0, n_channels * y_pix, x_pix))
        i_trial = 0
        while ims_recon.shape[0] < max_frames:
            recon = ae_model.decoding(
                torch.tensor(model_output['latents'][dtype][i_trial]).float(), None, None). \
                cpu().detach().numpy()
            recon = np.concatenate([recon[:, i] for i in range(recon.shape[1])], axis=1)
            zero_frames = np.zeros((n_buffer, n_channels * y_pix, x_pix))  # add a few black frames
            ims_recon = np.concatenate((ims_recon, recon, zero_frames), axis=0)
            i_trial += 1

        # push sampled latents through ae decoder
        ims_recon_samp = np.zeros((0, n_channels * y_pix, x_pix))
        i_trial = 0
        while ims_recon_samp.shape[0] < max_frames:
            recon = ae_model.decoding(torch.tensor(
                model_output['latents_gen'][i_trial]).float(), None, None).cpu().detach().numpy()
            recon = np.concatenate([recon[:, i] for i in range(recon.shape[1])], axis=1)
            zero_frames = np.zeros((n_buffer, n_channels * y_pix, x_pix))  # add a few black frames
            ims_recon_samp = np.concatenate((ims_recon_samp, recon, zero_frames), axis=0)
            i_trial += 1

        make_real_vs_sampled_movies(
            ims_recon, ims_recon_samp, conditional=conditional, save_file=save_file,
            frame_rate=frame_rate)

    if output_type == 'both' or output_type == 'plot':

        i_trial = 0
        latents = model_output['latents'][dtype][i_trial][:max_frames]
        states = model_output['states'][dtype][i_trial][:max_frames]
        latents_samp = model_output['latents_gen'][i_trial][:max_frames]
        if not conditional:
            states_samp = model_output['states_gen'][i_trial][:max_frames]
        else:
            states_samp = []

        fig = plot_real_vs_sampled(
            latents, latents_samp, states, states_samp, save_file=save_file, xtick_locs=xtick_locs,
            frame_rate=hparams['frame_rate'] if frame_rate_beh is None else frame_rate_beh,
            format=format)

    if output_type == 'movie':
        return None
    elif output_type == 'both' or output_type == 'plot':
        return fig
    else:
        raise ValueError('"%s" is an invalid output_type' % output_type)


def make_real_vs_sampled_movies(
        ims_recon, ims_recon_samp, conditional, save_file=None, frame_rate=15):
    """Produce movie with (AE) reconstructed video and sampled video.

    Parameters
    ----------
    ims_recon : :obj:`np.ndarray`
        shape (n_frames, y_pix, x_pix)
    ims_recon_samp : :obj:`np.ndarray`
        shape (n_frames, y_pix, x_pix)
    conditional : :obj:`bool`
        conditional vs unconditional samples; for creating reconstruction title
    save_file : :obj:`str`, optional
        full save file (path and filename)
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    n_frames = ims_recon.shape[0]

    n_plots = 2
    [y_pix, x_pix] = ims_recon[0].shape
    fig_dim_div = x_pix * n_plots / 10  # aiming for dim 1 being 10
    x_dim = x_pix * n_plots / fig_dim_div
    y_dim = y_pix / fig_dim_div
    fig, axes = plt.subplots(1, n_plots, figsize=(x_dim, y_dim))

    for j in range(2):
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    axes[0].set_title('Real Reconstructions\n', fontsize=16)
    if conditional:
        title_str = 'Generative Reconstructions\n(Conditional)'
    else:
        title_str = 'Generative Reconstructions\n(Unconditional)'
    axes[1].set_title(title_str, fontsize=16)
    fig.tight_layout(pad=0)

    im_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'animated': True}
    ims = []
    for i in range(n_frames):
        ims_curr = []
        im = axes[0].imshow(ims_recon[i], **im_kwargs)
        ims_curr.append(im)
        im = axes[1].imshow(ims_recon_samp[i], **im_kwargs)
        ims_curr.append(im)
        ims.append(ims_curr)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


def plot_real_vs_sampled(
        latents, latents_samp, states, states_samp, save_file=None, xtick_locs=None,
        frame_rate=None, format='png'):
    """Plot real and sampled latents overlaying real and (potentially sampled) states.

    Parameters
    ----------
    latents : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    latents_samp : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    states : :obj:`np.ndarray`
        shape (n_frames,)
    states_samp : :obj:`np.ndarray`
        shape (n_frames,) if :obj:`latents_samp` are not conditioned on :obj:`states`, otherwise
        shape (0,)
    save_file : :obj:`str`
        full save file (path and filename)
    xtick_locs : :obj:`array-like`, optional
        tick locations in bin values for plot
    frame_rate : :obj:`float`, optional
        behavioral video framerate; to properly relabel xticks
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # plot observations
    axes[0] = plot_states_overlaid_with_latents(
        latents, states, ax=axes[0], xtick_locs=xtick_locs, frame_rate=frame_rate)
    axes[0].set_xticks([])
    axes[0].set_xlabel('')
    axes[0].set_title('Inferred latents')

    # plot samples
    if len(states_samp) == 0:
        plot_states = states
        title_str = 'Sampled latents'
    else:
        plot_states = states_samp
        title_str = 'Sampled states and latents'
    axes[1] = plot_states_overlaid_with_latents(
        latents_samp, plot_states, ax=axes[1], xtick_locs=xtick_locs, frame_rate=frame_rate)
    axes[1].set_title(title_str)

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    return fig


def plot_states_overlaid_with_latents(
        latents, states, save_file=None, ax=None, xtick_locs=None, frame_rate=None, cmap='tab20b',
        format='png'):
    """Plot states for a single trial overlaid with latents.

    Parameters
    ----------
    latents : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    states : :obj:`np.ndarray`
        shape (n_frames,)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    ax : :obj:`matplotlib.Axes` object or :obj:`NoneType`, optional
        axes to plot into; if :obj:`NoneType`, a new figure is created
    xtick_locs : :obj:`array-like`, optional
        tick locations in bin values for plot
    frame_rate : :obj:`float`, optional
        behavioral video framerate; to properly relabel xticks
    cmap : :obj:`str`, optional
        matplotlib colormap
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle if :obj:`ax=None`, otherwise updated axis

    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.gca()
    else:
        fig = None
    spc = 1.1 * abs(latents.max())
    n_latents = latents.shape[1]
    plotting_latents = latents + spc * np.arange(n_latents)
    ymin = min(-spc, np.min(plotting_latents))
    ymax = max(spc * n_latents, np.max(plotting_latents))
    ax.imshow(
        states[None, :], aspect='auto', extent=(0, len(latents), ymin, ymax), cmap=cmap,
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
    """Plot Markov transition matrix for arhmm.

    Parameters
    ----------
    model : :obj:`ssm.HMM` object
    deridge : :obj:`bool`, optional
        remove diagonal to more clearly see dynamic range of off-diagonal entries

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

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
    """Plot autoregressive dynamics matrices for arhmm.

    Parameters
    ----------
    model : :obj:`ssm.HMM` object
    deridge : :obj:`bool`, optional
        remove diagonal to more clearly see dynamic range of off-diagonal entries

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

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
        for lag in range(n_lags - 1):
            plt.axvline((lag + 1) * D - 0.5, ymin=0, ymax=K, color=[0, 0, 0])
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
    """Plot observation bias vectors for arhmm.

    Parameters
    ----------
    model : :obj:`ssm.HMM` object

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

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
    """Plot observation covariance matrices for arhmm.

    Parameters
    ----------
    model : :obj:`ssm.HMM` object

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

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
