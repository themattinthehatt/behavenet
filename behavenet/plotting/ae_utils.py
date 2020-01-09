"""Plotting and video making functions for autoencoders."""

import copy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
import numpy as np
from behavenet.plotting import concat
from behavenet import make_dir_if_not_exists
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.fitting.eval import get_reconstruction


def make_ae_reconstruction_movie_wrapper(
        hparams, save_file, trial=None, sess_idx=0, version='best', include_linear=False,
        max_frames=400, frame_rate=15):
    """Produce movie with original video, reconstructed video, and residual.

    This is a high-level function that loads the model described in the hparams dictionary and
    produces the necessary predicted video frames.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an autoencoder
    save_file : :obj:`str`
        full save file (path and filename)
    trial : :obj:`int`, optional
        if :obj:`NoneType`, use first test trial
    sess_idx : :obj:`int`, optional
        session index into data generator
    version : :obj:`str` or :obj:`int`, optional
        test tube model version
    include_linear : :obj:`bool`, optional
        include reconstruction from corresponding linear ae (i.e. ame number of latents)
    max_frames : :obj:`int`, optional
        maximum number of frames to animate from a trial
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    # build model(s)
    if hparams['model_class'] == 'ae':
        from behavenet.models import AE as Model
    elif hparams['model_class'] == 'cond-ae':
        from behavenet.models import ConditionalAE as Model
    else:
        raise NotImplementedError('"%s" is an invalid model class' % hparams['model_class'])
    model_ae, data_generator = get_best_model_and_data(hparams, Model, version=version)

    if include_linear:
        import copy
        hparams_lin = copy.copy(hparams)
        hparams_lin['model_type'] = 'linear'
        if 'lin_experiment_name' in hparams:
            hparams_lin['experiment_name'] = hparams['lin_experiment_name']
        model_lin, _ = get_best_model_and_data(hparams_lin, Model, load_data=False)
    else:
        model_lin = None

    # push images through decoder
    if trial is None:
        # choose first test trial
        trial = data_generator.batch_idxs[sess_idx]['test'][0]
    batch = data_generator.datasets[sess_idx][trial]
    ims_orig_pt = batch['images'][:max_frames]
    if hparams['model_class'] == 'cond-ae':
        labels_pt = batch['labels'][:max_frames]
    else:
        labels_pt = None

    ims_recon_ae = get_reconstruction(model_ae, ims_orig_pt, labels=labels_pt)
    if include_linear:
        ims_recon_lin = get_reconstruction(model_lin, ims_orig_pt, labels=labels_pt)
    else:
        ims_recon_lin = None

    # mask images for plotting
    if hparams.get('use_output_mask', False):
        ims_orig_pt *= batch['masks'][:max_frames]

    make_ae_reconstruction_movie(
        ims_orig=ims_orig_pt.cpu().detach().numpy(),
        ims_recon_ae=ims_recon_ae,
        ims_recon_lin=ims_recon_lin,
        save_file=save_file,
        frame_rate=frame_rate)


def make_ae_reconstruction_movie(
        ims_orig, ims_recon_ae, ims_recon_lin=None, save_file=None, frame_rate=15):
    """Produce movie with original video, reconstructed video, and residual.

    Parameters
    ----------
    ims_orig : :obj:`np.ndarray`
        shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_ae : :obj:`np.ndarray`
        shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_lin : :obj:`np.ndarray`, optional
        shape (n_frames, n_channels, y_pix, x_pix)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    n_frames, n_channels, y_pix, x_pix = ims_orig.shape
    n_cols = 1 if ims_recon_lin is None else 2
    n_rows = 3
    offset = 1  # 0 if ims_recon_lin is None else 1
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    fig = plt.figure(figsize=(fig_width, fig_height + offset),dpi=100)

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    if ims_recon_lin is not None:
        axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
        axs.append(fig.add_subplot(gs[1, 0]))  # 1: ae reconstructed frames
        axs.append(fig.add_subplot(gs[1, 1]))  # 2: ae residuals
        axs.append(fig.add_subplot(gs[2, 0]))  # 3: linear reconstructed frames
        axs.append(fig.add_subplot(gs[2, 1]))  # 4: linear residuals
    else:
        axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
        axs.append(fig.add_subplot(gs[1, 0]))  # 1: ae reconstructed frames
        axs.append(fig.add_subplot(gs[2, 0]))  # 2: ae residuals
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 12
    axs[0].set_title('Original', fontsize=fontsize)
    axs[1].set_title('Conv AE reconstructed', fontsize=fontsize)
    axs[2].set_title('Conv AE residual', fontsize=fontsize)
    if ims_recon_lin is not None:
        axs[3].set_title('Linear AE reconstructed', fontsize=fontsize)
        axs[4].set_title('Linear AE residual', fontsize=fontsize)

    ims_res_ae = ims_orig - ims_recon_ae
    if ims_recon_lin is not None:
        ims_res_lin = ims_orig - ims_recon_lin
    else:
        ims_res_lin = None

    default_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims = []
    for i in range(ims_orig.shape[0]):

        ims_curr = []

        # original video
        ims_tmp = ims_orig[i, 0] if n_channels == 1 else concat(ims_orig[i])
        im = axs[0].imshow(ims_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[0].spines.values()]
        ims_curr.append(im)

        # ae reconstructed video
        ims_tmp = ims_recon_ae[i, 0] if n_channels == 1 else concat(ims_recon_ae[i])
        im = axs[1].imshow(ims_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[1].spines.values()]
        ims_curr.append(im)

        # ae residual video
        ims_tmp = ims_res_ae[i, 0] if n_channels == 1 else concat(ims_res_ae[i])
        im = axs[2].imshow(0.5 + ims_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[2].spines.values()]
        ims_curr.append(im)

        if ims_recon_lin is not None:

            # linear reconstructed video
            ims_tmp = ims_recon_lin[i, 0] if n_channels == 1 else concat(ims_recon_lin[i])
            im = axs[3].imshow(ims_tmp, **default_kwargs)
            [s.set_visible(False) for s in axs[3].spines.values()]
            ims_curr.append(im)

            # linear residual video
            ims_tmp = ims_res_lin[i, 0] if n_channels == 1 else concat(ims_res_lin[i])
            im = axs[4].imshow(0.5 + ims_tmp, **default_kwargs)
            [s.set_visible(False) for s in axs[4].spines.values()]
            ims_curr.append(im)

        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=frame_rate, bitrate=-1)

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        print('saving video to %s...' % save_file, end='')
        ani.save(save_file, writer=writer)
        # if save_file[-3:] != 'gif':
        #     save_file += '.gif'
        # ani.save(save_file, writer='imagemagick', fps=15)
        print('done')


def make_neural_reconstruction_movie_wrapper(
        hparams, save_file, trial=None, sess_idx=0, max_frames=400, max_latents=8, frame_rate=15):
    """Produce movie with original video, ae reconstructed video, and neural reconstructed video.

    This is a high-level function that loads the model described in the hparams dictionary and
    produces the necessary predicted video frames. Latent traces are additionally plotted, as well
    as the residual between the ae reconstruction and the neural reconstruction. Currently produces
    ae latents and decoder predictions from scratch (rather than saved pickle files).

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an autoencoder
    save_file : :obj:`str`
        full save file (path and filename)
    trial : :obj:`int`, optional
        if :obj:`NoneType`, use first test trial
    sess_idx : :obj:`int`, optional
        session index into data generator
    max_frames : :obj:`int`, optional
        maximum number of frames to animate from a trial
    max_latents : :obj:`int`, optional
        maximum number of ae latents to plot
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    from behavenet.models import Decoder

    ###############################
    # build ae model/data generator
    ###############################
    hparams_ae = copy.copy(hparams)
    hparams_ae['experiment_name'] = hparams['ae_experiment_name']
    hparams_ae['model_class'] = hparams['ae_model_class']
    hparams_ae['model_type'] = hparams['ae_model_type']
    if hparams['model_class'] == 'ae':
        from behavenet.models import AE as Model
    elif hparams['model_class'] == 'cond-ae':
        from behavenet.models import ConditionalAE as Model
    else:
        raise NotImplementedError('"%s" is an invalid model class' % hparams['model_class'])
    model_ae, data_generator_ae = get_best_model_and_data(
        hparams_ae, Model, version=hparams['ae_version'])
    # move model to cpu
    model_ae.to('cpu')

    if trial is None:
        # choose first test trial
        trial = data_generator_ae.batch_idxs[sess_idx]['test'][0]

    # get images from data generator (move to cpu)
    batch = data_generator_ae.datasets[sess_idx][trial]
    ims_orig_pt = batch['images'][:max_frames].cpu()  # 400
    if hparams['model_class'] == 'cond-ae':
        labels_pt = batch['labels'][:max_frames]
    else:
        labels_pt = None

    # push images through ae to get reconstruction
    ims_recon_ae = get_reconstruction(model_ae, ims_orig_pt, labels=labels_pt)
    # push images through ae to get latents
    latents_ae_pt, _, _ = model_ae.encoding(ims_orig_pt)

    # mask images for plotting
    if hparams.get('use_output_mask', False):
        ims_orig_pt *= batch['masks'][:max_frames]

    #######################################
    # build decoder model/no data generator
    #######################################
    hparams_dec = copy.copy(hparams)
    hparams_dec['experiment_name'] = hparams['decoder_experiment_name']
    hparams_dec['model_class'] = hparams['decoder_model_class']
    hparams_dec['model_type'] = hparams['decoder_model_type']

    model_dec, data_generator_dec = get_best_model_and_data(
        hparams_dec, Decoder, version=hparams['decoder_version'])
    # move model to cpu
    model_dec.to('cpu')

    # get neural activity from data generator (move to cpu)
    batch = data_generator_dec.datasets[0][trial]  # 0 not sess_idx since decoders only have 1 sess
    neural_activity_pt = batch['neural'][:max_frames].cpu()

    # push neural activity through decoder to get prediction
    latents_dec_pt, _ = model_dec(neural_activity_pt)
    # push prediction through ae to get reconstruction
    ims_recon_dec = get_reconstruction(model_ae, latents_dec_pt, labels=labels_pt)

    # away
    make_neural_reconstruction_movie(
        ims_orig=ims_orig_pt.cpu().detach().numpy(),
        ims_recon_ae=ims_recon_ae,
        ims_recon_neural=ims_recon_dec,
        latents_ae=latents_ae_pt.cpu().detach().numpy()[:, :max_latents],
        latents_neural=latents_dec_pt.cpu().detach().numpy()[:, :max_latents],
        save_file=save_file,
        frame_rate=frame_rate)


def make_neural_reconstruction_movie(
        ims_orig, ims_recon_ae, ims_recon_neural, latents_ae, latents_neural, save_file=None,
        frame_rate=15):
    """Produce movie with original video, ae reconstructed video, and neural reconstructed video.

    Latent traces are additionally plotted, as well as the residual between the ae reconstruction
    and the neural reconstruction.

    Parameters
    ----------
    ims_orig : :obj:`np.ndarray`
        shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_ae : :obj:`np.ndarray`
        shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_neural : :obj:`np.ndarray`, optional
        shape (n_frames, n_channels, y_pix, x_pix)
    latents_ae : :obj:`np.ndarray`, optional
        shape (n_frames, n_latents)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    means = np.mean(latents_ae, axis=0)
    std = np.std(latents_ae) * 2

    latents_ae_sc = (latents_ae - means) / std
    latents_dec_sc = (latents_neural - means) / std

    n_channels, y_pix, x_pix = ims_orig.shape[1:]
    n_time, n_ae_latents = latents_ae.shape

    n_cols = 3
    n_rows = 2
    offset = 2  # 0 if ims_recon_lin is None else 1
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    fig = plt.figure(figsize=(fig_width, fig_height + offset))

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))    # 0: original frames
    axs.append(fig.add_subplot(gs[0, 1]))    # 1: ae reconstructed frames
    axs.append(fig.add_subplot(gs[0, 2]))    # 2: neural reconstructed frames
    axs.append(fig.add_subplot(gs[1, 0]))    # 3: residual
    axs.append(fig.add_subplot(gs[1, 1:3]))  # 4: ae and predicted ae latents
    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        if i > 2:
            ax.get_xaxis().set_tick_params(labelsize=12, direction='in')
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([])
    axs[3].set_xticks([])

    # check that the axes are correct
    fontsize = 12
    idx = 0
    axs[idx].set_title('Original', fontsize=fontsize); idx += 1
    axs[idx].set_title('AE reconstructed', fontsize=fontsize); idx += 1
    axs[idx].set_title('Neural reconstructed', fontsize=fontsize); idx += 1
    axs[idx].set_title('Reconstructions residual', fontsize=fontsize); idx += 1
    axs[idx].set_title('AE latent predictions', fontsize=fontsize)
    axs[idx].set_xlabel('Time (bins)', fontsize=fontsize)

    time = np.arange(n_time)

    ims_res = ims_recon_ae - ims_recon_neural

    im_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    tr_kwargs = {'animated': True, 'linewidth': 2}
    latents_ae_color = [0.2, 0.2, 0.2]
    latents_dec_color = [0, 0, 0]

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(n_time):

        ims_curr = []
        idx = 0

        if i % 100 == 0:
            print('processing frame %03i/%03i' % (i, n_time))

        ###################
        # behavioral videos
        ###################
        # original video
        ims_tmp = ims_orig[i, 0] if n_channels == 1 else concat(ims_orig[i])
        im = axs[idx].imshow(ims_tmp, **im_kwargs)
        ims_curr.append(im)
        idx += 1

        # ae reconstruction
        ims_tmp = ims_recon_ae[i, 0] if n_channels == 1 else concat(ims_recon_ae[i])
        im = axs[idx].imshow(ims_tmp, **im_kwargs)
        ims_curr.append(im)
        idx += 1

        # neural reconstruction
        ims_tmp = ims_recon_neural[i, 0] if n_channels == 1 else concat(ims_recon_neural[i])
        im = axs[idx].imshow(ims_tmp, **im_kwargs)
        ims_curr.append(im)
        idx += 1

        # residual
        ims_tmp = ims_res[i, 0] if n_channels == 1 else concat(ims_res[i])
        im = axs[idx].imshow(0.5 + ims_tmp, **im_kwargs)
        ims_curr.append(im)
        idx += 1

        ########
        # traces
        ########
        # latents over time
        for latent in range(n_ae_latents):
            # just put labels on last lvs
            if latent == n_ae_latents - 1 and i == 0:
                label_ae = 'AE latents'
                label_dec = 'Predicted AE latents'
            else:
                label_ae = None
                label_dec = None
            im = axs[idx].plot(
                time[0:i + 1], latent + latents_ae_sc[0:i + 1, latent],
                color=latents_ae_color, alpha=0.7, label=label_ae,
                **tr_kwargs)[0]
            axs[idx].spines['top'].set_visible(False)
            axs[idx].spines['right'].set_visible(False)
            axs[idx].spines['left'].set_visible(False)
            ims_curr.append(im)
            im = axs[idx].plot(
                time[0:i + 1], latent + latents_dec_sc[0:i + 1, latent],
                color=latents_dec_color, label=label_dec, **tr_kwargs)[0]
            axs[idx].spines['top'].set_visible(False)
            axs[idx].spines['right'].set_visible(False)
            axs[idx].spines['left'].set_visible(False)
            plt.legend(
                loc='lower right', fontsize=fontsize, frameon=True,
                framealpha=0.7, edgecolor=[1, 1, 1])
            ims_curr.append(im)
        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=frame_rate, bitrate=-1)

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        print('saving video to %s...' % save_file, end='')
        ani.save(save_file, writer=writer)
        print('done')


def plot_neural_reconstruction_traces_wrapper(
        hparams, save_file=None, trial=None, xtick_locs=None, frame_rate=None, format='png'):
    """Plot ae latents and their neural reconstructions.

    This is a high-level function that loads the model described in the hparams dictionary and
    produces the necessary predicted latents.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an ae latent decoder
    save_file : :obj:`str`
        full save file (path and filename)
    trial : :obj:`int`, optional
        if :obj:`NoneType`, use first test trial
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate : :obj:`float`, optional
        frame rate of behavorial video; to properly relabel xticks
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle of plot

    """

    # find good trials
    import copy
    from behavenet.data.utils import get_transforms_paths
    from behavenet.data.data_generator import ConcatSessionsGenerator

    # ae data
    hparams_ae = copy.copy(hparams)
    hparams_ae['experiment_name'] = hparams['ae_experiment_name']
    hparams_ae['model_class'] = hparams['ae_model_class']
    hparams_ae['model_type'] = hparams['ae_model_type']

    ae_transform, ae_path = get_transforms_paths('ae_latents', hparams_ae, None)

    # ae predictions data
    hparams_dec = copy.copy(hparams)
    hparams_dec['neural_ae_experiment_name'] = hparams['decoder_experiment_name']
    hparams_dec['neural_ae_model_class'] = hparams['decoder_model_class']
    hparams_dec['neural_ae_model_type'] = hparams['decoder_model_type']
    ae_pred_transform, ae_pred_path = get_transforms_paths(
        'neural_ae_predictions', hparams_dec, None)

    signals = ['ae_latents', 'ae_predictions']
    transforms = [ae_transform, ae_pred_transform]
    paths = [ae_path, ae_pred_path]

    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], [hparams],
        signals_list=[signals], transforms_list=[transforms], paths_list=[paths],
        device='cpu', as_numpy=False, batch_load=False, rng_seed=0)

    if trial is None:
        # choose first test trial
        trial = data_generator.datasets[0].batch_idxs['test'][0]

    batch = data_generator.datasets[0][trial]
    traces_ae = batch['ae_latents'].cpu().detach().numpy()
    traces_neural = batch['ae_predictions'].cpu().detach().numpy()

    fig = plot_neural_reconstruction_traces(
        traces_ae, traces_neural, save_file, xtick_locs, frame_rate, format)

    return fig


def plot_neural_reconstruction_traces(
        traces_ae, traces_neural, save_file=None, xtick_locs=None, frame_rate=None, format='png'):
    """Plot ae latents and their neural reconstructions.

    Parameters
    ----------
    traces_ae : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    traces_neural : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate : :obj:`float`, optional
        frame rate of behavorial video; to properly relabel xticks
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

    """

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import seaborn as sns

    sns.set_style('white')
    sns.set_context('poster')

    means = np.mean(traces_ae, axis=0)
    std = np.std(traces_ae) * 2  # scale for better visualization

    traces_ae_sc = (traces_ae - means) / std
    traces_neural_sc = (traces_neural - means) / std

    traces_ae_sc = traces_ae_sc[:, :8]
    traces_neural_sc = traces_neural_sc[:, :8]

    fig = plt.figure(figsize=(12, 8))
    plt.plot(traces_neural_sc + np.arange(traces_neural_sc.shape[1]), linewidth=3)
    plt.plot(
        traces_ae_sc + np.arange(traces_ae_sc.shape[1]), color=[0.2, 0.2, 0.2], linewidth=3,
        alpha=0.7)

    # add legend
    # original latents - gray
    orig_line = mlines.Line2D([], [], color=[0.2, 0.2, 0.2], linewidth=3, alpha=0.7)
    # predicted latents - cycle through some colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dls = []
    for c in range(5):
        dls.append(mlines.Line2D(
            [], [], linewidth=3, linestyle='--', dashes=(0, 3 * c, 20, 1), color='%s' % colors[c]))
    plt.legend(
        [orig_line, tuple(dls)], ['Original latents', 'Predicted latents'],
        loc='lower right', frameon=True, framealpha=0.7, edgecolor=[1, 1, 1])

    if xtick_locs is not None and frame_rate is not None:
        plt.xticks(xtick_locs, (np.asarray(xtick_locs) / frame_rate).astype('int'))
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Time (bins)')
    plt.ylabel('Latent state')
    plt.yticks([])

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    plt.show()
    return fig
