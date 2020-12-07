"""Plotting and video making functions for autoencoders."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from behavenet.fitting.eval import get_reconstruction
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.plotting import concat, save_movie

# to ignore imports for sphix-autoapidoc
__all__ = ['make_ae_reconstruction_movie_wrapper', 'make_reconstruction_movie']


def make_reconstruction_movie(
        ims, titles=None, n_rows=0, n_cols=0, save_file=None, frame_rate=15, dpi=100):
    """Produce movie with original video and reconstructed videos.

    `ims` and `titles` are corresponding lists; this data is plotted using a linear index, i.e. if
    n_rows = 2 and n_cols = 3 the image stack in ims[2] will be in the first row, second column;
    the image stack in ims[4] will be in the second row, first column. If ims[i] is empty, that
    grid location will be skipped.

    Parameters
    ----------
    ims : :obj:`list` of :obj:`np.ndarray`
        each list element is of shape (n_frames, n_channels, y_pix, x_pix)
    titles : :obj:`list` of :obj:`str`, optional
        title for each panel
    n_rows : :obj:`int`
        number of rows in video grid layout
    n_cols : :obj:`int`
        number of columns in video grid layout
    save_file : :obj:`str`, optional
        full save file (path and filename)
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    dpi : :obj:`int`, optional
        dpi of movie figure; controls resolution of titles

    """

    for im in ims:
        if len(im) != 0:
            n_frames, n_channels, y_pix, x_pix = im.shape
            break
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    offset = 0.5 if n_rows == 1 else 0
    fig = plt.figure(figsize=(fig_width, fig_height + offset), dpi=dpi)

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    ax_count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if ax_count < len(ims):
                axs.append(fig.add_subplot(gs[i, j]))
                ax_count += 1
            else:
                break
    for ax_i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        if len(ims[ax_i]) == 0:
            ax.set_axis_off()

    fontsize = 12
    titles = ['' for _ in range(n_cols * n_rows)] if titles is None else titles
    for ax_i, ax in enumerate(axs):
        if len(ims[ax_i]) != 0:
            ax.set_title(titles[ax_i], fontsize=fontsize)

    default_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i in range(n_frames):

        ims_curr = []

        for ax_i, ax in enumerate(axs):
            if len(ims[ax_i]) != 0:
                ims_tmp = ims[ax_i][i, 0] if n_channels == 1 else concat(ims[ax_i][i])
                im = ax.imshow(ims_tmp, **default_kwargs)
                [s.set_visible(False) for s in ax.spines.values()]
                ims_curr.append(im)

        ims_ani.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims_ani, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


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
        model_lin, _ = get_best_model_and_data(
            hparams_lin, Model, load_data=False, version=version)
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

    ims_orig = ims_orig_pt.cpu().detach().numpy()
    ims = [ims_orig, ims_recon_ae, 0.5 + (ims_orig - ims_recon_ae)]
    titles = ['Original', 'Conv AE reconstructed', 'Conv AE residual']
    if include_linear:
        ims.append([])
        ims.append(ims_recon_lin)
        ims.append(0.5 + (ims_orig - ims_recon_lin))
        titles.append('')
        titles.append('Linear AE reconstructed')
        titles.append('Linear AE residual')
        n_rows = 2
        n_cols = 3
    else:
        n_rows = 1
        n_cols = 3

    make_reconstruction_movie(
        ims=ims, titles=titles, n_rows=n_rows, n_cols=n_cols, save_file=save_file,
        frame_rate=frame_rate)
