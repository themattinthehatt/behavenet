"""Plotting functions for decoders."""

import copy
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
import pickle
from behavenet import make_dir_if_not_exists
from behavenet.fitting.eval import get_reconstruction
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.data.utils import get_region_list
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_session_dir
from behavenet.fitting.utils import get_subdirs
from behavenet.plotting import concat, save_movie

# to ignore imports for sphix-autoapidoc
__all__ = [
    'get_r2s_by_trial', 'get_best_models', 'get_r2s_across_trials',
    'make_neural_reconstruction_movie_wrapper', 'make_neural_reconstruction_movie',
    'plot_neural_reconstruction_traces_wrapper', 'plot_neural_reconstruction_traces']


def _get_dataset_str(hparams):
    return os.path.join(hparams['expt'], hparams['animal'], hparams['session'])


def get_r2s_by_trial(hparams, model_types):
    """For a given session, load R^2 metrics from all decoders defined by hparams.

    Parameters
    ----------

    hparams : :obj:`dict`
        needs to contain enough information to specify decoders
    model_types : :obj:`list` of :obj:`strs`
        'mlp' | 'mlp-mv' | 'lstm'

    Returns
    -------
    :obj:`pd.DataFrame`
        pandas dataframe of decoder validation metrics

    """

    dataset = _get_dataset_str(hparams)
    region_names = get_region_list(hparams)

    metrics = []
    model_idx = 0
    model_counter = 0
    for region in region_names:
        hparams['region'] = region
        for model_type in model_types:

            hparams['session_dir'], _ = get_session_dir(
                hparams, session_source=hparams.get('all_source', 'save'))
            expt_dir = get_expt_dir(
                hparams,
                model_type=model_type,
                model_class=hparams['model_class'],
                expt_name=hparams['experiment_name'])

            # gather all versions
            try:
                versions = get_subdirs(expt_dir)
            except Exception:
                print('No models in %s; skipping' % expt_dir)

            # load csv files with model metrics (saved out from test tube)
            for i, version in enumerate(versions):
                # read metrics csv file
                model_dir = os.path.join(expt_dir, version)
                try:
                    metric = pd.read_csv(os.path.join(model_dir, 'metrics.csv'))
                    model_counter += 1
                except FileNotFoundError:
                    continue
                with open(os.path.join(model_dir, 'meta_tags.pkl'), 'rb') as f:
                    hparams = pickle.load(f)
                # append model info to metrics ()
                version_num = version[8:]
                metric['version'] = str('version_%i' % model_idx + version_num)
                metric['region'] = region
                metric['dataset'] = dataset
                metric['model_type'] = model_type
                for key, val in hparams.items():
                    if isinstance(val, (str, int, float)):
                        metric[key] = val
                metrics.append(metric)

            model_idx += 10000  # assumes no more than 10k model versions/expt
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    return metrics_df


def get_best_models(metrics_df):
    """Find best decoder over l2 regularization and learning rate.

    Returns a dataframe with test R^2s for each batch, for the best decoder in each category
    (defined by dataset, region, n_lags, and n_hid_layers).

    Parameters
    ----------
    metrics_df : :obj:`pd.DataFrame`
        output of :func:`get_r2s_by_trial`

    Returns
    -------
    :obj:`pd.DataFrame`
        test R^2s for each batch

    """
    # for each version, only keep rows where test_loss is not nan
    data_queried = metrics_df[pd.notna(metrics_df.test_loss)]
    best_models_list = []
    # take min over val losses
    loss_mins = metrics_df.groupby(
        ['dataset', 'n_lags', 'n_hid_layers', 'learning_rate', 'l2_reg', 'version', 'region']) \
        .min().reset_index()
    datasets = metrics_df.dataset.unique()
    datasets.sort()
    regions = metrics_df.region.unique()
    regions.sort()
    n_lags = metrics_df.n_lags.unique()
    n_lags.sort()
    n_hid_layers = metrics_df.n_hid_layers.unique()
    n_hid_layers.sort()
    for dataset in datasets:
        for region in regions:
            for lag in n_lags:
                for layer in n_hid_layers:
                    # get all models with this number of lags
                    single_hp = loss_mins[
                        (loss_mins.n_lags == lag)
                        & (loss_mins.n_hid_layers == layer)
                        & (loss_mins.region == region)
                        & (loss_mins.dataset == dataset)]
                    # find best version from these models
                    best_version = loss_mins.iloc[
                        single_hp.val_loss.idxmin()].version
                    # index back into original data to grab test loss on all
                    # batches
                    best_models_list.append(
                        data_queried[data_queried.version == best_version])
    return pd.concat(best_models_list)


def get_r2s_across_trials(hparams, best_models_df):
    """Calculate R^2 across all test trials (rather than on a trial-by-trial basis)

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain the keys 'lab', 'experiment', 'animal', 'session', 'model_type', 'region',
        'n_hid_layers', 'n_lags'
    best_models_df : :obj:`pd.DataFrame`
        output of :func:`get_best_models`

    Returns
    -------
    :obj:`pd.DataFrame`
        test R^2 across all trials
    """

    from behavenet.fitting.eval import get_test_metric

    dataset = _get_dataset_str(hparams)
    versions = best_models_df.version.unique()

    all_test_r2s = []
    for version in versions:
        model_version = str(int(version[8:]) % 10000)
        hparams['model_type'] = best_models_df[
            best_models_df.version == version].model_type.unique()[0]
        hparams['region'] = best_models_df[
            best_models_df.version == version].region.unique()[0]
        hparams_, r2 = get_test_metric(hparams, model_version)
        all_test_r2s.append(pd.DataFrame({
            'dataset': dataset,
            'region': hparams['region'],
            'n_hid_layers': hparams_['n_hid_layers'],
            'n_lags': hparams_['n_lags'],
            'model_type': hparams['model_type'],
            'r2': r2}, index=[0]))
    return pd.concat(all_test_r2s)


def make_neural_reconstruction_movie_wrapper(
        hparams, save_file, trials=None, sess_idx=0, max_frames=400, max_latents=8,
        zscore_by_dim=False, colored_predictions=False, xtick_locs=None, frame_rate=15):
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
    trials : :obj:`int` or :obj:`list`, optional
        if :obj:`NoneType`, use first test trial
    sess_idx : :obj:`int`, optional
        session index into data generator
    max_frames : :obj:`int`, optional
        maximum number of frames to animate from a trial
    max_latents : :obj:`int`, optional
        maximum number of ae latents to plot
    zscore_by_dim : :obj:`bool`, optional
        True to z-score each dim, False to leave relative scales
    colored_predictions : :obj:`bool`, optional
        False to plot reconstructions in black, True to plot in different colors
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    from behavenet.models import Decoder

    # define number of frames that separate trials
    n_buffer = 5

    ###############################
    # build ae model/data generator
    ###############################
    hparams_ae = copy.copy(hparams)
    hparams_ae['experiment_name'] = hparams['ae_experiment_name']
    hparams_ae['model_class'] = hparams['ae_model_class']
    hparams_ae['model_type'] = hparams['ae_model_type']
    model_ae, data_generator_ae = get_best_model_and_data(
        hparams_ae, Model=None, version=hparams['ae_version'])
    # move model to cpu
    model_ae.to('cpu')

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

    if trials is None:
        # choose first test trial, put in list
        trials = data_generator_ae.batch_idxs[sess_idx]['test'][0]

    if isinstance(trials, int):
        trials = [trials]

    # loop over trials, putting black frames/nans in between
    ims_orig = []
    ims_recon_ae = []
    ims_recon_neural = []
    latents_ae = []
    latents_neural = []
    for i, trial in enumerate(trials):

        # get images from data generator (move to cpu)
        batch = data_generator_ae.datasets[sess_idx][trial]
        ims_orig_pt = batch['images'][:max_frames].cpu()  # 400
        if hparams_ae['model_class'] == 'cond-ae':
            labels_pt = batch['labels'][:max_frames]
        else:
            labels_pt = None

        # push images through ae to get reconstruction
        ims_recon_ae_curr, latents_ae_curr = get_reconstruction(
            model_ae, ims_orig_pt, labels=labels_pt, return_latents=True)

        # mask images for plotting
        if hparams_ae.get('use_output_mask', False):
            ims_orig_pt *= batch['masks'][:max_frames]

        # get neural activity from data generator (move to cpu)
        # 0, not sess_idx, since decoders only have 1 sess
        batch = data_generator_dec.datasets[0][trial]
        neural_activity_pt = batch['neural'][:max_frames].cpu()

        # push neural activity through decoder to get prediction
        latents_dec_pt, _ = model_dec(neural_activity_pt)
        # push prediction through ae to get reconstruction
        ims_recon_dec_curr = get_reconstruction(model_ae, latents_dec_pt, labels=labels_pt)

        # store all relevant quantities
        ims_orig.append(ims_orig_pt.cpu().detach().numpy())
        ims_recon_ae.append(ims_recon_ae_curr)
        ims_recon_neural.append(ims_recon_dec_curr)
        latents_ae.append(latents_ae_curr[:, :max_latents])
        latents_neural.append(latents_dec_pt.cpu().detach().numpy()[:, :max_latents])

        # add blank frames
        if i < len(trials) - 1:
            n_channels, y_pix, x_pix = ims_orig[-1].shape[1:]
            n = latents_ae[-1].shape[1]
            ims_orig.append(np.zeros((n_buffer, n_channels, y_pix, x_pix)))
            ims_recon_ae.append(np.zeros((n_buffer, n_channels, y_pix, x_pix)))
            ims_recon_neural.append(np.zeros((n_buffer, n_channels, y_pix, x_pix)))
            latents_ae.append(np.nan * np.zeros((n_buffer, n)))
            latents_neural.append(np.nan * np.zeros((n_buffer, n)))

    latents_ae = np.vstack(latents_ae)
    latents_neural = np.vstack(latents_neural)
    if zscore_by_dim:
        means = np.nanmean(latents_ae, axis=0)
        std = np.nanstd(latents_ae, axis=0)
        latents_ae = (latents_ae - means) / std
        latents_neural = (latents_neural - means) / std

    # away
    make_neural_reconstruction_movie(
        ims_orig=np.vstack(ims_orig),
        ims_recon_ae=np.vstack(ims_recon_ae),
        ims_recon_neural=np.vstack(ims_recon_neural),
        latents_ae=latents_ae,
        latents_neural=latents_neural,
        ae_model_class=hparams_ae['model_class'].upper(),
        colored_predictions=colored_predictions,
        xtick_locs=xtick_locs,
        frame_rate_beh=hparams['frame_rate'],
        save_file=save_file,
        frame_rate=frame_rate)


def make_neural_reconstruction_movie(
        ims_orig, ims_recon_ae, ims_recon_neural, latents_ae, latents_neural, ae_model_class='AE',
        colored_predictions=False, scale=0.5, xtick_locs=None, frame_rate_beh=None, save_file=None,
        frame_rate=15):
    """Produce movie with original video, ae reconstructed video, and neural reconstructed video.

    Latent traces are additionally plotted, as well as the residual between the ae reconstruction
    and the neural reconstruction.

    Parameters
    ----------
    ims_orig : :obj:`np.ndarray`
        original images; shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_ae : :obj:`np.ndarray`
        images reconstructed by AE; shape (n_frames, n_channels, y_pix, x_pix)
    ims_recon_neural : :obj:`np.ndarray`
        images reconstructed by neural activity; shape (n_frames, n_channels, y_pix, x_pix)
    latents_ae : :obj:`np.ndarray`
        original AE latents; shape (n_frames, n_latents)
    latents_neural : :obj:`np.ndarray`
        latents reconstruted by neural activity; shape (n_frames, n_latents)
    ae_model_class : :obj:`str`, optional
        'AE', 'VAE', etc. for plot titles
    colored_predictions : :obj:`bool`, optional
        False to plot reconstructions in black, True to plot in different colors
    scale : :obj:`int`, optional
        scale magnitude of traces
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate_beh : :obj:`float`, optional
        frame rate of behavorial video; to properly relabel xticks
    save_file : :obj:`str`, optional
        full save file (path and filename)
    frame_rate : :obj:`float`, optional
        frame rate of saved movie

    """

    means = np.nanmean(latents_ae, axis=0)
    std = np.nanstd(latents_ae) / scale

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
    axs[idx].set_title('Original', fontsize=fontsize)
    idx += 1
    axs[idx].set_title('%s reconstructed' % ae_model_class, fontsize=fontsize)
    idx += 1
    axs[idx].set_title('Neural reconstructed', fontsize=fontsize)
    idx += 1
    axs[idx].set_title('Reconstructions residual', fontsize=fontsize)
    idx += 1
    axs[idx].set_title('%s latent predictions' % ae_model_class, fontsize=fontsize)
    if xtick_locs is not None and frame_rate_beh is not None:
        axs[idx].set_xticks(xtick_locs)
        axs[idx].set_xticklabels((np.asarray(xtick_locs) / frame_rate_beh).astype('int'))
        axs[idx].set_xlabel('Time (s)', fontsize=fontsize)
    else:
        axs[idx].set_xlabel('Time (bins)', fontsize=fontsize)

    time = np.arange(n_time)

    ims_res = ims_recon_ae - ims_recon_neural

    im_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    tr_kwargs = {'animated': True, 'linewidth': 2}
    latents_ae_color = [0.2, 0.2, 0.2]

    label_ae_base = '%s latents' % ae_model_class
    label_dec_base = 'Predicted %s latents' % ae_model_class

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
        axs[idx].set_prop_cycle(None)  # reset colors
        for latent in range(n_ae_latents):
            if colored_predictions:
                latents_dec_color = axs[idx]._get_lines.get_next_color()
            else:
                latents_dec_color = [0, 0, 0]
            # just put labels on last lvs
            if latent == n_ae_latents - 1 and i == 0:
                label_ae = label_ae_base
                label_dec = label_dec_base
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
            if colored_predictions:
                # original latents - gray
                orig_line = mlines.Line2D([], [], color=[0.2, 0.2, 0.2], linewidth=3, alpha=0.7)
                # predicted latents - cycle through some colors
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                dls = []
                for c in range(5):
                    dls.append(mlines.Line2D(
                        [], [], linewidth=3, linestyle='--', dashes=(0, 3 * c, 20, 1),
                        color='%s' % colors[c]))
                plt.legend(
                    [orig_line, tuple(dls)], [label_ae_base, label_dec_base],
                    loc='lower right', fontsize=fontsize, frameon=True, framealpha=0.7,
                    edgecolor=[1, 1, 1])
            else:
                plt.legend(
                    loc='lower right', fontsize=fontsize, frameon=True,
                    framealpha=0.7, edgecolor=[1, 1, 1])
            ims_curr.append(im)
        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


def plot_neural_reconstruction_traces_wrapper(
        hparams, save_file=None, trial=None, xtick_locs=None, frame_rate=None, format='png',
        **kwargs):
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
        device='cpu', as_numpy=False, batch_load=True, rng_seed=0)

    if trial is None:
        # choose first test trial
        trial = data_generator.datasets[0].batch_idxs['test'][0]

    batch = data_generator.datasets[0][trial]
    traces_ae = batch['ae_latents'].cpu().detach().numpy()
    traces_neural = batch['ae_predictions'].cpu().detach().numpy()

    n_max_lags = hparams.get('n_max_lags', 0)  # only plot valid segment of data
    if n_max_lags > 0:
        fig = plot_neural_reconstruction_traces(
            traces_ae[n_max_lags:-n_max_lags], traces_neural[n_max_lags:-n_max_lags],
            save_file, xtick_locs, frame_rate, format, **kwargs)
    else:
        fig = plot_neural_reconstruction_traces(
            traces_ae, traces_neural, save_file, xtick_locs, frame_rate, format, **kwargs)
    return fig


def plot_neural_reconstruction_traces(
        traces_ae, traces_neural, save_file=None, xtick_locs=None, frame_rate=None, format='png',
        scale=0.5, max_traces=8, add_r2=True, add_legend=True, colored_predictions=True):
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
    scale : :obj:`int`, optional
        scale magnitude of traces
    max_traces : :obj:`int`, optional
        maximum number of traces to plot, for easier visualization
    add_r2 : :obj:`bool`, optional
        print R2 value on plot
    add_legend : :obj:`bool`, optional
        print legend on plot
    colored_predictions : :obj:`bool`, optional
        color predictions using default seaborn colormap; else predictions are black


    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

    """

    import seaborn as sns

    sns.set_style('white')
    sns.set_context('poster')

    means = np.nanmean(traces_ae, axis=0)
    std = np.nanstd(traces_ae) / scale  # scale for better visualization

    traces_ae_sc = (traces_ae - means) / std
    traces_neural_sc = (traces_neural - means) / std

    traces_ae_sc = traces_ae_sc[:, :max_traces]
    traces_neural_sc = traces_neural_sc[:, :max_traces]

    fig = plt.figure(figsize=(12, 8))
    if colored_predictions:
        plt.plot(traces_neural_sc + np.arange(traces_neural_sc.shape[1]), linewidth=3)
    else:
        plt.plot(traces_neural_sc + np.arange(traces_neural_sc.shape[1]), linewidth=3, color='k')
    plt.plot(
        traces_ae_sc + np.arange(traces_ae_sc.shape[1]), color=[0.2, 0.2, 0.2], linewidth=3,
        alpha=0.7)

    # add legend if desired
    if add_legend:
        # original latents - gray
        orig_line = mlines.Line2D([], [], color=[0.2, 0.2, 0.2], linewidth=3, alpha=0.7)
        # predicted latents - cycle through some colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        dls = []
        for c in range(5):
            dls.append(mlines.Line2D(
                [], [], linewidth=3, linestyle='--', dashes=(0, 3 * c, 20, 1),
                color='%s' % colors[c]))
        plt.legend(
            [orig_line, tuple(dls)], ['Original latents', 'Predicted latents'],
            loc='lower right', frameon=True, framealpha=0.7, edgecolor=[1, 1, 1])

    # add r2 info if desired
    if add_r2:
        from sklearn.metrics import r2_score
        r2 = r2_score(traces_ae, traces_neural, multioutput='variance_weighted')
        plt.text(
            0.05, 0.06, '$R^2$=%1.3f' % r2, horizontalalignment='left', verticalalignment='bottom',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=[1, 1, 1]))

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
