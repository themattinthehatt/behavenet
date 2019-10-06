import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from behavenet.data.utils import get_best_model_and_data
from behavenet.fitting.eval import get_reconstruction


def rotate(img_stack):
    """
    Helper function to correct rotations in musall data

    Args:
        img_stack (batch x channel x y_pix x x_pix np array)
    """
    if img_stack is not None:
        tmp = np.concatenate([np.flip(np.swapaxes(
            img_stack[:, 0, :, :], -1, -2), axis=-1)[:, None, :, :],
                              img_stack[:, 1, None, :, :]],
                             axis=1)
    else:
        tmp = None
    return tmp


def concat(ims_3channel):
    """Concatenate two images along first channel"""
    return np.concatenate(
        [ims_3channel[0, :, :], ims_3channel[1, :, :]], axis=1)


def make_ae_reconstruction_movie(
        hparams, save_file, trial=None, version='best', include_linear=False):
    """
    High-level function; calls _make_ae_reconstruction_movie

    Args:
        hparams (dict):
        save_file (str):
        trial (int, optional):
        version (str or int, optional):
        include_linear (bool, optional):
    """

    from behavenet.models import AE

    max_bins = 400

    # build model(s)
    model_ae, data_generator = get_best_model_and_data(
        hparams, AE, version=version)

    if include_linear:
        import copy
        hparams_lin = copy.copy(hparams)
        hparams_lin['model_type'] = 'linear'
        if 'lin_experiment_name' in hparams:
            hparams_lin['experiment_name'] = hparams['lin_experiment_name']
        model_lin, _ = get_best_model_and_data(hparams_lin, AE, load_data=False)

    # push images through decoder
    if trial is None:
        batch, dataset = data_generator.next_batch('test')
        ims_orig_pt = batch['images'][0, :max_bins]
        if hparams['lab'] == 'datta':
            mask = batch['masks'][0, :max_bins]
            ims_orig_pt = ims_orig_pt*mask
    else:
        batch = data_generator.datasets[0][trial]
        ims_orig_pt = batch['images'][:max_bins]
        if hparams['lab'] == 'datta':
            mask = batch['masks'][:max_bins]
            ims_orig_pt = ims_orig_pt*mask

    ims_recon_ae = get_reconstruction(model_ae, ims_orig_pt)
    if include_linear:
        ims_recon_lin = get_reconstruction(model_lin, ims_orig_pt)
    else:
        ims_recon_lin = None

    # rotate first channel of musall data
    if hparams['lab'] == 'musall':
        ims_orig = rotate(ims_orig_pt.cpu().detach().numpy())
        ims_recon_ae = rotate(ims_recon_ae)
        ims_recon_lin = rotate(ims_recon_lin)
    else:
        ims_orig = ims_orig_pt.cpu().detach().numpy()

    _make_ae_reconstruction_movie(
        ims_orig=ims_orig,
        ims_recon_ae=ims_recon_ae,
        ims_recon_lin=ims_recon_lin,
        save_file=save_file,
        frame_rate=hparams['frame_rate'])


def _make_ae_reconstruction_movie(
        ims_orig, ims_recon_ae, ims_recon_lin=None, save_file=None,
        frame_rate=20):
    """
    Args:
        ims_orig (np array):
        ims_recon_ae (np array):
        ims_recon_lin (np array, optional):
        save_file (str, optional):
        frame_rate (float, optional):
    """

    n_channels, y_pix, x_pix = ims_orig.shape[1:]
    n_cols = 2
    n_rows = 2 if ims_recon_lin is None else 3
    offset = 1 #0 if ims_recon_lin is None else 1
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    fig = plt.figure(figsize=(fig_width, fig_height + offset),dpi=100)
    # if n_ae_latents is not None:
    #     title = str(
    #         'Behavorial video compression\n%i dimensions' % n_ae_latents)
    # else:
    #     title = 'Behavorial video compression'
    # fig.suptitle(title, fontsize=20)

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
    axs.append(fig.add_subplot(gs[1, 0]))  # 1: ae reconstructed frames
    axs.append(fig.add_subplot(gs[1, 1]))  # 2: ae residuals
    if ims_recon_lin is not None:
        axs.append(fig.add_subplot(gs[2, 0]))  # 3: linear reconstructed frames
        axs.append(fig.add_subplot(gs[2, 1]))  # 4: linear residuals
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

    # TODO: concat all images here to clean up frame loop

    default_kwargs = {
        'animated': True,
        'cmap': 'gray',
        'vmin': 0,
        'vmax': 1}
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(ims_orig.shape[0]):

        ims_curr = []

        # original video
        if n_channels == 1:
            ims_orig_tmp = ims_orig[i, 0]
        else:
            ims_orig_tmp = concat(ims_orig[i])

        im = axs[0].imshow(ims_orig_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[0].spines.values()]
        ims_curr.append(im)
        # ae reconstructed video
        if n_channels == 1:
            ims_recon_ae_tmp = ims_recon_ae[i, 0]
        else:
            ims_recon_ae_tmp = concat(ims_recon_ae[i])
        im = axs[1].imshow(ims_recon_ae_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[1].spines.values()]
        ims_curr.append(im)
        # ae residual video
        if n_channels == 1:
            ims_res_ae_tmp = ims_res_ae[i, 0]
        else:
            ims_res_ae_tmp = concat(ims_res_ae[i])
        im = axs[2].imshow(0.5 + ims_res_ae_tmp, **default_kwargs)
        [s.set_visible(False) for s in axs[2].spines.values()]
        ims_curr.append(im)
        if ims_recon_lin is not None:
            # linear reconstructed video
            if n_channels == 1:
                ims_recon_lin_tmp = ims_recon_lin[i, 0]
            else:
                ims_recon_lin_tmp = concat(ims_recon_lin[i])
            im = axs[3].imshow(ims_recon_lin_tmp, **default_kwargs)
            [s.set_visible(False) for s in axs[3].spines.values()]
            ims_curr.append(im)
            # linear residual video
            if n_channels == 1:
                ims_res_lin_tmp = ims_res_lin[i, 0]
            else:
                ims_res_lin_tmp = concat(ims_res_lin[i])
            im = axs[4].imshow(0.5 + ims_res_lin_tmp, **default_kwargs)
            [s.set_visible(False) for s in axs[4].spines.values()]
            ims_curr.append(im)

        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=frame_rate, bitrate=-1)

    if save_file is not None:
        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        ani.save(save_file, writer=writer)
        # if save_file[-3:] != 'gif':
        #     save_file += '.gif'
        # ani.save(save_file, writer='imagemagick', fps=15)
        print('video saved to %s' % save_file)


def make_ae_reconstruction_movie_multisession(
        hparams, save_file, batch=None, trial=None, version='best'):
    """
    High-level function; calls make_ae_make_ae_reconstruction_movie

    Args:
        hparams (dict):
        save_file (str):
        batch (np array, optional): batch of images to reconstruct;
            makes everything much easier
        trial (int, optional):
        version (str or int, optional):
    """

    from behavenet.models import AE
    from behavenet.fitting.utils import find_session_dirs

    # find all relevant sessions
    sess_dirs, sess_ids = find_session_dirs(hparams)

    # loop over different sessions
    ims_recon_ae = []
    sess_strs = []
    for sess_id in sess_ids:

        hparams['lab'] = sess_id['lab']
        hparams['expt'] = sess_id['expt']
        hparams['animal'] = sess_id['animal']
        hparams['session'] = sess_id['session']
        hparams['multisession'] = sess_id['multisession']

        if sess_id.get('multisession', None) is not None:
            multisession = str('multisession-%02i' % sess_id['multisession'])
        if sess_id['expt'] == 'all':
            sess_str = os.path.join(multisession)
        elif sess_id['animal'] == 'all':
            sess_str = os.path.join(sess_id['expt'], multisession)
        elif sess_id['session'] == 'all':
            sess_str = os.path.join(
                sess_id['expt'], sess_id['animal'], multisession)
        else:
            sess_str = os.path.join(
                sess_id['expt'], sess_id['animal'], sess_id['session'])

        # build model(s) if they exist in the specified tt experiment
        try:

            model_ae, _ = get_best_model_and_data(
                hparams, AE, load_data=False, version=version)

            # push images through decoder
            if batch is None:
                raise NotImplementedError
            else:
                ims_orig_pt = batch

            ims_recon_ae.append(get_reconstruction(model_ae, ims_orig_pt))
            sess_strs.append(sess_str)

        except Exception:  # why doesn't StopIteration work here?
            print('Model does not exist for %s; skipping' % sess_str)
            continue
            
    if len(ims_recon_ae) == 0:
        raise Exception('No models found')
    
    # rotate first channel of musall data
    if hparams['lab'] == 'musall':
        ims_orig = rotate(ims_orig_pt.cpu().detach().numpy())
        for i, ims_recon in enumerate(ims_recon_ae):
            ims_recon_ae[i] = rotate(ims_recon)
    else:
        ims_orig = ims_orig_pt.cpu().detach().numpy()

    _make_ae_reconstruction_movie_multisession(
        ims_orig=ims_orig,
        ims_recon_ae=ims_recon_ae,
        panel_titles=sess_strs,
        save_file=save_file,
        frame_rate=hparams['frame_rate'])


def _make_ae_reconstruction_movie_multisession(
        ims_orig, ims_recon_ae, panel_titles=None, save_file=None,
        frame_rate=20):
    """
    Args:
        ims_orig (np array):
        ims_recon_ae (list of np arrays):
        panel_titles (list of strs or NoneType, optional):
        save_file (str):
        frame_rate (int):
    """

    ims_recon_ae.insert(0, ims_orig)
    if panel_titles is not None:
        panel_titles.insert(0, 'Original')

    n_channels, y_pix, x_pix = ims_recon_ae[0].shape[1:]
    n_cols = 3
    n_rows = int(np.ceil(len(ims_recon_ae) / n_cols))
    offset = 1  # 0 if ims_recon_lin is None else 1
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    fig = plt.figure(figsize=(fig_width, fig_height + offset), dpi=100)

    fontsize = 12

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    for i in range(len(ims_recon_ae)):
        row = int(np.floor(i / n_cols))
        col = int(i % n_cols)
        axs.append(fig.add_subplot(gs[row, col]))
        if panel_titles is not None:
            axs[-1].set_title(panel_titles[i], fontsize=fontsize)
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    #     ims_res_ae = ims_orig - ims_recon_ae

    # TODO: concat all images here to clean up frame loop

    default_kwargs = {
        'animated': True,
        'cmap': 'gray',
        'vmin': 0,
        'vmax': 1}
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(ims_orig.shape[0]):

        ims_curr = []

        for ax_indx, ax in enumerate(fig.axes):

            # ae reconstructed video
            if n_channels == 1:
                ims_recon_ae_tmp = ims_recon_ae[ax_indx][i, 0]
            else:
                ims_recon_ae_tmp = concat(ims_recon_ae[ax_indx][i])
            im = axs[ax_indx].imshow(ims_recon_ae_tmp, **default_kwargs)
            [s.set_visible(False) for s in axs[ax_indx].spines.values()]
            ims_curr.append(im)

        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=frame_rate, bitrate=-1)

    if save_file is not None:
        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        ani.save(save_file, writer=writer)
        # if save_file[-3:] != 'gif':
        #     save_file += '.gif'
        # ani.save(save_file, writer='imagemagick', fps=15)
        print('video saved to %s' % save_file)


def make_neural_reconstruction_movie(hparams, save_file, trial=None):
    """
    Produces ae latents and predictions from scratch

    Args:
        hparams (dict):
        save_file (str):
        trial (int, optional)
    """

    from behavenet.models import Decoder
    from behavenet.models import AE

    max_bins = 400
    max_latents = 8

    ###############################
    # build ae model/data generator
    ###############################
    hparams_ae = copy.copy(hparams)
    hparams_ae['experiment_name'] = hparams['ae_experiment_name']
    hparams_ae['model_class'] = hparams['ae_model_class']
    hparams_ae['model_type'] = hparams['ae_model_type']
    model_ae, data_generator_ae = get_best_model_and_data(
        hparams_ae, AE, version=hparams['ae_version'])
    # move model to cpu
    model_ae.to('cpu')

    if trial is None:
        # choose first test trial
        trial = data_generator_ae.batch_indxs[0]['test'][0]

    # get images from data generator (move to cpu)
    batch = data_generator_ae.datasets[0][trial]
    ims_orig_pt = batch['images'][:max_bins].cpu()  # 400

    # push images through ae to get reconstruction
    ims_recon_ae = get_reconstruction(model_ae, ims_orig_pt)
    # push images through ae to get latents
    latents_ae_pt, _, _ = model_ae.encoding(ims_orig_pt)

    #######################################
    # build decoder model/no data generator
    #######################################
    hparams_dec = copy.copy(hparams)
    hparams_dec['experiment_name'] = hparams['decoder_experiment_name']
    hparams_dec['model_class'] = hparams['decoder_model_class']
    hparams_dec['model_type'] = hparams['decoder_model_type']

    # try to load presaved latents first
    #     sess_dir, results_dir, expt_dir = get_expt_dir(hparams_dec)
    #     if hparams['decoder_version'] == 'best':
    #         best_version = get_best_model_version(expt_dir)[0]
    #     else:
    #         best_version = str('version_{}'.format(version))
    #     version_dir = os.path.join(expt_dir, best_version)
    #     latents_file = os.path.join(version_dir, 'latents.pkl')
    #     if os.exists(latents_file):
    #         print('') # TODO:load presaved latents
    model_dec, data_generator_dec = get_best_model_and_data(
        hparams_dec, Decoder, version=hparams['decoder_version'])
    # move model to cpu
    model_dec.to('cpu')

    # get neural activity from data generator (move to cpu)
    batch = data_generator_dec.datasets[0][trial]
    neural_activity_pt = batch['neural'][:max_bins].cpu()

    # push neural activity through decoder to get prediction
    latents_dec_pt, _ = model_dec(neural_activity_pt)
    # push prediction through ae to get reconstruction
    ims_recon_dec = get_reconstruction(model_ae, latents_dec_pt)

    #####################################
    # rotate first channel of musall data
    #####################################
    if hparams['lab'] == 'musall':
        ims_orig = rotate(ims_orig_pt.cpu().detach().numpy())
        ims_recon_ae = rotate(ims_recon_ae)
        ims_recon_dec = rotate(ims_recon_dec)
    else:
        ims_orig = ims_orig_pt.cpu().detach().numpy()

    # away
    _make_neural_reconstruction_movie(
        ims_orig=ims_orig,
        ims_recon_ae=ims_recon_ae,
        ims_recon_neural=ims_recon_dec,
        latents_ae=latents_ae_pt.cpu().detach().numpy()[:, :max_latents],
        latents_neural=latents_dec_pt.cpu().detach().numpy()[:, :max_latents],
        save_file=save_file)


def _make_neural_reconstruction_movie(
        ims_orig, ims_recon_ae, ims_recon_neural, latents_ae, latents_neural,
        save_file=None):
    """
    Args:
        ims_orig (np array):
        ims_recon_ae (np array):
        ims_recon_neural (np array):
        latents_ae (np array):
        latents_neural (np array):
        save_file (str, optional):
    """

    import matplotlib.pyplot as plt
    import numpy as np

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
    axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
    axs.append(fig.add_subplot(gs[0, 1]))  # 1: ae reconstructed frames
    axs.append(fig.add_subplot(gs[0, 2]))  # 2: neural reconstructed frames
    axs.append(fig.add_subplot(gs[1, 0]))  # 3: residual
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
    indx = 0
    axs[indx].set_title('Original', fontsize=fontsize); indx += 1
    axs[indx].set_title('AE reconstructed', fontsize=fontsize); indx += 1
    axs[indx].set_title('Neural reconstructed', fontsize=fontsize); indx += 1
    axs[indx].set_title('Reconstructions residual', fontsize=fontsize); indx += 1
    axs[indx].set_title('AE latent predictions', fontsize=fontsize)
    axs[indx].set_xlabel('Time (bins)', fontsize=fontsize)

    time = np.arange(ims_orig.shape[0])

    ims_res = ims_recon_ae - ims_recon_neural

    # TODO: concat all images here to clean up frame loop

    im_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    tr_kwargs = {'animated': True, 'linewidth': 2}
    latents_ae_color = [0.2, 0.2, 0.2]
    latents_dec_color = [0, 0, 0]
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(ims_orig.shape[0]):

        ims_curr = []
        indx = 0

        if i % 100 == 0:
            print('processing frame %03i' % i)

        ###################
        # behavioral videos
        ###################
        # original video
        if n_channels == 1:
            ims_orig_tmp = ims_orig[i, 0]
        else:
            ims_orig_tmp = concat(ims_orig[i])

        im = axs[indx].imshow(ims_orig_tmp, **im_kwargs)
        ims_curr.append(im);
        indx += 1;
        # ae reconstruction
        if n_channels == 1:
            ims_recon_ae_tmp = ims_recon_ae[i, 0]
        else:
            ims_recon_ae_tmp = concat(ims_recon_ae[i])
        im = axs[indx].imshow(ims_recon_ae_tmp, **im_kwargs)
        ims_curr.append(im);
        indx += 1;
        # neural reconstruction
        if n_channels == 1:
            ims_recon_neural_tmp = ims_recon_neural[i, 0]
        else:
            ims_recon_neural_tmp = concat(ims_recon_neural[i])
        im = axs[indx].imshow(ims_recon_neural_tmp, **im_kwargs)
        ims_curr.append(im);
        indx += 1
        # residual
        if n_channels == 1:
            ims_res_tmp = ims_res[i, 0]
        else:
            ims_res_tmp = concat(ims_res[i])
        im = axs[indx].imshow(0.5 + ims_res_tmp, **im_kwargs)
        ims_curr.append(im);
        indx += 1

        ########
        # traces
        ########
        # # activity over time
        # for r in range(num_regions):
        #     if pred_type == 'dlc':
        #         offset = r // 2
        #     else:
        #         offset = r
        #     im = axs[3].plot(
        #         time[0:i+1], num_regions - offset + tr_multiplier * region_means[0:i+1, r],
        #         animated=animated, label=region_labels[r], color=colors[r])[0]
        #     ims_curr.append(im)

        # latents over time
        for l in range(n_ae_latents):
            # just put labels on last lvs
            if l == n_ae_latents - 1 and i == 0:
                label_ae = 'AE latents'
                label_dec = 'Predicted AE latents'
            else:
                label_ae = None
                label_dec = None
            im = axs[indx].plot(
                time[0:i + 1], l + latents_ae_sc[0:i + 1, l],
                color=latents_ae_color, alpha=0.7, label=label_ae,
                **tr_kwargs)[0]
            axs[indx].spines['top'].set_visible(False);
            axs[indx].spines['right'].set_visible(False);
            axs[indx].spines['left'].set_visible(False)
            ims_curr.append(im)
            im = axs[indx].plot(
                time[0:i + 1], l + latents_dec_sc[0:i + 1, l],
                color=latents_dec_color, label=label_dec, **tr_kwargs)[0]
            axs[indx].spines['top'].set_visible(False);
            axs[indx].spines['right'].set_visible(False);
            axs[indx].spines['left'].set_visible(False)
            plt.legend(
                loc='lower right', fontsize=fontsize, frameon=True,
                framealpha=0.7, edgecolor=[1, 1, 1])
            ims_curr.append(im)
        ims.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=15, bitrate=-1)

    if save_file is not None:
        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        print('saving video')
        ani.save(save_file, writer=writer)
        print('video saved to %s' % save_file)


def plot_neural_reconstruction_traces(hparams, save_file, trial=None):
    """
    Loads previously saved ae latents and predictions

    Args:
        hparams (dict):
        save_file (str):
        trial (int, optional):
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

    # export latents if they don't exist
    # export_predictions_best(hparams_ae_pred)

    signals = ['ae', 'ae_predictions']
    transforms = [ae_transform, ae_pred_transform]
    paths = [ae_path, ae_pred_path]

    data_generator = ConcatSessionsGenerator(
        hparams['data_dir'], hparams,
        signals_list=[signals], transforms_list=[transforms], paths_list=[paths],
        device='cpu', as_numpy=False, batch_load=False, rng_seed=0)

    if trial is None:
        # choose first test trial
        trial = data_generator.datasets[0].batch_indxs['test'][0]

    batch = data_generator.datasets[0][trial]
    traces_ae = batch['ae'].cpu().detach().numpy()
    traces_neural = batch['ae_predictions'].cpu().detach().numpy()

    _plot_neural_reconstruction_traces(traces_ae, traces_neural, save_file)


def _plot_neural_reconstruction_traces(traces_ae, traces_neural, save_file=None):
    """
    Args:
        traces_ae (np array):
        traces_neural (np array):
        save_file (str, optional):
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

    plt.figure(figsize=(12, 8))
    plt.plot(
        traces_neural_sc + np.arange(traces_neural_sc.shape[1]), linewidth=3)
    plt.plot(
        traces_ae_sc + np.arange(traces_ae_sc.shape[1]), color=[0.2, 0.2, 0.2],
        linewidth=3, alpha=0.7)

    # add legend
    # original latents - gray
    orig_line = mlines.Line2D(
        [], [], color=[0.2, 0.2, 0.2], linewidth=3, alpha=0.7)
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

    plt.xlabel('Time (bins)')
    plt.ylabel('Latent state')
    plt.yticks([])

    if save_file is not None:
        plt.savefig(save_file + '.jpg', dpi=300, format='jpeg')

    plt.show()


def plot_latent_psths(
        latents_list, latent_strs=None, latent_indx=None, align_indx=0,
        window_len=10, style_type='light'):
    """
    For a given latent, plot its PSTH across multiple sessions (one panel per
    session)

    Args:
        latents_list (list of np arrays): trial x time x latents, one for each
            session
        latent_strs (list of strs, optional): one for each session
        latent_indx (int or NoneType): index of latent to plot; if given, plots
            average latent with all trials, one session per panel; if None,
            plots average latent across all sessions, one latent per panel
        align_indx (int): time index to center psth on
        window_len (int): size of window in each direction around `align_indx`
        style_type (str): 'dark' | 'light'
    """

    def _set_ticks(
            ax, ylim_min, ylim_max, yticks, xticks, xtick_space, r, c,
                fontsize, title_str=None):
        ax.set_xticks(xticks, [])
        ax.set_xticklabels([])
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_yticks(yticks, [])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #  axes[r, c].spines['left'].set_visible(False)
        if r == n_rows - 1:
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                np.arange(-window_len, window_len + 1, xtick_space))
            ax.set_xlabel('Time (bins)')
        if c == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([0])
        if title_str is not None:
            ax.set_title(title_str, fontsize=fontsize)

    # define fig params
    xtick_space = int(2 * window_len / 5)
    xticks = np.arange(0, 2 * window_len + 1, xtick_space)
    yticks = [0]
    fontsize = 15

    # standardize trace magnitudes (no centering)
    latents_std = np.std(np.concatenate(latents_list, axis=0))
    ylim_min = -2.5 * latents_std
    ylim_max = 2.5 * latents_std

    if style_type == 'dark':
        plt.style.use('dark_background')
        col_avg = 'white'
        col_ind = 'gray'
    else:
        col_avg = 'black'
        col_ind = 'gray'

    n_datasets = len(latents_list)
    n_latents = latents_list[0].shape[2]
    if latent_indx is None:
        n_cols = 4
        n_rows = int(np.ceil(n_latents / n_cols))
    else:
        n_cols = 2
        n_rows = int(np.ceil(n_datasets / n_cols))
    fig_width = 4 * n_cols
    fig_height = 3 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if len(axes.shape) == 1:
        axes = axes[None, :]
    for ax1 in axes:
        for ax2 in ax1:
            ax2.set_axis_off()

    # build fig
    slc = (align_indx - window_len, align_indx + window_len + 1)

    if latent_indx is None:
        for i in range(n_latents):
            r = int(np.floor(i / n_cols))
            c = i % n_cols

            # plot data
            tmp_data = [
                np.mean(latents_list[j][:, slice(*slc), i], axis=0)[:, None]
                for j in range(n_datasets)]
            axes[r, c].set_axis_on()
            for data, label in zip(tmp_data, latent_strs):
                axes[r, c].plot(data, alpha=0.8, linewidth=3, label=label)

            # plot crosshairs
            axes[r, c].plot(  # horizontal
                [0, 2 * window_len + 1], [0, 0], color=col_avg, linewidth=1)

            # deal w/ axes
            title_str = str('Latent %i' % i)
            _set_ticks(
                axes[r, c], ylim_min, ylim_max, yticks, xticks, xtick_space,
                r, c, fontsize, title_str)

        handles, labels = axes[r, c].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
            fontsize=fontsize, frameon=False)
        # put legend on side
        plt.tight_layout()

    else:
        for i, latents in enumerate(latents_list):
            r = int(np.floor(i / n_cols))
            c = i % n_cols

            # plot data
            axes[r, c].set_axis_on()
            tmp_data = latents[:, slice(*slc), latent_indx]
            axes[r, c].plot(tmp_data.T, color=col_ind, alpha=0.2)
            tmp_data = np.mean(latents[:, slice(*slc), latent_indx], axis=0)
            axes[r, c].plot(tmp_data, color=col_avg, linewidth=5)

            # plot crosshairs
            axes[r, c].plot(
                [window_len, window_len], [ylim_min, ylim_max], color=col_avg,
                linewidth=1)
            axes[r, c].plot(
                [0, 2 * window_len + 1], [0, 0], color=col_avg, linewidth=1)

            # deal w/ axes
            if latent_strs is not None:
                title_str = latent_strs[i]
            _set_ticks(
                axes[r, c], ylim_min, ylim_max, yticks, xticks, xtick_space,
                r, c, fontsize, title_str)

    plt.show()

    return fig
