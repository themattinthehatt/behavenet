import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from fitting.utils import get_best_model_and_data
from fitting.utils import get_reconstruction


def make_ae_reconstruction_movie(
        hparams, save_file, trial=None, version='best', include_linear=False):

    if 'lib' not in hparams:
        hparams['lib'] = 'pt'
    if hparams['lib'] == 'pt' or hparams['lib'] == 'pytorch':
        from behavenet.models import AE
    elif hparams['lib'] == 'tf':
        from behavenet.models_tf import AE
    else:
        raise ValueError('"%s" is an invalid library' % hparams['lib'])

    max_bins = 400

    # build model(s)
    model_cae, data_generator = get_best_model_and_data(
        hparams, AE, version=version)

    if include_linear:
        import copy
        hparams_lin = copy.copy(hparams)
        hparams_lin['model_type'] = 'linear'
        model_lin, _ = get_best_model_and_data(hparams_lin, AE, load_data=False)

    # push images through decoder
    if trial is None:
        batch, dataset = data_generator.next_batch('test')
        ims_orig_pt = batch['images'][0, :max_bins]
    else:
        batch = data_generator.datasets[0][trial]
        ims_orig_pt = batch['images'][:max_bins]

    ims_recon_cae = get_reconstruction(model_cae, ims_orig_pt)
    if include_linear:
        ims_recon_lin = get_reconstruction(model_lin, ims_orig_pt)
    else:
        ims_recon_lin = None

    # rotate first channel of musall data
    if hparams['lab'] == 'musall':
        def rotate(img_stack):
            if img_stack is not None:
                tmp = np.concatenate([np.flip(np.swapaxes(
                    img_stack[:, 0, :, :], -1, -2), axis=-1)[:, None, :, :],
                    img_stack[:, 1, None, :, :]],
                    axis=1)
            else:
                tmp = None
            return tmp
        ims_orig = rotate(ims_orig_pt.cpu().detach().numpy())
        ims_recon_cae = rotate(ims_recon_cae)
        ims_recon_lin = rotate(ims_recon_lin)
    else:
        ims_orig = ims_orig_pt.cpu().detach().numpy()

    _make_ae_reconstruction_movie(
        ims_orig=ims_orig,
        ims_recon_cae=ims_recon_cae,
        ims_recon_lin=ims_recon_lin,
        save_file=save_file)


def _make_ae_reconstruction_movie(
        ims_orig, ims_recon_cae, ims_recon_lin=None, save_file=None):

    n_channels, y_pix, x_pix = ims_orig.shape[1:]
    n_cols = 3
    n_rows = 1 if ims_recon_lin is None else 2
    offset = 1 #0 if ims_recon_lin is None else 1
    scale_ = 5
    fig_width = scale_ * n_cols * n_channels / 2
    fig_height = y_pix / x_pix * scale_ * n_rows / 2
    fig = plt.figure(figsize=(fig_width, fig_height + offset))
    # if n_ae_latents is not None:
    #     title = str(
    #         'Behavorial video compression\n%i dimensions' % n_ae_latents)
    # else:
    #     title = 'Behavorial video compression'
    # fig.suptitle(title, fontsize=20)

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
    axs.append(fig.add_subplot(gs[0, 1]))  # 1: cae reconstructed frames
    axs.append(fig.add_subplot(gs[0, 2]))  # 2: cae residuals
    if ims_recon_lin is not None:
        axs.append(fig.add_subplot(gs[1, 1]))  # 3: linear reconstructed frames
        axs.append(fig.add_subplot(gs[1, 2]))  # 4: linear residuals
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 12
    axs[0].set_title('Original', fontsize=fontsize)
    axs[1].set_title('Conv reconstructed', fontsize=fontsize)
    axs[2].set_title('Conv residual', fontsize=fontsize)
    if ims_recon_lin is not None:
        axs[3].set_title('Linear reconstructed', fontsize=fontsize)
        axs[4].set_title('Linear residual', fontsize=fontsize)

    ims_res_cae = ims_orig - ims_recon_cae
    if ims_recon_lin is not None:
        ims_res_lin = ims_orig - ims_recon_lin

    def concat(ims_3channel):
        return np.concatenate(
            [ims_3channel[0, :, :], ims_3channel[1, :, :]], axis=1)

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
        ims_curr.append(im)
        # cae reconstructed video
        if n_channels == 1:
            ims_recon_cae_tmp = ims_recon_cae[i, 0]
        else:
            ims_recon_cae_tmp = concat(ims_recon_cae[i])
        im = axs[1].imshow(ims_recon_cae_tmp, **default_kwargs)
        ims_curr.append(im)
        # cae residual video
        if n_channels == 1:
            ims_res_cae_tmp = ims_res_cae[i, 0]
        else:
            ims_res_cae_tmp = concat(ims_res_cae[i])
        im = axs[2].imshow(0.5 + ims_res_cae_tmp, **default_kwargs)
        ims_curr.append(im)
        if ims_recon_lin is not None:
            # linear reconstructed video
            if n_channels == 1:
                ims_recon_lin_tmp = ims_recon_lin[i, 0]
            else:
                ims_recon_lin_tmp = concat(ims_recon_lin[i])
            im = axs[3].imshow(ims_recon_lin_tmp, **default_kwargs)
            ims_curr.append(im)
            # linear residual video
            if n_channels == 1:
                ims_res_lin_tmp = ims_res_lin[i, 0]
            else:
                ims_res_lin_tmp = concat(ims_res_lin[i])
            im = axs[4].imshow(0.5 + ims_res_lin_tmp, **default_kwargs)
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
        ani.save(save_file, writer=writer)
        print('video saved to %s' % save_file)


def make_neural_reconstruction_movie(hparams, save_file, trial=None):

    from behavenet.models import Decoder

    if 'lib' not in hparams:
        hparams['lib'] = 'pt'
    if hparams['lib'] == 'pt' or hparams['lib'] == 'pytorch':
        from behavenet.models import AE
    else:
        raise NotImplementedError

    max_bins = 400
    max_latents = 8

    ###############################
    # build ae model/data generator
    ###############################
    hparams_ae = copy.copy(hparams)
    hparams_ae['model_class'] = hparams['ae_model_class']
    hparams_ae['experiment_name'] = hparams['ae_experiment_name']
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
    hparams_dec['model_class'] = hparams['decoder_model_class']
    hparams_dec['experiment_name'] = hparams['decoder_experiment_name']
    hparams_dec['model_type'] = hparams['decoder_model_type']

    # try to load presaved latents first
    #     sess_dir, results_dir, expt_dir = get_output_dirs(hparams_dec)
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
    latents_dec_pt = model_dec(neural_activity_pt)
    # push prediction through ae to get reconstruction
    ims_recon_dec = get_reconstruction(model_ae, latents_dec_pt)

    #####################################
    # rotate first channel of musall data
    #####################################
    if hparams['lab'] == 'musall':
        def rotate(img_stack):
            if img_stack is not None:
                tmp = np.concatenate([np.flip(np.swapaxes(
                    img_stack[:, 0, :, :], -1, -2), axis=-1)[:, None, :, :],
                    img_stack[:, 1, None, :, :]],
                    axis=1)
            else:
                tmp = None
            return tmp

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

    def concat(ims_3channel):
        return np.concatenate(
            [ims_3channel[0, :, :], ims_3channel[1, :, :]], axis=1)

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
