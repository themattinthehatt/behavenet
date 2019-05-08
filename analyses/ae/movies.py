import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter


def reconstruction(hparams, save_file, trial=None, include_linear=False):

    from fitting.utils import get_best_model_and_data
    from fitting.utils import get_reconstruction
    if hparams['lib'] == 'torch':
        from behavenet.models import AE
    elif hparams['lib'] == 'tf':
        from behavenet.models_tf import AE

    # build model(s)
    model_cae, data_generator = get_best_model_and_data(hparams, AE)

    if include_linear:
        import copy
        hparams_lin = copy.copy(hparams)
        hparams_lin['model_type'] = 'linear'
        model_lin, _ = get_best_model_and_data(hparams_lin, AE, load_data=False)

    # push images through decoder
    if trial is None:
        batch, dataset = data_generator.next_batch('test')
        ims_orig_pt = batch['images'][0, :200]
    else:
        batch = data_generator.datasets[0][trial]
        ims_orig_pt = batch['images'][:200]

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

    make_ae_reconstruction_movie(
        ims_orig=ims_orig,
        ims_recon_cae=ims_recon_cae,
        ims_recon_lin=ims_recon_lin,
        save_file=save_file,
        n_ae_latents=hparams['n_ae_latents'])


def make_ae_reconstruction_movie(
        ims_orig, ims_recon_cae, ims_recon_lin=None, save_file=None,
        n_ae_latents=None):

    n_channels, y_pix, x_pix = ims_orig.shape[1:]
    n_cols = 3 if ims_recon_lin is None else 2
    n_rows = 1 if ims_recon_lin is None else 3
    scale_ = 4 if n_channels == 1 else 3.5
    fig_width = scale_ * n_cols * n_channels
    fig_height = y_pix / x_pix * scale_ * n_rows
    offset = 1.5  # if n_channels == 1 else 0
    fig = plt.figure(figsize=(fig_width, fig_height + offset))
    if n_ae_latents is not None:
        title = str(
            'Behavorial video compression\n%i dimensions' % n_ae_latents)
    else:
        title = 'Behavorial video compression'
    fig.suptitle(title, fontsize=20)

    gs = GridSpec(n_rows, n_cols, figure=fig)
    axs = []
    if ims_recon_lin is None:
        axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
        axs.append(fig.add_subplot(gs[0, 1]))  # 1: cae reconstructed frames
        axs.append(fig.add_subplot(gs[0, 2]))  # 2: cae residuals
    else:
        axs.append(fig.add_subplot(gs[0, 0]))  # 0: original frames
        axs.append(fig.add_subplot(gs[1, 0]))  # 1: cae reconstructed frames
        axs.append(fig.add_subplot(gs[1, 1]))  # 2: cae residuals
        axs.append(fig.add_subplot(gs[2, 0]))  # 3: linear reconstructed frames
        axs.append(fig.add_subplot(gs[2, 1]))  # 4: linear residuals
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 16
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

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    metadata = {'title': 'ae reconstruction'}
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=-1)

    if save_file is not None:
        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if save_file[-3:] != 'mp4':
            save_file += '.mp4'
        ani.save(save_file, writer=writer)
        print('video saved to %s' % save_file)
