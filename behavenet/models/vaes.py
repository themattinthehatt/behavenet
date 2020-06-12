"""Variational autoencoder models implemented in PyTorch."""

import numpy as np
import torch
import behavenet.fitting.losses as losses
from behavenet.models.aes import AE


class VAE(AE):
    """Base variational autoencoder class.

    This class constructs convolutional variational autoencoders. The convolutional autoencoder
    architecture is defined by various keys in the dict that serves as the constructor input. See
    the :mod:`behavenet.fitting.ae_model_architecture_generator` module to see examples for how
    this is done.
    """

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            - 'model_type' (:obj:`int`): 'conv'
            - 'model_class' (:obj:`str`): 'vae'
            - 'y_pixels' (:obj:`int`)
            - 'x_pixels' (:obj:`int`)
            - 'n_input_channels' (:obj:`int`)
            - 'n_ae_latents' (:obj:`int`)
            - 'fit_sess_io_layers; (:obj:`bool`): fit session-specific input/output layers
            - 'ae_encoding_x_dim' (:obj:`list`)
            - 'ae_encoding_y_dim' (:obj:`list`)
            - 'ae_encoding_n_channels' (:obj:`list`)
            - 'ae_encoding_kernel_size' (:obj:`list`)
            - 'ae_encoding_stride_size' (:obj:`list`)
            - 'ae_encoding_x_padding' (:obj:`list`)
            - 'ae_encoding_y_padding' (:obj:`list`)
            - 'ae_encoding_layer_type' (:obj:`list`)
            - 'ae_decoding_x_dim' (:obj:`list`)
            - 'ae_decoding_y_dim' (:obj:`list`)
            - 'ae_decoding_n_channels' (:obj:`list`)
            - 'ae_decoding_kernel_size' (:obj:`list`)
            - 'ae_decoding_stride_size' (:obj:`list`)
            - 'ae_decoding_x_padding' (:obj:`list`)
            - 'ae_decoding_y_padding' (:obj:`list`)
            - 'ae_decoding_layer_type' (:obj:`list`)
            - 'ae_decoding_starting_dim' (:obj:`list`)
            - 'ae_decoding_last_FF_layer' (:obj:`bool`)

        """
        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        super().__init__(hparams)

    def forward(self, x, dataset=None, use_mean=False, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers
        use_mean : :obj:`bool`
            True to skip sampling step

        Returns
        -------
        :obj:`tuple`
            - x_hat (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - z (:obj:`torch.Tensor`): sampled latent variable of shape (n_frames, n_latents)
            - mu (:obj:`torch.Tensor`): mean paramter of shape (n_frames, n_latents)
            - logvar (:obj:`torch.Tensor`): logvar paramter of shape (n_frames, n_latents)

        """
        mu, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        return x_hat, z, mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from N(mu, var)

        Parameters
        ----------
        mu : :obj:`torch.Tensor`
            vector of mean parameters
        logvar : :obj:`torch.Tensor`
            vector of log variances; only mean field approximation is currently implemented

        Returns
        -------
        :obj:`torch.Tensor`
            sampled vector of shape (n_frames, n_latents)

        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate MSE loss for autoencoder.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        data : :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        dataset : :obj:`int`, optional
            used for session-specific io layers
        accumulate_grad : :obj:`bool`, optional
            accumulate gradient for training step
        chunk_size : :obj:`int`, optional
            batch is split into chunks of this size to keep memory requirements low

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): mse loss

        """

        if self.hparams['device'] == 'cuda':
            data = {key: val.to('cuda') for key, val in data.items()}

        x = data['images'][0]
        m = data['masks'][0] if 'masks' in data else None

        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        loss_val = 0
        loss_ll_val = 0
        loss_kl_val = 0
        loss_mse_val = 0
        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            x_hat, _, mu, logvar = self.forward(x_in, dataset=dataset, use_mean=False)

            # log-likelihood
            loss_ll = losses.gaussian_ll(x_in, x_hat, m_in)

            # kl
            loss_kl = losses.kl_div_to_std_normal(mu, logvar)

            # combine
            loss = -loss_ll + self.hparams['vae.beta'] * loss_kl

            if accumulate_grad:
                loss.backward()

            # get loss value (weighted by batch size)
            loss_val += loss.item() * (idx_end - idx_beg)
            loss_ll_val += loss_ll.item() * (idx_end - idx_beg)
            loss_kl_val += loss_kl.item() * (idx_end - idx_beg)
            loss_mse_val += losses.gaussian_ll_to_mse(
                loss_ll.item(), np.prod(x.shape[1:])) * (idx_end - idx_beg)

        loss_val /= batch_size
        loss_ll_val /= batch_size
        loss_kl_val /= batch_size
        loss_mse_val /= batch_size

        loss_dict = {
            'loss': loss_val, 'loss_ll': loss_ll_val, 'loss_kl': loss_kl_val,
            'loss_mse': loss_mse_val}

        return loss_dict
