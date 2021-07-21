"""Variational autoencoder models implemented in PyTorch."""

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn

import behavenet.fitting.losses as losses
from behavenet.models.aes import AE, ConvAEDecoder, ConvAEEncoder

# to ignore imports for sphix-autoapidoc
__all__ = [
    'reparameterize', 'VAE', 'ConditionalVAE', 'BetaTCVAE', 'PSVAE', 'MSPSVAE', 'ConvAEPSEncoder',
    'ConvAEMSPSEncoder']


def reparameterize(mu, logvar):
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


class VAE(AE):
    """Base variational autoencoder class.

    This class constructs convolutional variational autoencoders. The convolutional autoencoder
    architecture is defined by various keys in the dict that serves as the constructor input. See
    the :mod:`behavenet.fitting.ae_model_architecture_generator` module to see examples for how
    this is done.

    The VAE class can also be used to fit β-VAE models (see https://arxiv.org/pdf/1804.03599.pdf)
    by changing the value of the `vae.beta` parameter in the `ae_model.json` file; a value of 1
    corresponds to a standard VAE; a value >1 will upweight the KL divergence term which, in some
    cases, can lead to disentangling of the latent representation.
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
            - 'vae.beta' (:obj:`float`)
            - 'vae.beta_anneal_epochs' (:obj:`int`)
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
        hparams['variational'] = True
        super().__init__(hparams)

        # set up kl annealing
        anneal_epochs = self.hparams.get('vae.beta_anneal_epochs', 0)
        self.curr_epoch = 0  # must be modified by training script
        if anneal_epochs > 0:
            self.beta_vals = np.append(
                np.linspace(0, hparams['vae.beta'], anneal_epochs),
                np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
        else:
            self.beta_vals = hparams['vae.beta'] * np.ones(hparams['max_n_epochs'] + 1)

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
            z = reparameterize(mu, logvar)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        return x_hat, z, mu, logvar

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate ELBO loss for VAE.

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
            - 'loss' (:obj:`float`): full elbo
            - 'loss_ll' (:obj:`float`): log-likelihood portion of elbo
            - 'loss_kl' (:obj:`float`): kl portion of elbo
            - 'loss_mse' (:obj:`float`): mse (without gaussian constants)
            - 'beta' (:obj:`float`): weight in front of kl term

        """

        x = data['images'][0]
        m = data['masks'][0] if 'masks' in data else None
        beta = self.beta_vals[self.curr_epoch]

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
            loss = -loss_ll + beta * loss_kl

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
            'loss_mse': loss_mse_val, 'beta': beta}

        return loss_dict


class ConditionalVAE(VAE):
    """Conditional variational autoencoder class.

    This class constructs conditional convolutional variational autoencoders. At the latent layer
    an additional set of variables, saved under the 'labels' key in the hdf5 data file, are
    concatenated with the latents before being reshaped into a 2D array for decoding.
    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain :obj:`n_labels` and
            :obj:`conditional_encoder`

        """
        super().__init__(hparams)

    def build_model(self):
        """Construct the model using hparams.

        The ConditionalAE is initialized when :obj:`model_class='cond-ae`, and currently only
        supports :obj:`model_type='conv` (i.e. no linear)
        """
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents'] + self.hparams['n_labels']
        self.encoding = ConvAEEncoder(self.hparams)
        self.decoding = ConvAEDecoder(self.hparams)

    def forward(self, x, dataset=None, labels=None, labels_2d=None, use_mean=False, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data of shape (batch, n_channels, y_pix, x_pix)
        dataset : :obj:`int`
            used with session-specific io layers
        labels : :obj:`torch.Tensor` object
            continuous labels corresponding to input data, of shape (batch, n_labels)
        labels_2d: :obj:`torch.Tensor` object
            one-hot labels corresponding to input data, of shape (batch, n_labels, y_pix, x_pix);
            for a given frame, each channel corresponds to a label and is all zeros with a single
            value of one in the proper x/y position
        use_mean : :obj:`bool`
            True to skip sampling step

        Returns
        -------
        :obj:`tuple`
            - y (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - x (:obj:`torch.Tensor`): hidden representation of shape (n_frames, n_latents)

        """
        if self.hparams['conditional_encoder']:
            # append label information to input
            x = torch.cat((x, labels_2d), dim=1)
        mu, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        if use_mean:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        z_aug = torch.cat((z, labels), dim=1)
        x_hat = self.decoding(z_aug, pool_idx, outsize, dataset=dataset)
        return x_hat, z, mu, logvar

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate ELBO loss for ConditionalVAE.

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
            - 'loss' (:obj:`float`): full elbo
            - 'loss_ll' (:obj:`float`): log-likelihood portion of elbo
            - 'loss_kl' (:obj:`float`): kl portion of elbo
            - 'loss_mse' (:obj:`float`): mse (without gaussian constants)
            - 'beta' (:obj:`float`): weight in front of kl term

        """

        x = data['images'][0]
        y = data['labels'][0]
        m = data['masks'][0] if 'masks' in data else None
        if self.hparams['conditional_encoder']:
            # continuous labels transformed into 2d one-hot array as input to encoder
            y_2d = data['labels_sc'][0]
        else:
            y_2d = None
        beta = self.beta_vals[self.curr_epoch]

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
            y_in = y[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            y_2d_in = y_2d[idx_beg:idx_end] if y_2d is not None else None
            x_hat, _, mu, logvar = self.forward(
                x_in, dataset=dataset, use_mean=False, labels=y_in, labels_2d=y_2d_in)

            # log-likelihood
            loss_ll = losses.gaussian_ll(x_in, x_hat, m_in)

            # kl
            loss_kl = losses.kl_div_to_std_normal(mu, logvar)

            # combine
            loss = -loss_ll + beta * loss_kl

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
            'loss_mse': loss_mse_val, 'beta': beta}

        return loss_dict


class BetaTCVAE(VAE):
    """Beta Total Correlation VAE class.

    This class constructs convolutional variational autoencoders and decomposes the KL divergence
    term in the ELBO into three terms:
    1. index code mutual information
    2. total correlation
    3. dimension-wise KL

    The total correlation term is up-weighted to encourage "disentangled" latents; for more
    information, see https://arxiv.org/pdf/1802.04942.pdf.
    """

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain :obj:`btcvae.beta`

        """
        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        super().__init__(hparams)

        # set up beta annealing
        anneal_epochs = self.hparams.get('beta_tcvae.beta_anneal_epochs', 0)
        self.curr_epoch = 0  # must be modified by training script
        beta = hparams['beta_tcvae.beta']
        # TODO: these values should not be precomputed
        if anneal_epochs > 0:
            # annealing for total correlation term
            self.beta_vals = np.append(
                np.linspace(0, beta, anneal_epochs),  # USED TO START AT 1!!
                beta * np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
            # annealing for remaining kl terms - index code mutual info and dim-wise kl
            self.kl_anneal_vals = np.append(
                np.linspace(0, 1, anneal_epochs),
                np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
        else:
            self.beta_vals = beta * np.ones(hparams['max_n_epochs'] + 1)
            self.kl_anneal_vals = np.ones(hparams['max_n_epochs'] + 1)

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate (decomposed) ELBO loss for VAE.

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
            - 'loss' (:obj:`float`): full elbo
            - 'loss_ll' (:obj:`float`): log-likelihood portion of elbo
            - 'loss_mi' (:obj:`float`): code index mutual info portion of kl of elbo
            - 'loss_tc' (:obj:`float`): total correlation portion of kl of elbo
            - 'loss_dwkl' (:obj:`float`): dim-wise kl portion of kl of elbo
            - 'loss_mse' (:obj:`float`): mse (without gaussian constants)
            - 'beta' (:obj:`float`): weight in front of kl term

        """

        x = data['images'][0]
        m = data['masks'][0] if 'masks' in data else None
        beta = self.beta_vals[self.curr_epoch]
        kl = self.kl_anneal_vals[self.curr_epoch]

        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))

        loss_strs = ['loss', 'loss_ll', 'loss_mi', 'loss_tc', 'loss_dwkl']

        loss_dict_vals = {loss: 0 for loss in loss_strs}
        loss_dict_vals['loss_mse'] = 0

        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            x_hat, sample, mu, logvar = self.forward(x_in, dataset=dataset, use_mean=False)

            # reset losses
            loss_dict_torch = {loss: 0 for loss in loss_strs}

            # data log-likelihood
            loss_dict_torch['loss_ll'] = losses.gaussian_ll(x_in, x_hat, m_in)
            loss_dict_torch['loss'] -= loss_dict_torch['loss_ll']

            # compute all terms of decomposed elbo at once
            index_code_mi, total_correlation, dimension_wise_kl = losses.decomposed_kl(
                sample, mu, logvar)

            # unsupervised latents index-code mutual information
            loss_dict_torch['loss_mi'] = index_code_mi
            loss_dict_torch['loss'] += kl * loss_dict_torch['loss_mi']

            # unsupervised latents total correlation
            loss_dict_torch['loss_tc'] = total_correlation
            loss_dict_torch['loss'] += beta * loss_dict_torch['loss_tc']

            # unsupervised latents dimension-wise kl
            loss_dict_torch['loss_dwkl'] = dimension_wise_kl
            loss_dict_torch['loss'] += kl * loss_dict_torch['loss_dwkl']

            if accumulate_grad:
                loss_dict_torch['loss'].backward()

            # get loss value (weighted by batch size)
            bs = idx_end - idx_beg
            for key, val in loss_dict_torch.items():
                loss_dict_vals[key] += val.item() * bs
            loss_dict_vals['loss_mse'] += losses.gaussian_ll_to_mse(
                loss_dict_vals['loss_ll'] / bs, np.prod(x.shape[1:])) * bs

        # compile (properly weighted) loss terms
        for key in loss_dict_vals.keys():
            loss_dict_vals[key] /= batch_size
        # store hyperparams
        loss_dict_vals['beta'] = beta

        return loss_dict_vals


class PSVAE(AE):
    """Partitioned subspace variational autoencoder class.

    This class constructs a VAE that...

    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain:
            - 'n_labels' (:obj:`n_labels`)
            - 'ps_vae.alpha' (:obj:`float`)
            - 'ps_vae.beta' (:obj:`float`)

        """

        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        if hparams['n_ae_latents'] < hparams['n_labels']:
            raise ValueError('PS-VAE model must contain at least as many latents as labels')

        self.n_latents = hparams['n_ae_latents']
        self.n_labels = hparams['n_labels']

        hparams['variational'] = True
        super().__init__(hparams)

        # set up beta annealing
        anneal_epochs = self.hparams.get('ps_vae.anneal_epochs', 0)
        self.curr_epoch = 0  # must be modified by training script
        beta = hparams['ps_vae.beta']
        # TODO: these values should not be precomputed
        if anneal_epochs > 0:
            # annealing for total correlation term
            self.beta_vals = np.append(
                np.linspace(0, beta, anneal_epochs),  # USED TO START AT 1!!
                beta * np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
            # annealing for remaining kl terms - index code mutual info and dim-wise kl
            self.kl_anneal_vals = np.append(
                np.linspace(0, 1, anneal_epochs),
                np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
        else:
            self.beta_vals = beta * np.ones(hparams['max_n_epochs'] + 1)
            self.kl_anneal_vals = np.ones(hparams['max_n_epochs'] + 1)

    def build_model(self):
        """Construct the model using hparams."""
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents']
        if self.model_type == 'conv':
            self.encoding = ConvAEPSEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            raise NotImplementedError
            # if self.hparams.get('fit_sess_io_layers', False):
            #     raise NotImplementedError
            # n_latents = self.hparams['n_ae_latents']
            # self.encoding = LinearAEEncoder(n_latents, self.img_size)
            # self.decoding = LinearAEDecoder(n_latents, self.img_size, self.encoding)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

    def forward(self, x, dataset=None, use_mean=False, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data of shape (n_frames, n_channels, y_pix, x_pix)
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
            - y_hat (:obj:`torch.Tensor`): output of shape (n_frames, n_labels)

        """
        y, w, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        mu = torch.cat([y, w], axis=1)
        if use_mean:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        y_hat = self.encoding.D(y)
        return x_hat, z, mu, logvar, y_hat

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate modified ELBO loss for PSVAE.

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
            - 'loss' (:obj:`float`): full elbo
            - 'loss_ll' (:obj:`float`): log-likelihood portion of elbo
            - 'loss_kl' (:obj:`float`): kl portion of elbo
            - 'loss_mse' (:obj:`float`): mse (without gaussian constants)
            - 'beta' (:obj:`float`): weight in front of kl term

        """

        x = data['images'][0]
        y = data['labels'][0]
        m = data['masks'][0] if 'masks' in data else None
        n = data['labels_masks'][0] if 'labels_masks' in data else None
        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))
        n_labels = self.hparams['n_labels']
        # n_latents = self.hparams['n_ae_latents']

        # compute hyperparameters
        alpha = self.hparams['ps_vae.alpha']
        beta = self.beta_vals[self.curr_epoch]
        kl = self.kl_anneal_vals[self.curr_epoch]

        loss_strs = [
            'loss', 'loss_data_ll', 'loss_label_ll', 'loss_zs_kl', 'loss_zu_mi', 'loss_zu_tc',
            'loss_zu_dwkl']

        loss_dict_vals = {loss: 0 for loss in loss_strs}
        loss_dict_vals['loss_data_mse'] = 0

        y_hat_all = []

        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            y_in = y[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            n_in = n[idx_beg:idx_end] if n is not None else None
            x_hat, sample, mu, logvar, y_hat = self.forward(x_in, dataset=dataset, use_mean=False)

            # reset losses
            loss_dict_torch = {loss: 0 for loss in loss_strs}

            # data log-likelihood
            loss_dict_torch['loss_data_ll'] = losses.gaussian_ll(x_in, x_hat, m_in)
            loss_dict_torch['loss'] -= loss_dict_torch['loss_data_ll']

            # label log-likelihood
            loss_dict_torch['loss_label_ll'] = losses.gaussian_ll(y_in, y_hat, n_in)
            loss_dict_torch['loss'] -= alpha * loss_dict_torch['loss_label_ll']

            # supervised latents kl
            loss_dict_torch['loss_zs_kl'] = losses.kl_div_to_std_normal(
                mu[:, :n_labels], logvar[:, :n_labels])
            loss_dict_torch['loss'] += loss_dict_torch['loss_zs_kl']

            # compute all terms of decomposed elbo at once
            index_code_mi, total_correlation, dimension_wise_kl = losses.decomposed_kl(
                sample[:, n_labels:], mu[:, n_labels:], logvar[:, n_labels:])

            # unsupervised latents index-code mutual information
            loss_dict_torch['loss_zu_mi'] = index_code_mi
            loss_dict_torch['loss'] += kl * loss_dict_torch['loss_zu_mi']

            # unsupervised latents total correlation
            loss_dict_torch['loss_zu_tc'] = total_correlation
            loss_dict_torch['loss'] += beta * loss_dict_torch['loss_zu_tc']

            # unsupervised latents dimension-wise kl
            loss_dict_torch['loss_zu_dwkl'] = dimension_wise_kl
            loss_dict_torch['loss'] += kl * loss_dict_torch['loss_zu_dwkl']

            if accumulate_grad:
                loss_dict_torch['loss'].backward()

            # get loss value (weighted by batch size)
            bs = idx_end - idx_beg
            for key, val in loss_dict_torch.items():
                loss_dict_vals[key] += val.item() * bs
            loss_dict_vals['loss_data_mse'] += losses.gaussian_ll_to_mse(
                loss_dict_vals['loss_data_ll'] / bs, np.prod(x.shape[1:])) * bs

            # collect predicted labels to compute R2
            y_hat_all.append(y_hat.cpu().detach().numpy())

        # use variance-weighted r2s to ignore small-variance latents
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        y_all = y.cpu().detach().numpy()
        if n is not None:
            n_np = n.cpu().detach().numpy()
            r2 = r2_score(y_all[n_np == 1], y_hat_all[n_np == 1], multioutput='variance_weighted')
        else:
            r2 = r2_score(y_all, y_hat_all, multioutput='variance_weighted')

        # compile (properly weighted) loss terms
        for key in loss_dict_vals.keys():
            loss_dict_vals[key] /= batch_size

        # store hyperparams
        loss_dict_vals['alpha'] = alpha
        loss_dict_vals['beta'] = beta
        loss_dict_vals['label_r2'] = r2

        return loss_dict_vals

    def get_predicted_labels(self, x, dataset=None, use_mean=True):
        """Process input data to get predicted labels.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data of shape (n_frames, n_channels, y_pix, x_pix)
        dataset : :obj:`int`
            used with session-specific io layers
        use_mean : :obj:`bool`
            True to skip sampling step

        Returns
        -------
        :obj:`torch.Tensor`
            output of shape (n_frames, n_labels)

        """
        y, w, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        if not use_mean:
            y = reparameterize(y, logvar[:, :self.n_labels])
        y_hat = self.encoding.D(y)
        return y_hat

    def get_transformed_latents(self, inputs, dataset=None, as_numpy=True):
        """Return latents after supervised subspace has been transformed to original label space.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor` object
            - image tensor of shape (n_frames, n_channels, y_pix, x_pix)
            - latents tensor of shape (n_frames, n_ae_latents)
        dataset : :obj:`int`, optional
            used with session-specific io layers
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in transformed latent space of shape (n_frames, n_latents)

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)

        # check to see if inputs are images or latents
        if len(inputs.shape) == 2:
            input_type = 'latents'
        else:
            input_type = 'images'

        # get latents in original space
        if input_type == 'images':
            y_og, w_og, logvar, pool_idx, outsize = self.encoding(inputs, dataset=dataset)
        else:
            y_og = inputs[:, :self.hparams['n_labels']]
            w_og = inputs[:, self.hparams['n_labels']:]

        # transform supervised latents to label space
        y_new = self.encoding.D(y_og)

        latents_tr = torch.cat([y_new, w_og], axis=1)

        if as_numpy:
            return latents_tr.cpu().detach().numpy()
        else:
            return latents_tr

    def get_inverse_transformed_latents(self, inputs, dataset=None, as_numpy=True):
        """Return latents after they have been transformed using the diagonal mapping D.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor` object
            - image tensor of shape (n_frames, n_channels, y_pix, x_pix)
            - latents tensor of shape (n_frames, n_ae_latents) where the first n_labels entries are
              assumed to be labels in the original pixel space
        dataset : :obj:`int`, optional
            used with session-specific io layers
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in transformed latent space of shape (n_frames, n_latents)

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)

        # check to see if inputs are images or latents
        if len(inputs.shape) == 2:
            input_type = 'latents'
        else:
            input_type = 'images'

        # get latents in original space
        if input_type == 'images':
            raise NotImplementedError
        else:
            y_og = inputs[:, :self.hparams['n_labels']]
            w_og = inputs[:, self.hparams['n_labels']:]

        # transform given labels to latent space
        y_new = torch.div(torch.sub(y_og, self.encoding.D.bias), self.encoding.D.weight)

        latents_tr = torch.cat([y_new, w_og], axis=1)

        if as_numpy:
            return latents_tr.cpu().detach().numpy()
        else:
            return latents_tr


class MSPSVAE(PSVAE):
    """Partitioned subspace variational autoencoder class for multiple sessions."""

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain:
            - 'n_labels' (:obj:`n_labels`)  # number of supervised dims (number of labels)
            - 'ps_vae.alpha' (:obj:`float`)  # weight on label reconstruction loss
            - 'ps_vae.beta' (:obj:`float`)  # weight on unsupervised TC loss
            - 'ps_vae.gamma' (:obj:`float`)  # weight on orthogonalization loss
            - 'ps_vae.delta' (:obj:`float`)  # weight on background embedding loss
            - 'n_background' (:obj:`int`)  # dimensionality of background latent space
            - 'n_sessions_per_batch' (:obj:`int`)  # data generator param, >1
            - 'ps_vae.ms_loss' (:obj:`str`)  # multi-session loss: 'triplet' | 'classification'

        """
        if hparams['n_sessions_per_batch'] == 1:
            raise ValueError('must choose "n_sessions_per_batch" > 1 in hparams')
        super().__init__(hparams)
        n_background = self.hparams.get('n_background', 4)
        self.hparams['n_background'] = n_background  # make sure this gets saved
        self.TripletLoss = nn.TripletMarginLoss(margin=1.0, p=2)

    def build_model(self):
        """Construct the model using hparams."""
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents']
        if self.model_type == 'conv':
            self.encoding = ConvAEMSPSEncoder(self.hparams)
            self.decoding = ConvAEDecoder(self.hparams)
        elif self.model_type == 'linear':
            raise NotImplementedError
            # if self.hparams.get('fit_sess_io_layers', False):
            #     raise NotImplementedError
            # n_latents = self.hparams['n_ae_latents']
            # self.encoding = LinearAEEncoder(n_latents, self.img_size)
            # self.decoding = LinearAEDecoder(n_latents, self.img_size, self.encoding)
        else:
            raise ValueError('"%s" is an invalid model_type' % self.model_type)

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
            - y_hat (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - z (:obj:`torch.Tensor`): sampled latent variable of shape (n_frames, n_latents)
            - mu (:obj:`torch.Tensor`): mean paramter of shape (n_frames, n_latents)
            - logvar (:obj:`torch.Tensor`): logvar paramter of shape (n_frames, n_latents)

        """
        z_s, z_b, z, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        mu = torch.cat([z_s, z_b, z], axis=1)
        if use_mean:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        y_hat = self.encoding.D(z_s)
        return x_hat, z, mu, logvar, y_hat

    def loss(self, datas, dataset=None, accumulate_grad=True, chunk_size=None):
        """Calculate modified ELBO loss for MSPSVAE.

        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.

        Parameters
        ----------
        datas : :obj:`list` of :obj:`dict`
            batch of data; keys should include 'images' and 'masks', if necessary
        datasets : :obj:`list` of :obj:`int`
            used for embedding loss
        accumulate_grad : :obj:`bool`, optional
            accumulate gradient for training step
        chunk_size : :obj:`int`, optional
            deprecated

        Returns
        -------
        :obj:`dict`
            - 'loss' (:obj:`float`): full elbo
            - 'loss_ll' (:obj:`float`): log-likelihood portion of elbo
            - 'loss_kl' (:obj:`float`): kl portion of elbo
            - 'loss_mse' (:obj:`float`): mse (without gaussian constants)
            - 'beta' (:obj:`float`): weight in front of kl term

        """

        if isinstance(datas, list):
            x = torch.cat([data['images'][0] for data in datas], dim=0)
            y = torch.cat([data['labels'][0] for data in datas], dim=0)
            m = torch.cat([data['masks'][0] for data in datas], dim=0) \
                if 'masks' in datas[0] else None
            n = torch.cat([data['labels_masks'][0] for data in datas], dim=0) \
                if 'labels_masks' in datas[0] else None
            datasets = np.concatenate(
                [d * np.ones(datas[d_idx]['images'].shape[1]) for d_idx, d in enumerate(dataset)])
        else:
            x = datas['images'][0]
            y = datas['labels'][0]
            m = datas['masks'][0] if 'masks' in datas else None
            n = datas['labels_masks'][0] if 'labels_masks' in datas else None
            datasets = None

        n_labels = self.hparams['n_labels']
        n_background = self.hparams['n_background']
        # n_latents = self.hparams['n_ae_latents']

        # compute hyperparameters
        alpha = self.hparams['ps_vae.alpha']
        beta = self.beta_vals[self.curr_epoch]
        # gamma = self.hparams['ps_vae.gamma']
        delta = self.hparams['ps_vae.delta']
        kl = self.kl_anneal_vals[self.curr_epoch]

        loss_strs = [
            'loss', 'loss_data_ll', 'loss_label_ll', 'loss_zs_kl', 'loss_zu_mi', 'loss_zu_tc',
            'loss_zu_dwkl',
            # 'loss_AB_orth',
            'loss_triplet']

        loss_dict_vals = {loss: 0 for loss in loss_strs}
        loss_dict_vals['loss_data_mse'] = 0

        x_hat, sample, mu, logvar, y_hat = self.forward(x, dataset=None, use_mean=False)

        # reset losses
        loss_dict_torch = {loss: 0 for loss in loss_strs}

        # data log-likelihood
        loss_dict_torch['loss_data_ll'] = losses.gaussian_ll(x, x_hat, m)
        loss_dict_torch['loss'] -= loss_dict_torch['loss_data_ll']

        # label log-likelihood
        loss_dict_torch['loss_label_ll'] = losses.gaussian_ll(y, y_hat, n)
        loss_dict_torch['loss'] -= alpha * loss_dict_torch['loss_label_ll']

        # supervised latents kl
        loss_dict_torch['loss_zs_kl'] = losses.kl_div_to_std_normal(
            mu[:, :n_labels], logvar[:, :n_labels])
        loss_dict_torch['loss'] += loss_dict_torch['loss_zs_kl']

        # compute all terms of decomposed elbo at once
        index_code_mi, total_correlation, dimension_wise_kl = losses.decomposed_kl(
            sample[:, n_labels + n_background:], mu[:, n_labels + n_background:],
            logvar[:, n_labels + n_background:])

        # unsupervised latents index-code mutual information
        loss_dict_torch['loss_zu_mi'] = index_code_mi
        loss_dict_torch['loss'] += kl * loss_dict_torch['loss_zu_mi']

        # unsupervised latents total correlation
        loss_dict_torch['loss_zu_tc'] = total_correlation
        loss_dict_torch['loss'] += beta * loss_dict_torch['loss_zu_tc']

        # unsupervised latents dimension-wise kl
        loss_dict_torch['loss_zu_dwkl'] = dimension_wise_kl
        loss_dict_torch['loss'] += kl * loss_dict_torch['loss_zu_dwkl']

        # orthogonality between A, B, and C
        # A shape: [n_labels, n_latents]
        # B shape: [n_latents - n_labels - n_background, n_latents]
        # C shape: [n_background, n_latents]
        # compute ||UU^T||^2
        # loss_dict_torch['loss_AB_orth'] = losses.subspace_overlap(
        #     self.encoding.A.weight, self.encoding.B.weight, C=self.encoding.C.weight)
        #
        # loss_dict_torch['loss'] += gamma * loss_dict_torch['loss_AB_orth']

        # triplet loss
        if isinstance(datas, list):
            loss_dict_torch['loss_triplet'] = losses.triplet_loss(
                self.TripletLoss, mu[:, n_labels:n_labels + n_background:], datasets)
            loss_dict_torch['loss'] += delta * loss_dict_torch['loss_triplet']
        else:
            # don't record triplet loss info
            del loss_dict_torch['loss_triplet']
            pass

        if accumulate_grad:
            loss_dict_torch['loss'].backward()

        # get loss values as scalars
        for key, val in loss_dict_torch.items():
            loss_dict_vals[key] += val.item()
        loss_dict_vals['loss_data_mse'] += losses.gaussian_ll_to_mse(
            loss_dict_vals['loss_data_ll'], np.prod(x.shape[1:]))

        # use variance-weighted r2s to ignore small-variance latents
        y_hat_all = y_hat.cpu().detach().numpy()
        y_all = y.cpu().detach().numpy()
        if n is not None:
            n_np = n.cpu().detach().numpy()
            r2 = r2_score(y_all[n_np == 1], y_hat_all[n_np == 1], multioutput='variance_weighted')
        else:
            r2 = r2_score(y_all, y_hat_all, multioutput='variance_weighted')

        # store hyperparams
        loss_dict_vals['alpha'] = alpha
        loss_dict_vals['beta'] = beta
        # loss_dict_vals['gamma'] = gamma
        loss_dict_vals['delta'] = delta
        loss_dict_vals['label_r2'] = r2

        # print(self.encoding.A.weight)
        # print(self.encoding.B.weight)
        # print(self.encoding.C.weight)

        return loss_dict_vals

    def get_predicted_labels(self, x, dataset=None, use_mean=True):
        """Process input data to get predicted labels.

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
            - y_hat (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - z (:obj:`torch.Tensor`): sampled latent variable of shape (n_frames, n_latents)
            - mu (:obj:`torch.Tensor`): mean paramter of shape (n_frames, n_latents)
            - logvar (:obj:`torch.Tensor`): logvar paramter of shape (n_frames, n_latents)

        """
        z_s, _, _, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        if not use_mean:
            z_s = reparameterize(z_s, logvar[:, :self.n_labels])
        y_hat = self.encoding.D(z_s)
        return y_hat

    def get_transformed_latents(self, inputs, dataset=None, as_numpy=True):
        """Return latents after supervised subspace has been transformed to original label space.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor` object
            - image tensor of shape (batch, n_channels, y_pix, x_pix)
            - latents tensor of shape (batch, n_ae_latents)
        dataset : :obj:`int`, optional
            used with session-specific io layers
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in transformed latent space

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)

        # check to see if inputs are images or latents
        if len(inputs.shape) == 2:
            input_type = 'latents'
        else:
            input_type = 'images'

        # get latents in original space
        if input_type == 'images':
            z_s_og, z_b_og, z_og, logvar, _, _ = self.encoding(inputs, dataset=dataset)
        else:
            z_s_og = inputs[:, :self.hparams['n_labels']]
            z_b_og = inputs[:,
                self.hparams['n_labels']:self.hparams['n_labels'] + self.hparams['n_background']]
            z_og = inputs[:, self.hparams['n_labels'] + self.hparams['n_background']:]

        # transform supervised latents to label space
        y_new = self.encoding.D(z_s_og)

        latents_tr = torch.cat([y_new, z_b_og, z_og], axis=1)

        if as_numpy:
            return latents_tr.cpu().detach().numpy()
        else:
            return latents_tr

    def get_inverse_transformed_latents(self, inputs, dataset=None, as_numpy=True):
        """Return latents after they have been transformed using the diagonal mapping D.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor` object
            - image tensor of shape (batch, n_channels, y_pix, x_pix)
            - latents tensor of shape (batch, n_ae_latents) where the first n_labels entries are
              assumed to be labels in the original pixel space
        dataset : :obj:`int`, optional
            used with session-specific io layers
        as_numpy : :obj:`bool`, optional
            True to return as numpy array, False to return as torch Tensor

        Returns
        -------
        :obj:`np.ndarray` or :obj:`torch.Tensor` object
            array of latents in transformed latent space

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs)

        # check to see if inputs are images or latents
        if len(inputs.shape) == 2:
            input_type = 'latents'
        else:
            input_type = 'images'

        # get latents in original space
        if input_type == 'images':
            raise NotImplementedError
        else:
            z_s_og = inputs[:, :self.hparams['n_labels']]
            z_b_og = inputs[:,
                self.hparams['n_labels']:self.hparams['n_labels'] + self.hparams['n_background']]
            z_og = inputs[:, self.hparams['n_labels'] + self.hparams['n_background']:]

        # transform given labels to latent space
        z_s_new = torch.div(torch.sub(z_s_og, self.encoding.D.bias), self.encoding.D.weight)

        latents_tr = torch.cat([z_s_new, z_b_og, z_og], axis=1)

        if as_numpy:
            return latents_tr.cpu().detach().numpy()
        else:
            return latents_tr

    def export_latents(self, data_gen, filename=None):
        """Need to create standard data generator in order to export latents."""

        import os
        import pickle

        from behavenet.data.utils import build_data_generator
        from copy import deepcopy
        hp_new = deepcopy(self.hparams)
        hp_new['n_sessions_per_batch'] = 1  # force standard data generator
        hp_new['train_frac'] = 1  # use all training batches
        hp_new['trial_splits'] = '1;0;0;0'  # no gaps
        data_generator = build_data_generator(hp_new, data_gen.datasets_info)

        self.eval()

        # initialize container for latents
        latents = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            latents[sess] = [np.array([]) for _ in range(dataset.n_trials)]

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.n_tot_batches[dtype]):
                data, sess = data_generator.next_batch(dtype)

                # process batch, perhaps in chunks if full batch is too large to fit on gpu
                chunk_size = 200
                y = data['images'][0]
                batch_size = y.shape[0]
                if batch_size > chunk_size:
                    latents[sess][data['batch_idx'].item()] = np.full(
                        shape=(data['images'].shape[1], self.hparams['n_ae_latents']),
                        fill_value=np.nan)
                    # split into chunks
                    n_chunks = int(np.ceil(batch_size / chunk_size))
                    for chunk in range(n_chunks):
                        # take chunks of size chunk_size, plus overlap due to
                        # max_lags
                        idx_beg = chunk * chunk_size
                        idx_end = np.min([(chunk + 1) * chunk_size, batch_size])
                        y_in = y[idx_beg:idx_end]
                        output = self.encoding(y_in, dataset=sess)
                        curr_latents = torch.cat([output[0], output[1], output[2]], axis=1)
                        latents[sess][data['batch_idx'].item()][idx_beg:idx_end, :] = \
                            curr_latents.cpu().detach().numpy()
                else:
                    y_in = y
                    output = self.encoding(y_in, dataset=sess)
                    curr_latents = torch.cat([output[0], output[1], output[2]], axis=1)
                    latents[sess][data['batch_idx'].item()] = curr_latents.cpu().detach().numpy()

        # save latents separately for each dataset
        filenames = []
        for sess, dataset in enumerate(data_generator.datasets):
            if filename is None:
                # get save name which includes lab/expt/animal/session
                sess_id = str('%s_%s_%s_%s_latents.pkl' % (
                    dataset.lab, dataset.expt, dataset.animal, dataset.session))
                filename_save = os.path.join(
                    self.hparams['expt_dir'], 'version_%i' % self.version, sess_id)
            else:
                filename_save = filename
            # save out array in pickle file
            print('saving latents %i of %i:\n%s' % (
                sess + 1, data_generator.n_datasets, filename_save))
            latents_dict = {'latents': latents[sess], 'trials': dataset.batch_idxs}
            with open(filename_save, 'wb') as f:
                pickle.dump(latents_dict, f)
            filenames.append(filename_save)

        return filenames


class ConvAEPSEncoder(ConvAEEncoder):
    """Convolutional encoder that separates label-related subspace."""

    def __init__(self, hparams):

        from behavenet.models.base import DiagLinear

        super().__init__(hparams)

        # add linear transformations mapping from NN output to label-, non-label-related subspaces
        n_latents = self.hparams['n_ae_latents']
        n_labels = self.hparams['n_labels']
        # NN -> constrained latents
        self.A = nn.Linear(n_latents, n_labels, bias=False)
        # NN -> unconstrained latents
        self.B = nn.Linear(n_latents, n_latents - n_labels, bias=False)
        # constrained latents -> labels (diagonal matrix + bias)
        self.D = DiagLinear(n_labels, bias=True)

        # fix A, B to be orthogonal (and not trainable)
        from scipy.stats import ortho_group
        m = ortho_group.rvs(dim=n_latents).astype('float32')
        with torch.no_grad():
            self.A.weight = nn.Parameter(
                torch.from_numpy(m[:n_labels, :]), requires_grad=False)
            self.B.weight = nn.Parameter(
                torch.from_numpy(m[n_labels:, :]), requires_grad=False)

    def __str__(self):
        """Pretty print encoder architecture."""
        format_str = 'Encoder architecture:\n'
        i = 0
        for module in self.encoder:
            format_str += str('    {:02d}: {}\n'.format(i, module))
            i += 1
        # final ff layer
        format_str += str('    {:02d}: {}\n'.format(i, self.FF))
        # final linear transformations
        format_str += str('    {:02d}: {} (to constrained latents)\n'.format(i, self.A))
        format_str += str('    {:02d}: {} (to unconstrained latents)\n'.format(i, self.B))
        format_str += str('    {:02d}: {} (constrained latents to labels)\n'.format(i, self.D))
        return format_str

    def forward(self, x, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - encoder output y (:obj:`torch.Tensor`): constrained latents (predicted labels) of
              shape (n_labels)
            - encoder output z (:obj:`torch.Tensor`): unconstrained latents of shape
              (n_latents - n_labels)
            - logvar (:obj:`torch.Tensor`): log variance of latents of shape (n_latents)
            - pool_idx (:obj:`list`): max pooling indices for each layer
            - output_size (:obj:`list`): output size for each layer

        """
        # loop over layers, have to collect pool_idx and output sizes if using max pooling to use
        # in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                target_output_size.append(x.size())
                x, idx = layer(x)
                pool_idx.append(idx)
            elif isinstance(layer, nn.ModuleList):
                x = layer[dataset](x)
            else:
                x = layer(x)

        # reshape for ff layer
        x1 = x.view(x.size(0), -1)
        x = self.FF(x1)

        # push through linear transformations
        y = self.A(x)  # NN -> constrained latents
        w = self.B(x)  # NN -> unconstrained latents

        return y, w, self.logvar(x1), pool_idx, target_output_size


class ConvAEMSPSEncoder(ConvAEEncoder):
    """Convolutional encoder that separates label-related subspace."""

    def __init__(self, hparams):

        from behavenet.models.base import DiagLinear

        super().__init__(hparams)

        # add linear transformations mapping from NN output to label-, non-label-related subspaces
        n_latents = self.hparams['n_ae_latents']
        n_labels = self.hparams['n_labels']
        n_background = self.hparams['n_background']

        # NN -> supervised latents
        self.A = nn.Linear(n_latents, n_labels, bias=False)
        # NN -> unsupervised latents
        self.B = nn.Linear(n_latents, n_latents - n_labels - n_background, bias=False)
        # NN -> background latents
        self.C = nn.Linear(n_latents, n_background, bias=True)
        # supervised latents -> labels (diagonal matrix + bias)
        self.D = DiagLinear(n_labels, bias=True)

        # fix A, B, C to be orthogonal (and not trainable)
        from scipy.stats import ortho_group
        m = ortho_group.rvs(dim=n_latents).astype('float32')

        with torch.no_grad():
            self.A.weight = nn.Parameter(
                torch.from_numpy(m[:n_labels, :]), requires_grad=False)
            self.B.weight = nn.Parameter(
                torch.from_numpy(m[n_labels + n_background:, :]), requires_grad=False)
            self.C.weight = nn.Parameter(
                torch.from_numpy(m[n_labels:n_labels + n_background, :]), requires_grad=False)

    def __str__(self):
        """Pretty print encoder architecture."""
        format_str = 'Encoder architecture:\n'
        i = 0
        for module in self.encoder:
            format_str += str('    {:02d}: {}\n'.format(i, module))
            i += 1
        # final ff layer
        format_str += str('    {:02d}: {}\n'.format(i, self.FF))
        # final linear transformations
        format_str += str('    {:02d}: {} (to supervised latents)\n'.format(i, self.A))
        format_str += str('    {:02d}: {} (to unsupervised latents)\n'.format(i, self.B))
        format_str += str('    {:02d}: {} (to background latents)\n'.format(i, self.C))
        format_str += str('    {:02d}: {} (supervised latents to labels)\n'.format(i, self.D))
        return format_str

    def forward(self, x, dataset=None):
        """Process input data.

        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers

        Returns
        -------
        :obj:`tuple`
            - encoder output y (:obj:`torch.Tensor`): constrained latents (predicted labels) of
              shape (n_labels)
            - encoder output z (:obj:`torch.Tensor`): unconstrained latents of shape
              (n_latents - n_labels)
            - logvar (:obj:`torch.Tensor`): log variance of latents of shape (n_latents)
            - pool_idx (:obj:`list`): max pooling indices for each layer
            - output_size (:obj:`list`): output size for each layer

        """
        # loop over layers, have to collect pool_idx and output sizes if using max pooling to use
        # in unpooling
        pool_idx = []
        target_output_size = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                target_output_size.append(x.size())
                x, idx = layer(x)
                pool_idx.append(idx)
            elif isinstance(layer, nn.ModuleList):
                x = layer[dataset](x)
            else:
                x = layer(x)

        # reshape for ff layer
        x1 = x.view(x.size(0), -1)
        x = self.FF(x1)

        # push through linear transformations
        z_s = self.A(x)  # NN -> supervised latents
        z = self.B(x)  # NN -> unsupervised latents
        z_b = self.C(x)  # NN -> background latents

        return z_s, z_b, z, self.logvar(x1), pool_idx, target_output_size
