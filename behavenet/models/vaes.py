"""Variational autoencoder models implemented in PyTorch."""

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import nn

import behavenet.fitting.losses as losses
from behavenet.models.aes import AE, ConvAEDecoder, ConvAEEncoder

# to ignore imports for sphix-autoapidoc
__all__ = ['reparameterize', 'VAE', 'BetaTCVAE', 'SSSVAE', 'ConvAESSSEncoder']


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
        if anneal_epochs > 0:
            self.beta_vals = np.append(
                np.linspace(1, beta, anneal_epochs),
                beta * np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
        else:
            self.beta_vals = beta * np.ones(hparams['max_n_epochs'] + 1)

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
            loss_dict_torch['loss'] += loss_dict_torch['loss_mi']

            # unsupervised latents total correlation
            loss_dict_torch['loss_tc'] = total_correlation
            loss_dict_torch['loss'] += beta * loss_dict_torch['loss_tc']

            # unsupervised latents dimension-wise kl
            loss_dict_torch['loss_dwkl'] = dimension_wise_kl
            loss_dict_torch['loss'] += loss_dict_torch['loss_dwkl']

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


class SSSVAE(AE):
    """Semi-supervised subspace variational autoencoder class.

    This class constructs a VAE that...

    """

    def __init__(self, hparams):
        """See constructor documentation of AE for hparams details.

        Parameters
        ----------
        hparams : :obj:`dict`
            in addition to the standard keys, must also contain:
            - 'n_labels' (:obj:`n_labels`)
            - 'sss.alpha' (:obj:`float`)
            - 'sss.beta' (:obj:`float`)
            - 'sss.gamma' (:obj:`float`)

        """

        if hparams['model_type'] == 'linear':
            raise NotImplementedError
        if hparams['n_ae_latents'] < hparams['n_labels']:
            raise ValueError('AEMSP model must contain at least as many latents as labels')

        self.n_latents = hparams['n_ae_latents']
        self.n_labels = hparams['n_labels']

        hparams['variational'] = True
        super().__init__(hparams)

        # # set up kl annealing
        # anneal_epochs = self.hparams.get('vae.beta_anneal_epochs', 0)
        # self.curr_epoch = 0  # must be modified by training script
        # if anneal_epochs > 0:
        #     self.beta_vals = np.append(
        #         np.linspace(0, hparams['vae.beta'], anneal_epochs),
        #         np.ones(hparams['max_n_epochs'] + 1))  # sloppy addition to fully cover rest
        # else:
        #     self.beta_vals = np.ones(hparams['max_n_epochs'] + 1)

    def build_model(self):
        """Construct the model using hparams."""
        self.hparams['hidden_layer_size'] = self.hparams['n_ae_latents']
        if self.model_type == 'conv':
            self.encoding = ConvAESSSEncoder(self.hparams)
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
        y, w, logvar, pool_idx, outsize = self.encoding(x, dataset=dataset)
        mu = torch.cat([y, w], axis=1)
        if use_mean:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        x_hat = self.decoding(z, pool_idx, outsize, dataset=dataset)
        y_hat = self.encoding.D(y)
        return x_hat, y_hat, z, mu, logvar

    def loss(self, data, dataset=0, accumulate_grad=True, chunk_size=200):
        """Calculate modified ELBO loss for SSSVAE.

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
        batch_size = x.shape[0]
        n_chunks = int(np.ceil(batch_size / chunk_size))
        n_labels = self.hparams['n_labels']
        n_latents = self.hparams['n_latents']

        # beta = self.beta_vals[self.curr_epoch]
        alpha = self.hparams['sss.alpha']
        beta = self.hparams['sss.beta']
        gamma = self.hparams['sss.gamma']

        loss_strs = [
            'loss', 'loss_data_ll', 'loss_label_ll', 'loss_zs_kl', 'loss_zu_mi', 'loss_zu_tc',
            'loss_zu_kl', 'loss_AB_orth']

        loss_dict_vals = {loss: 0 for loss in loss_strs}
        loss_dict_vals['loss_data_mse'] = 0

        y_hat_all = []

        for chunk in range(n_chunks):

            idx_beg = chunk * chunk_size
            idx_end = np.min([(chunk + 1) * chunk_size, batch_size])

            x_in = x[idx_beg:idx_end]
            y_in = y[idx_beg:idx_end]
            m_in = m[idx_beg:idx_end] if m is not None else None
            x_hat, y_hat, sample, mu, logvar = self.forward(x_in, dataset=dataset, use_mean=False)

            # reset losses
            loss_dict_torch = {loss: 0 for loss in loss_strs}

            # data log-likelihood
            loss_dict_torch['loss_data_ll'] = losses.gaussian_ll(x_in, x_hat, m_in)
            loss_dict_torch['loss'] -= loss_dict_torch['loss_data_ll']

            # label log-likelihood
            loss_dict_torch['loss_label_ll'] = losses.gaussian_ll(y_in, y_hat)
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
            loss_dict_torch['loss'] += loss_dict_torch['loss_zu_mi']

            # unsupervised latents total correlation
            loss_dict_torch['loss_zu_tc'] = total_correlation
            loss_dict_torch['loss'] += beta * loss_dict_torch['loss_zu_tc']

            # unsupervised latents dimension-wise kl
            loss_dict_torch['loss_zu_kl'] = dimension_wise_kl
            loss_dict_torch['loss'] += loss_dict_torch['loss_zu_kl']

            # orthogonality between A and B
            loss_dict_torch['loss_AB_orth'] = losses.subspace_overlap(
                self.encoding.A, self.encoding.B)
            loss_dict_torch['loss'] += gamma * loss_dict_torch['loss_AB_orth']

            if accumulate_grad:
                loss_dict_torch['loss'].backward()

            # get loss value (weighted by batch size)
            bs = idx_end - idx_beg
            for key, val in loss_dict_torch.items():
                loss_dict_vals[key] += val.item() * bs
            loss_dict_vals['loss_data_mse'] += losses.gaussian_ll_to_mse(
                loss_dict_vals['loss_data_ll'], np.prod(x.shape[1:])) * bs

            # collect predicted labels to compute R2
            y_hat_all.append(y_hat.cpu().detach().numpy())

        # use variance-weighted r2s to ignore small-variance latents
        y_hat_all = np.concatenate(y_hat_all, axis=0)
        r2 = r2_score(y.cpu().detach().numpy(), y_hat_all, multioutput='variance_weighted')

        # compile (properly weighted) loss terms
        for key in loss_dict_vals.keys():
            loss_dict_vals[key] /= batch_size
        # store hyperparams
        loss_dict_vals['alpha'] = alpha
        loss_dict_vals['beta'] = beta
        loss_dict_vals['gamma'] = gamma
        loss_dict_vals['label_r2'] = r2

        return loss_dict_vals


class ConvAESSSEncoder(ConvAEEncoder):
    """Convolutional encoder that separates label-related subspace."""

    def __init__(self, hparams):

        super().__init__(hparams)

        # add linear transformations mapping from NN output to label-, non-label-related subspaces
        n_latents = self.hparams['n_ae_latents']
        n_labels = self.hparams['n_labels']
        # NN -> constrained latents
        self.A = nn.Linear(n_latents, n_labels, bias=False)
        # NN -> unconstrained latents
        self.B = nn.Linear(n_latents, n_latents - n_labels, bias=False)
        # unconstrained latents -> labels
        self.D = nn.Linear(n_labels, n_labels)

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
        x = x.view(x.size(0), -1)
        x = self.FF(x)

        # push through linear transformations
        y = self.A(x)
        z = self.B(x)

        return y, z, self.logvar(x), pool_idx, target_output_size
