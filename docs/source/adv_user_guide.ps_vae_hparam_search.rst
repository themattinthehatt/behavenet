.. _psvae_hparams:

PS-VAE hyperparameter search guide
===================================

The PS-VAE objective function :math:`\mathscr{L}_{\text{PS-VAE}}` is comprised of several
different terms:

.. math::

    \mathscr{L}_{\text{PS-VAE}} =
    \mathscr{L}_{\text{frames}} +
    \alpha \mathscr{L}_{\text{labels}} +
    \mathscr{L}_{\text{KL-s}} +
    \mathscr{L}_{\text{ICMI}} +
    \beta \mathscr{L}_{\text{TC}} +
    \mathscr{L}_{\text{DWKL}} +
    \gamma \mathscr{L}_{\text{orth}}

where

 * :math:`\mathscr{L}_{\text{frames}}`: log-likelihood of the video frames
 * :math:`\mathscr{L}_{\text{labels}}`: log-likelihood of the labels
 * :math:`\mathscr{L}_{\text{KL-s}}`: KL divergence of the supervised latents
 * :math:`\mathscr{L}_{\text{ICMI}}`: index-code mutual information of the unsupervised latents
 * :math:`\mathscr{L}_{\text{TC}}`: total correlation of the unsupervised latents
 * :math:`\mathscr{L}_{\text{DWKL}}`: dimension-wise KL of the unsupervised latents
 * :math:`\mathscr{L}_{\text{orth}}`: orthogonality of the full latent space (supervised + unsupervised)

There are three important hyperparameters of the model that we address below: :math:`\alpha`, which
weights the reconstruction of the labels; :math:`\beta`, which weights the factorization of the
unsupervised latent space; and :math:`\gamma`, which weights the orthogonality of the entire latent
space. The purpose of this guide is to propose a series of model fits that efficiently explores
this space of hyperparameters, as well as point out several BehaveNet plotting utilities to assist
in this exploration.


How to select :math:`\alpha`
----------------------------
The hyperparameter :math:`\alpha` controls the strength of the label log-likelihood term, which
needs to be balanced against the frame log-likelihood term. We first recommend z-scoring each
individual label, which removes the scale of the labels as a confound. We then recommend fitting
models with a range of :math:`\alpha` values, while setting the defaults :math:`\beta=1` (no extra
weight on the total correlation term) and :math:`\gamma=0` (no constraint on orthogonality). In our
experience the range :math:`\alpha=[50, 100, 500, 1000]` is a reasonable range to start with. The
"best" value for :math:`\alpha` is subjective because it involves a tradeoff between pixel
log-likelihood (or the related mean square error, MSE) and label log-likelihood (or MSE).
After choosing a suitable value, we will fix :math:`\alpha` and vary :math:`\beta` and
:math:`\gamma`.


How to select :math:`\beta` and :math:`\gamma`
----------------------------------------------
The choice of :math:`\beta` and :math:`\gamma` is more difficult because there does not yet exist
a single robust measure of "disentanglement" that can tell us which models learn a suitable
unsupervised representation. Instead we will fit models with a range of hypeparameters, then use
a quantitative metric to guide a qualitative analysis.

A reasonable range to start with is :math:`\beta=[1, 5, 10, 20]` and :math:`\gamma=1000`. While it
is possible to extend the range for :math:`\gamma`, we have found :math:`\gamma=1000` to work for
many datasets. How, then, do we choose a good value for :math:`\beta`? Currently our best advice is
to compute the correlation of the training data across all pairs of unsupervised dimensions. The
value of :math:`\beta` that minimizes the average of the pairwise correlations is a good place to
start more qualitative evaluations.

Ultimately, the choice of the "best" model comes down to a qualitative evaluation, the *latent
traversal*. A latent traversal is the result of changing the value of a latent dimension while
keeping the value of all other latent dimensions fixed. If the model has learned an interpretable
representation then the resulting generated frames should show one single behavioral feature
changing per dimension - an arm, or a jaw, or the chest (see :ref:`below<ps_vae_plotting>`
for more information on tools
for constructing and visualizing these traversals). In order to choose the "best" model, we perform
these latent traversals for all values of :math:`\beta` and look at the resulting latent traversal
outputs. The model with the (subjectively) most interpretable dimensions is then chosen.


A note on model robustness
--------------------------
We have found the PS-VAE to be somewhat sensitive to initialization of the neural network
parameters. We also recommend choosing the set of hyperparamters with the lowest pairwise
correlations and refitting the model with several random seeds (by changing the ``rng_seed_model``
parameter of the ``ae_model.json`` file), which may lead to even better results.

.. _ps_vae_plotting:

Tools for investigating PS-VAE model fits
------------------------------------------
The functions listed below are provided in the BehaveNet plotting module (
:mod:`behavenet.plotting`) to facilitate model checking and comparison at different stages.

Hyperparameter search visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function :func:`behavenet.plotting.cond_ae_utils.plot_hyperparameter_search_results` creates
a variety of diagnostic plots after the user has performed the :math:`\alpha` search and the
:math:`\beta/\gamma` search detailed above:

- pixel mse as a function of :math:`\alpha`, num latents (for fixed :math:`\beta, \gamma`)
- label mse as a function of :math:`\alpha`, num_latents (for fixed :math:`\beta, \gamma`)
- pixel mse as a function of :math:`\beta, \gamma` (for fixed :math:`\alpha`, n_ae_latents)
- label mse as a function of :math:`\beta, \gamma` (for fixed :math:`\alpha`, n_ae_latents)
- index-code mutual information (part of the KL decomposition) as a function of
  :math:`\beta, \gamma` (for fixed :math:`\alpha`, n_ae_latents)
- total correlation(part of the KL decomposition) as a function of :math:`\beta, \gamma`
  (for fixed :math:`\alpha`, n_ae_latents)
- dimension-wise KL (part of the KL decomposition) as a function of :math:`\beta, \gamma`
  (for fixed :math:`\alpha`, n_ae_latents)
- average correlation coefficient across all pairs of unsupervised latent dims as a function of
  :math:`\beta, \gamma` (for fixed :math:`\alpha`, n_ae_latents)
- subspace overlap computed as :math:`||[A; B] - I||_2^2` for :math:`A, B` the projections to the
  supervised and unsupervised subspaces, respectively, and :math:`I` the identity - as a function
  of :math:`\beta, \gamma` (for fixed :math:`\alpha`, n_ae_latents)
- example subspace overlap matrix for :math:`\gamma=0` and :math:`\beta=1`, with fixed
  :math:`\alpha`, n_ae_latents
- example subspace overlap matrix for :math:`\gamma=1000` and :math:`\beta=1`, with fixed
  :math:`\alpha`, n_ae_latents

These plots help with the selection of hyperparameter settings.

Model training curves
^^^^^^^^^^^^^^^^^^^^^
The function :func:`behavenet.plotting.cond_ae_utils.plot_psvae_training_curves` creates training
plots for each term in the PS-VAE objective function for a *single* model:

- total loss
- pixel mse
- label R^2 (note the objective function contains the label MSE, but R^2 is easier to parse)
- KL divergence of supervised latents
- index-code mutual information of unsupervised latents
- total correlation of unsupervised latents
- dimension-wise KL of unsupervised latents
- subspace overlap

A function argument allows the user to plot either training or validation curves. These plots allow
the user to check whether or not models have trained to completion.

Label reconstruction
^^^^^^^^^^^^^^^^^^^^
The function :func:`behavenet.plotting.cond_ae_utils.plot_label_reconstructions` creates a series
of plots that show the true labels and their PS-VAE reconstructions for a given list of batches.
These plots are useful for qualitatively evaluating the supervised subspace of the PS-VAE;
a quantitative evaluation (the label MSE) can be found in the ``metrics.csv`` file created in the
model folder during training.

Latent traversals: plots
^^^^^^^^^^^^^^^^^^^^^^^^
The function :func:`behavenet.plotting.cond_ae_utils.plot_latent_traversals` displays video frames
representing the traversal of chosen dimensions in the latent space. This function uses a
single base frame to create all traversals.

Latent traversals: movies
^^^^^^^^^^^^^^^^^^^^^^^^^
The function :func:`behavenet.plotting.cond_ae_utils.make_latent_traversal_movie` creates a
multi-panel movie with each panel showing traversals of an individual latent dimension.
The traversals will start at a lower bound, increase to an upper bound, then return to a lower
bound; the traversal of each dimension occurs simultaneously. It is also possible to specify
multiple base frames for the traversals; the traversal of each base frame is separated by
several blank frames.
