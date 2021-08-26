.. _msps_vae:

Multi-session PS-VAE
====================

The :ref:`Partitioned Subspace VAE (PS-VAE)<ps_vae>` (see preprint
`here <https://www.biorxiv.org/content/10.1101/2021.02.22.432309v2>`_)
finds a low-dimensional latent representation of a single behavioral video that is partitioned into
two subspaces: a supervised subspace that reconstructs user-provided labels, and an unsupervised
subspace that captures remaining variability. In practice, though, we will typically want to
produce a low-dimensional latent representation that is shared across multiple experimental
sessions, rather than fitting session-specific models.  However, the inclusion of multiple videos
during training introduces a new problem: different videos from the same experimental setup will
contain variability in the experimental equipment, lighting, or even physical differences between
animals, despite efforts to standardize these features. We do not want these differences (which we
refer to collectively as the “background”) to contaminate the latent representation, as they do not
contain the behavioral information we wish to extract for downstream analyses.

To address this issue within the framework of the PS-VAE, we introduce a new subspace into our
model which captures static differences between sessions (the “background” subspace) while leaving
the other subspaces (supervised and unsupervised) to capture dynamic behaviors.

As with the PS-VAE, the data HDF5 needs to be augmented to include a new HDF5 group named
``labels``, which contains an HDF5 dataset for each trial. The labels for each trial must match up
with the corresponding video frames; for example, if the image data in ``images/trial_0013``
contains 100 frames (a numpy array of shape [100, n_channels, y_pix, x_pix]), the label data in
``labels/trial_0013`` should contain the corresponding labels (a numpy array of shape
[100, n_labels]). See the :ref:`data structure documentation<data_structure>` for more
information. Also see the documentation for
:ref:`fitting models on multiple sessions<multisession>` for more information on how to specify
which sessions are used for fitting.

To fit an MSPS-VAE with the default CAE BehaveNet architecture, edit the ``model_class``,
``ps_vae.alpha``, ``ps_vae.beta``, ``ps_vae.delta``, ``n_background``, ``n_sessions_per_batch`` and
``ps_vae.anneal_epochs`` parameters of the ``ae_model.json`` file:

.. code-block:: json

    {
    "experiment_name": "ae-example",
    "model_type": "conv",
    "n_ae_latents": 12,
    "l2_reg": 0.0,
    "rng_seed_model": 0,
    "fit_sess_io_layers": false,
    "ae_arch_json": null,
    "model_class": "msps-vae",
    "ps_vae.alpha": 1000,
    "ps_vae.beta": 10,
    "ps_vae.delta": 50,
    "ps_vae.anneal_epochs": 100,
    "n_background": 3,
    "n_sessions_per_batch": 2,
    "conditional_encoder": false
    }

The ``n_background`` parameter sets the dimensionality of the background subspace; we find 3 works
well in practice. The ``n_sessions_per_batch`` parameter determines how many many sessions comprise
a single batch during training; this value should be greater than 1 for the triplet loss to work.
The current implementation supports values of ``n_sessions_per_batch = [2, 3, 4]``.

To fit the model, use the ``ae_grid_search.py`` function using this updated model json. All
other input jsons remain unchanged. See the :ref:`hyperparameter search guide<mspsvae_hparams>` for
information on how to efficiently search over the ``ps_vae.alpha``, ``ps_vae.beta``, and
``ps_vae.delta`` hyperparameters.
