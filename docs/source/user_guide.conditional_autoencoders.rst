.. _conditional_aes:

Conditional autoencoders
========================

One drawback to the use of unsupervised dimensionality reduction (performed by the convolutional
autoencoder) is that the resulting latents are generally uninterpretable, because any animal
movement in a behavioral video will be represented across many (if not all) of the latents. Thus
there is no simple way to find an "arm" dimension that is separate from a "pupil" dimension,
distinctions that may be important for downstream analyses.

Semi-supervised approaches to dimensionality reduction offer a partial resolution to this problem.
In this framework, the user first collects a set of markers that track body parts of interest over
time. These markers can be, for example, the output of standard pose estimation software such as
`DeepLabCut <http://www.mousemotorlab.org/deeplabcut>`_, `LEAP <https://github.com/talmo/leap>`_,
or `DeepPoseKit <https://github.com/jgraving/deepposekit>`_. These markers can then be used to
augment the latent space (using :ref:`conditional autoencoders<cond_ae>`) or regularize the latent
space (using the :ref:`matrix subspace projection loss<ae_msp>`), both of which are described
below.

In order to fit these models, the data HDF5 needs to be augmented to include a new HDF5 group named
``labels``, which contains an HDF5 dataset for each trial. The labels for each trial must match up
with the corresponding video frames; for example, if the image data in ``images/trial_0013``
contains 100 frames (a numpy array of shape [100, n_channels, y_pix, x_pix]), the label data in
``labels/trial_0013`` should contain the corresponding labels (a numpy array of shape
[100, n_labels]). See the :ref:`data structure documentation<data_structure>` for more
information).

.. _cond_ae:

Conditional autoencoders
------------------------
The `conditional autoencoder <https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf>`_
implemented in BehaveNet is a simple extension of the convolutional autoencoder. Each frame is
pushed through the encoder to produce a set of latents, which are concatenated with the
corresponding labels; this augmented vector is then used as input to the decoder.

To fit a single conditional autoencoder with the default CAE BehaveNet architecture, edit the
``model_class`` parameter of the ``ae_model.json`` file:

.. code-block:: json

    {
    "experiment_name": "ae-example",
    "model_type": "conv",
    "n_ae_latents": 12,
    "l2_reg": 0.0,
    "rng_seed_model": 0,
    "fit_sess_io_layers": false,
    "ae_arch_json": null,
    "model_class": "cond-ae",
    "conditional_encoder": false
    }

Then to fit the model, use the ``ae_grid_search.py`` function using this updated model json. All
other input jsons remain unchanged.

By concatenating the labels to the latents, we are learning a conditional decoder. We can also
condition the latents on the labels by learning a conditional encoder. Turning on this feature
requires an additional HDF5 group; documentation coming soon.


.. _ae_msp:

Matrix subspace projection loss
-------------------------------
An alternative way to obtain a more interpretable latent space is to encourage a subspace to
predict the labels themselves, rather than appending them to the latents. With appropriate
additions to the loss function, we can ensure that the subspace spanned by the label-predicting
latents is orthogonal to the subspace spanned by the remaining unconstrained latents. This is the
idea of the `matrix subspace projection loss <https://arxiv.org/pdf/1907.12385.pdf>`_.

For example, imagine we are tracking 4 body parts, each with their own x-y coordinates for each
frame. This gives us 8 dimensions of behavior to predict. If we fit a CAE with 10 latent
dimensions, we will use 8 of those dimensions to predict the 8 marker dimensions - one latent
dimension for each marker dimension. This leaves 2 unconstrained dimensions to predict remaining
variability in the images not captured by the labels. The model is trained by minimizing the mean
square error between the true and predicted images, as well as the true and predicted labels.
Unlike the conditional autoencoder described above, this new loss function has an additional
hyperparameter that governs the tradeoff between image reconstruction and label reconstruction.

To fit a single autoencoder with the matrix subspace projection loss (and the default CAE BehaveNet
architecture), edit the ``model_class`` and ``msp.alpha`` parameters of the ``ae_model.json`` file:

.. code-block:: json

    {
    "experiment_name": "ae-example",
    "model_type": "conv",
    "n_ae_latents": 12,
    "l2_reg": 0.0,
    "rng_seed_model": 0,
    "fit_sess_io_layers": false,
    "ae_arch_json": null,
    "model_class": "cond-ae-msp",
    "msp.alpha": 1e-4,
    "conditional_encoder": false
    }

The ``msp.alpha`` parameter needs to be tuned for each dataset, but ``msp.alpha=1.0`` is a
reasonable starting value if the labels have each been z-scored.

.. note::
    
    The matrix subspace projection model implemented in BehaveNet learns a linear mapping from the
    original latent space to the predicted labels that **does not contain a bias term**. Therefore
    you should center each label before adding them to the HDF5 file. Additionally, normalizing
    each label by its standard deviation can make searching across msp weights less dependent on
    the size of the input image.

Then to fit the model, use the ``ae_grid_search.py`` function using this updated model json. All
other input jsons remain unchanged.
