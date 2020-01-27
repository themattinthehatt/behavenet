ARHMMs
======

The next step of the BehaveNet pipeline is to model the low-dimensional representation of behavior with a simple class of nonlinear dynamical systems called autoregressive hidden Markov models (ARHMMs). An ARHMM models the sequence of continuous latents as a stochastic process that switches between a small number K of discrete states, each characterized by linear-Gaussian dynamics. These discrete state variables also exhibit temporal dependences through Markovian dynamics - the discrete state at time t may depend on its preceding value.

Fitting a single ARHMM is very similar to the AE fitting procedure; first copy the example json files ``arhmm_compute.json``, ``arhmm_model.json``, and ``arhmm_training.json`` into your ``.behavenet`` directory, ``cd`` to the ``behavenet`` directory in the terminal, and run:

.. code-block:: console

    $: python behavenet/fitting/arhmm_grid_search.py --data_config /user_home/.behavenet/musall_vistrained_params.json --model_config /user_home/.behavenet/arhmm_model.json --training_config /user_home/.behavenet/arhmm_training.json --compute_config /user_home/.behavenet/arhmm_compute.json
