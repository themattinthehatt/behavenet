Introduction
============

The command line interface
--------------------------

json files

see :ref:`hyperparameters glossary<glossary>` for more info 


Organizing model fits with test-tube
------------------------------------

Brief explanation of test-tube

what are test tube experiments? what are test tube versions?

Some of the files:

* **best_val_model.pt**: the best CAE as determined by computing the loss on validation data
* **meta_tags.csv**: hyperparameters associated with data, computational resources, and model
* **metrics.csv**: metrics computed on dataset as a function of epochs; the default is that metrics are computed on training and validation data every epoch (and reported as a mean over all batches) while metrics are computed on test data only at the end of training using the best model (and reported per batch).
* **lab-id_expt-id_animal-id_session-id_latents.pkl**: list of np.ndarrays of CAE latents computed using the best model
* **session_info.csv**: sessions used to fit the model
