# BehaveNet Testing

### Unit testing

BehaveNet has partial unit testing coverage. The package uses the `pytest` package for unit testing. This package must be installed separately (in the BehaveNet conda environment) and can be run from the command line within the top-level `behavenet` directory:

```bash
(behavenet) $: pytest
```

where `(behavenet)` indicates the shell is in the `behavenet` conda environment.

As of June 2020 most helper functions have been unit tested, though modeling code (i.e. `pytorch` code) has not.

### Integration testing

BehaveNet also has a rudimentary integration test. From the top-level `behavenet` directory, run the following from the command line:

```bash
(behavenet) $: python tests/integration.py
```

The integration test will 

1. create temporary data/results directories
2. create simulated data
3. fit the following models:
    * autoencoder
    * several arhmms
    * several neural-ae decoders
    * several neural-arhmm decoders
    * autoencoder on multiple sessions
    * variational autoencoder
    * autoencoder with matrix subspace projection loss
    * labels -> images convolutional decoder
4. delete the temporary data/directories

The integration test checks that all models finished training. 
Models are only fit for a single epoch with a small amount of data, so total fit time should be around one minute (if using a GPU to fit the autoencoders). 
The purpose of the integration test is to ensure that both `pytorch` and `ssm` models are fitting properly, and that all path handling functions linking outputs of one model to inputs of another are working.