# BehaveNet Testing

### Unit testing

BehaveNet has partial unit testing coverage. The package uses the `pytest` package for unit testing. This package must be installed separately (in the BehaveNet conda environment) and can be run from the command line within the top-level `behavenet` directory:

```bash
(behavenet) $: pytest
```

where `(behavenet)` indicates the shell is in the `behavenet` conda environment.

As of April 2020 most helper functions have been unit tested, though modeling code (i.e. `pytorch` code) has not.

### Integration testing

BehaveNet also has a rudimentary integration test. This test requires the Musall (WFCI) data to be downloaded first (see instructions [here](../example/00_data.ipynb)). Then, from the top-level `behavenet` directory, run the following from the command line:

```bash
(behavenet) $: python tests/integration.py
```

The integration test will 

1. create a temporary directory
2. fit an autoencoder
3. fit several arhmms
4. fit several neural-ae decoders
5. fit several neural-arhmm decoders
6. delete the temporary directory

The integration test checks that all models finished training. Models are only fit for a single epoch using a small fraction of the data, so total fit time should be less than one minute (if using a GPU to fit the autoencoder). The purpose of the integration test is to ensure that both `pytorch` and `ssm` models are fitting properly, and that all path handling functions linking outputs of one model to inputs of another are working.