# How to contribute

If you're interested in contributing to the behavenet package, please contact the project 
developer Matt Whiteway at m.whiteway ( at ) columbia.edu.

If you would like to add a new Pytorch model to the package, you can find more detailed information
[here](behavenet/models/README.md).
 
Before submitting a pull request, please follow these steps:

## Style

The behavenet package follows the PEP8 style guidelines, and allows for line lengths of up to 99 
characters. To ensure that your code matches these guidelines, please flake your code using the 
provided configuration file. You will need to first install flake8 in the behavenet conda 
environment:

```bash
(behavenet) $: pip install flake8
```

Once all code, tests, and documentation are in place, you can run the flaker from from the project
directory:

```bash
(behavenet) $: flake8
```

## Documentation

Behavenet uses Sphinx and readthedocs to provide documentation to developers and users. 

* complete all docstrings in new functions using google's style (see source code for examples)
* provide inline comments when necessary; the more the merrier
* add a new user guide if necessary (`docs/source/user_guide.[new_model].rst`)
* update data structure docs if adding to hdf5 (`docs/source/data_structure.rst`)
* add new hyperparams to glossary (`docs/source/glossary.rst`)

To check the documentation, you can compile it on your local machine first. To do so you will need
to first install sphinx in the behavenet conda environment:

```bash
(behavenet) $: pip install sphinx==3.2.0 sphinx_rtd_theme==0.4.3 sphinx-automodapi==0.12 
```

To compile the documentation, from the behavenet project directory cd to `behavenet/docs` and run 
the make file:

```bash
(behavenet) $: cd docs
(behavenet) $: make html 
```

## Testing

Behavenet uses pytest to unit test the package; in addition, there is an integration script 
provided to ensure the interlocking pieces play nicely. Please write unit tests for all new 
(non-plotting) functions, and if you updated any existing functions please update the corresponding
unit tests.

To run the unit tests, first install pytest in the behavenet conda environment:

```bash
(behavenet) $: pip install pytest
```

Then, from the project directory, run:

```bash
(behavenet) $: pytest
```

To run the integration script:

```bash
(behavenet) $: python tests/integration.py
```

Running the integration test will take approximately 1 minute with a GPU.
