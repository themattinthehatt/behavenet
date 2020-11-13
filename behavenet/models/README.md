A checklist for adding a new model to the BehaveNet package:
===

Model-related code
---

* define a new class in `behavenet.models` package
    * note that the key-value pairs returned in the `loss` method are the values logged to csv throughout training
    * add to `behavenet.models.__init__.py` imports
* add model to relevant grid search script for model construction, i.e. `behavenet.fitting.ae_grid_search.py` (or create a new one)
* required function updates:
    * `behavenet.data.utils.get_data_generator_inputs` [UPDATE UNIT TEST!]
    * `behavenet.fitting.utils.get_expt_dir` [UPDATE UNIT TEST!]
    * `behavenet.fitting.utils.get_model_params` [UPDATE UNIT TEST!]
    * `behavenet.fitting.utils.get_best_data_and_model`
    * `behavenet.fitting.eval.export_xxx` (latents, states, predictions, etc)
* potential function updates:
    * other `behavenet.fitting.eval` methods (like `get_reconstruction`)
    * `behavenet.fitting.hyperparam_utils.add_dependent_params` [UPDATE UNIT TEST!]
* update relevant jsons (e.g. extra hyperparameters)


Testing
---

* add new model to integration script `tests/integration.py`
    * add to `MODELS_TO_FIT` list at top of file 
    * update `get_model_config_files()`
    * update `define_new_config_values()`
*  run tests
    * unit tests: from behavenet parent directory run `pytest`
    * integration test: from behavenet parent directory run `python tests/integration.py`


Documentation
---

* complete all docstrings in new functions
* add new user guide if necessary (`docs/source/user_guide.[new_model].rst`)
* update data structure docs if adding to hdf5 (`docs/source/data_structure.rst`)
* add new hyperparams to glossary (`docs/source/glossary.rst`)
* compile documentation: from the command line in `behavenet/docs`, run: `make html`
* [optional] add new jupyter notebook in `behavenet/examples` directory; useful if you have developed lots of new analysis tools as well