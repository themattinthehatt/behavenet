A checklist for adding a new model to the BehaveNet package:
===

Model-related code
---

* define a new class in `behavenet.models` package; 
    * add to `behavenet.models.__init__.py` imports
* add new `FitMethod` class to `behavenet.fitting.training.py`
    * add to `__all__` variable at top of file for proper sphinx parsing
* add model to relevant grid search script, i.e. `behavenet.fitting.ae_grid_search.py` (or create a new one)
* required function updates [UPDATE UNIT TESTS!!]:
    * `behavenet.data.utils.get_data_generator_inputs`
    * `behavenet.fitting.utils.get_expt_dir`
    * `behavenet.fitting.utils.get_model_params`
    * `behavenet.fitting.eval.export_xxx`
* potential function updates:
    * other `behavenet.fitting.eval` methods
    * `behavenet.fitting.hyperparam_utils.add_dependent_params`
* update relevant jsons (e.g. extra hyperparameters)
    
    
Testing
---

* add new model to integration script `tests/integration.py`
    * add to `model_classes`, `model_files`, and `sessions` lists in `main()` 
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