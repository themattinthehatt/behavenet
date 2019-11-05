# behavenet

NOTE: this repo is currently under construction! Check back in December 2019 for a more user-friendly codebase.

## Environment Set-Up
```
$: conda create --name=behavenet python=3.7.2
$: source activate behavenet
(behavenet) $: pip install -r requirements.txt
```
To be able to use this environment for jupyter notebooks:
```
(behavenet) $: python -m ipykernel install --user --name behavenet --display-name "behavenet"
``` 

### pytorch

Install pytorch 1.0.1 in accordance with platform/cuda requirements

### ssm

To install ssm, `cd` to any directory where you would like to keep the ssm code and run the following:

```
(behavenet) $: git clone git@github.com:slinderman/ssm.git
(behavenet) $: cd ssm
(behavenet) $: pip install cython
(behavenet) $: pip install -e .
```

### behavenet

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main behavenet directory:

```
(behavenet) $: pip install -e .
```

Next, set up your paths to data, results, and figures. Launch python from the behavenet environment, and type:

```python
from behavenet import setup
setup()
```

You will be asked to input a base data directory; all data should be stored as `base_data_dir/lab_name/expt_name/animal_id/session_id/data.hdf5`. More information on the dataset structure to come.

You will also be asked to input a base results directory, which will store all of the model fits. Finally, the base figure directory will be used to store figure and video outputs. 
