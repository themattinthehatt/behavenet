# behavenet

NOTE: this repo is currently under construction! Check back in December 2019 for a more 
user-friendly codebase.

## Environment Setup
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

To install ssm, `cd` to any directory where you would like to keep the ssm code and run the 
following:

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

## Additional Setup

### User paths

Next, set up your paths to data, results, and figures. Launch python from the behavenet 
environment, and type:

```python
from behavenet import setup
setup()
```

You will be asked to input a base data directory; all data should be stored as 
`base_data_dir/lab_name/expt_name/animal_id/session_id/data.hdf5`. More information on the dataset 
structure can be found below.

You will also be asked to input a base results directory, which will store all of the model fits. 
Finally, the base figure directory will be used to store figure and video outputs. 

### Add a new dataset

Next you will be asked to input some prior information about the dataset to avoid supplying this 
information at all intermediate steps (examples shown for Musall dataset):

* lab or experimenter name (musall)
* experiment name (vistrained)
* example animal name (mSM36)
* example session name (05-Dec-2017)
* trial splits (8;1;1;0) - this is how trials will be split among training, validation, testing, 
and gap trials, respectively. Typically we use training data to train the models; validation data
to choose the best model from a collection of models using different hyperparameters; test data to
produce plot and videos; and gap trials can optionally be inserted between training, validation, 
and test trials if desired.
* x pixels (128)
* y pixels (128)
* input channels (2) - this can refer to color channels (for RGB data) and/or multiple camera 
views, which should be concatenated along the color channel dimension. In the Musall dataset we use
grayscale images from two camera views, so a trial with 189 frames will have a block of video data
of shape (189, 2, 128, 128)
* use output mask (False) - an optional output mask can be applied to each video frame if desired;
these output masks must also be stored in the `data.hdf5` files as `masks`.
* frame rate (30) - in Hz; behavenet assumes that the video data and neural data are binned at the 
same temporal resolution
* neural data type (ca) - either 'ca' for 2-photon/widefield data, or 'spikes' for ephys data. This 
parameter controls the noise distribution for encoding models, as well as several other model 
hyperparameters

To input this information, launch python from the behavenet environment and type:

```python
from behavenet import add_dataset
add_dataset()
```
