# behavenet

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
