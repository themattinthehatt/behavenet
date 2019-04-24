# behavenet

## Environment Set-Up
```
conda create --name=behavenet python=3.7.2
source activate behavenet
pip install -r requirements.txt
```
To be able to use this environment for jupyter notebooks:
```
python -m ipykernel install --user --name behavenet --display-name "behavenet"
``` 
Install pytorch 1.0.1 in accordance with platform/cuda requirements

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main behavenet directory:

```
pip install -e .
```
