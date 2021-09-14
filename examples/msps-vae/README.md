## MSPS-VAE data download

The jupyter notebook (`00_download_and_preprocess.ipynb`) helps users to download public IBL data - both raw behavioral videos and DLC traces - and preprocess this raw data into a single hdf5 file for each session. These hdf5s can then be used to fit PS-VAE or MSPS-VAE models.

Additional files in this directory:
* `ibl_ephys_params.json`: use this as the data json file when fitting the MSPS-VAE
* `4-session.csv`: set the `"sessions_csv"` parameter in the `ibl_ephys_params.json` file to the absolute path of this file to fit the MSPS-VAE on the four sessions used in the original paper


See the documentation [here](https://behavenet.readthedocs.io/en/latest/source/user_guide.msps_vae.html) for more information on how to fit the MSPS-VAE.
