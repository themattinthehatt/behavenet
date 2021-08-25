## PS-VAE example notebooks

These notebooks guide users through the process of downloading example data, as well as already trained PS-VAE models, and performing some simple analyses with these models.

The first notebook prompts users to input paths for the storage of data, models, and figures. Then users can download each of four different datasets used in the PS-VAE paper found [here](https://www.biorxiv.org/content/10.1101/2021.02.22.432309v2).

The second notebook performs the following analyses (and saves the associated figures/videos):

* plot training curves for each component of the PS-VAE cost function (as well as the full cost function)
* make a video of original frames, PS-VAE frame reconstructions, and VAE frame reconstructions
* plot ground truth labels and their PS-VAE reconstructions
* plot latent traversals for a subset of latent dimensions
* make a video of latent traversals for each latent dimension
