Autoencoders
============

What are autoencoders?

How to fit a single example autoencoder?

copy jsons over to home directory

How to perform an architecture search

* 'initial': randomly sample and partially train multiple convolutional architectures
* 'top_n': fully fit the best performing models from the 'initial' architecture search
* 'latent_search': sweep over multiple values of latents for a single architecture

How to define a custom autoencoder architecture?

python behavenet/fitting/ae_grid_search.py --data_config ~/.behavenet/musall_vistrained.json --model_config ~/.behavenet/ae_model.json --training_config ~/.behavenet/ae_training.json --compute_config ~/.behavenet/ae_compute.json
