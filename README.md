# BehaveNet 
[![Build Status](https://travis-ci.com/themattinthehatt/behavenet.svg?branch=master)](https://travis-ci.com/themattinthehatt/behavenet)
[![Documentation Status](https://readthedocs.org/projects/behavenet/badge/?version=latest)](https://behavenet.readthedocs.io/en/latest/?badge=latest)

BehaveNet is a probabilistic framework for the analysis of behavioral video and neural activity. 
This framework provides tools for compression, segmentation, generation, and decoding of behavioral 
videos. Please see the 
[paper](https://papers.nips.cc/paper/9701-behavenet-nonlinear-embedding-and-bayesian-neural-decoding-of-behavioral-videos) 
for additional details about the algorithms, and see the
[BehaveNet documentation](https://behavenet.readthedocs.io/en/latest/) 
for more information about how to install the software and begin fitting models to data. 

Additionally, we provide an example dataset and several jupyter notebooks that walk you through how 
to download the dataset, fit models, and analyze the results. The jupyter notebooks can be found 
[here](examples).

## Bibtex

If you use BehaveNet in your analysis of behavioral data, please cite us!

    @inproceedings{batty2019behavenet,
      title={BehaveNet: nonlinear embedding and Bayesian neural decoding of behavioral videos},
      author={Batty, Eleanor and Whiteway, Matthew and Saxena, Shreya and Biderman, Dan and Abe, Taiga and Musall, Simon and Gillis, Winthrop and Markowitz, Jeffrey and Churchland, Anne and Cunningham, John P and others},
      booktitle={Advances in Neural Information Processing Systems},
      pages={15680--15691},
      year={2019}
    }

Citation for the Partioned Subspace VAE (PS-VAE)

    @article{whiteway2021partitioning,
      title={Partitioning variability in animal behavioral videos using semi-supervised variational autoencoders},
      author={Whiteway, Matthew R and Biderman, Dan and Friedman, Yoni and Dipoppa, Mario and Buchanan, E Kelly and Wu, Anqi and Zhou, John and Bonacchi, Niccol{\`o} and Miska, Nathaniel J and Noel, Jean-Paul and others},
      journal={PLoS computational biology},
      volume={17},
      number={9},
      pages={e1009439},
      year={2021},
      publisher={Public Library of Science San Francisco, CA USA}
    }
