
Slurm Job Submission
====================

Using Behavenet with Slurm is simple: given the slurm submission information in a premade .sh file, test-tube will automatically submit all of the jobs for a given grid search with those slurm settings (see Introduction/Grid searching with test-tube for more details).

Steps
------
1) Create an .sh file with the slurm job parameters that you wish to use for this group of jobs (i.e. all the models in the grid search). For example, your .sh file could be:


.. code-block:: console

    #!/bin/bash
    #SBATCH --job-name=ae_grid_search    # Job name
    #SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
    #SBATCH --mail-user=email@gmail.com   # Where to send mail	
    #SBATCH --time=00:05:00               # Time limit hrs:min:sec

2) Add slurm hyperparameters (as specified in Hyperparameter glossary) to your compute.json 

3) Run the python script as specified throughout these docs and BehaveNet/test-tube will take care of the rest!
