# PINNs for Time-Domain Simulations: Accuracy, Computational Cost, and Flexibility

This repository contains the implementation used for [Physics-Informed Neural Networks for Time-Domain Simulations: Accuracy, Computational Cost, and Flexibility](https://www.sciencedirect.com/science/article/pii/S0378779623006375)
which has been published in the journal Electric Power Systems Research.

We showcase how PINNs relate to conventional numerical solvers for the solution of Differential-Algebraic Equations (DAEs) found in power systems. 

## Installation

To install and activate the environment using conda, run the following in the terminal:

```setup
conda env create -f environment.yml
conda activate PINNs_time_domain_simulations
```
If problems occur, you can also install the following packages and its dependencies manually: `pytorch`, `assimulo`, `pandas`, `wandb`, `matplotlib`. 

The present implementation relies on the platform [wandb](https://wandb.ai) for configuring, launching, and logging the different runs.
Wandb requires an account (free for light-weight logging, as it is the case here) and an API key for the login, see their webpage for further details.
`definitions.py` holds then the values for the entity and project to which the runs should be allocated. See [Getting started](#getting-started) for examples of using wandb.

## Getting started
The core functionality of the repository can be found in the folder `src`. The folder `utils` contains various helper files that were used for the specific cases in the paper. 
There are a few examples/scripts in the folder `utils/examples` that might be helpful for getting started with the code.

- `plot_power_system_trajectory.py` constructs a power system model, simulates a trajectory and plots the results.
- `create_datasets.py` simulates the datasets used in the paper. The resulting datasets can then be found in utils/datasets.
- `train_model.py` starts a training run using the pipeline in `training_workflow.py` 
- `run_sweep.py` illustrates how a larger number of models with different configurations can be trained. It utilises the sweep functionality from Wandb and is ideally run on a cluster as the execution of the entire script is costly.

## License
This project is made available under the MIT License.
## Reference this work

If you find this work helpful, please cite

```
@article{StiasnyPINNs2023,
title = {Physics-informed neural networks for time-domain simulations: Accuracy, computational cost, and flexibility},
journal = {Electric Power Systems Research},
volume = {224},
pages = {109748},
year = {2023},
doi = {https://doi.org/10.1016/j.epsr.2023.109748},
author = {Jochen Stiasny and Spyros Chatzivasileiadis},
}
```