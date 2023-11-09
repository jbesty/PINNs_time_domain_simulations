#!/usr/bin/env python3
import pathlib
import shutil
import time

import numpy as np
import torch
import wandb

import definitions
from utils.datasets.load_dataset import load_dataset_by_name

from .loss_functions import loss_function
from .neural_network import NeuralNetwork
from .power_system import PowerSystem
from .train_epoch import train_epoch_lbfgs
from .trajectory_dataset import TrajectoryDataset
from .validate_epoch import validate_epoch


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, project=definitions.wandb_project) as run:
        config = wandb.config

        torch.set_num_threads(config.threads)

        # Setup of the logging
        model_artifact = wandb.Artifact(f"model_{run.id}", type="model")
        model_save_path = f"{run.dir}\\model.pth"
        loss_validation_best = torch.tensor(10000.0)
        best_epoch = 0

        # %% Load and prepare dataset ---------------------------------------------
        # If desired add a function that creates a dataset
        dataset_raw = load_dataset_by_name(dataset_name=config.dataset_name)

        power_system = PowerSystem(power_system_config=dataset_raw["system_parameters"])

        dataset = TrajectoryDataset(dataset=dataset_raw, PowerSystem=power_system)

        dataset.divide_dataset(config=config)

        config.n_input_neurons = dataset.input_data.shape[1]
        config.n_output_neurons = dataset.output_data.shape[1]

        # optional for best generalisation
        if config.validate_with_testset:
            dataset.validation_indices = dataset.testing_indices

        # %% Model setup ---------------------------------------------
        # Basic setup of the model, optimizer and scheduler
        # (includes the standardisation  of the dataset, which could/should be moved to the pre-processing stage)

        neural_network_model = NeuralNetwork(
            hidden_layer_size=config.hidden_layer_size,
            n_hidden_layers=config.n_hidden_layers,
            n_input_neurons=config.n_input_neurons,
            n_output_neurons=config.n_output_neurons,
            pytorch_init_seed=config.pytorch_init_seed,
        )

        neural_network_model.standardise_input(
            input_mean=dataset.mean_input_data_training[0, :],
            input_standard_deviation=dataset.std_input_data_training[0, :],
        )
        neural_network_model.scale_output(
            output_mean=dataset.mean_output_data_training[0, :],
            output_standard_deviation=dataset.std_output_data_training_state_type[0, :],
        )

        optimiser = torch.optim.LBFGS(
            neural_network_model.parameters(),
            lr=config.learning_rate,
            tolerance_change=config.tolerance_change,
            tolerance_grad=config.tolerance_grad,
            line_search_fn="strong_wolfe" if config.line_search else None,
            history_size=config.history_size,
            max_iter=config.max_iterations,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimiser, gamma=config.lr_decay
        )

        training_time = 0.0
        loss_previous = torch.tensor(100000.0)
        loss_patience_counter = 0

        # possibly include a variable adjustment of loss function weights
        neural_network_model.dt_regulariser = torch.tensor(config.dt_regulariser)

        physics_regulariser_start = 1e-7

        if config.physics_regulariser_max > 0.0:
            epoch_physics_regulariser_max = np.ceil(
                np.log10(config.physics_regulariser_max / physics_regulariser_start)
                * config.physics_regulariser_epochs_to_tenfold
            )

        # Training loop
        # executing a full epoch including a train, validation, logging
        while (
            neural_network_model.epochs_total < config.epochs
            and loss_patience_counter < 5
        ):
            if config.physics_regulariser_max > 0.0:
                exponential_factor = np.min(
                    [
                        neural_network_model.epochs_total
                        / config.physics_regulariser_epochs_to_tenfold,
                        epoch_physics_regulariser_max,
                    ]
                )
                physics_regulariser = np.min(
                    [
                        config.physics_regulariser_max,
                        physics_regulariser_start * (10**exponential_factor),
                    ]
                )
            else:
                physics_regulariser = 0.0
            neural_network_model.physics_regulariser = torch.tensor(physics_regulariser)

            time_epoch_train_start = time.time()
            loss = train_epoch_lbfgs(neural_network_model, dataset, optimiser)
            time_epoch_train_end = time.time()
            training_time += time_epoch_train_end - time_epoch_train_start

            loss_validation = validate_epoch(neural_network_model, dataset)

            if loss_validation < loss_validation_best:
                best_epoch = neural_network_model.epochs_total
                loss_validation_best = loss_validation.detach()
                torch.save(neural_network_model.state_dict(), model_save_path)

            scheduler.step()

            wandb.log(
                {
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": neural_network_model.epochs_total,
                    "physics_regulariser_epoch": neural_network_model.physics_regulariser,
                }
            )

            loss_difference = torch.subtract(loss_previous, loss.detach())
            loss_previous = loss.detach()
            if torch.absolute(loss_difference) <= config.tolerance_change:
                loss_patience_counter += 1
            else:
                loss_patience_counter = 0
            neural_network_model.epochs_total += 1

        # %% Model assessment ---------------------------------------------
        neural_network_model.load_state_dict(torch.load(model_save_path))
        neural_network_model.eval()

        data, target = dataset.get_testing_data()
        with torch.no_grad():
            prediction, _ = neural_network_model.predict_dt(data)

        loss_testing = loss_function(
            prediction=prediction,
            target=target,
            loss_weights=1 / dataset.std_output_data_training_state_type,
        )

        wandb.log(
            {
                "best_epoch": best_epoch,
                "loss_validation_best": loss_validation_best,
                "training_time": training_time,
                "time_per_epoch": training_time
                / max((neural_network_model.epochs_total - 5), 1),
                "loss_testing": loss_testing,
            }
        )

        # log the model - it can be downloaded from wandb
        model_artifact.add_file(model_save_path)
        run.log_artifact(model_artifact)
        run.finish()

    # clean the directory
    logs_directory = pathlib.Path(run.dir).parent
    shutil.rmtree(logs_directory)

    return run
