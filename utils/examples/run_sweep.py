import itertools

import wandb

import definitions
import src
from utils.config_files.hyperparameter_config import (
    create_config_hyperparameter_accuracy,
    create_config_hyperparameter_exploration_scenarios,
    create_config_scenarios,
    create_lbfgs_hyperparameters,
)


def run_agent_by_sweep_id(sweep_id):
    wandb.login()
    wandb.agent(
        sweep_id=sweep_id,
        function=src.train,
        project=definitions.wandb_project,
    )


if __name__ == "__main__":
    # !!!
    # Run this file on HPC, ideally with multiple agents, each executing a specific run from the created sweep.
    # Wandb assigns the run config automatically
    # !!!

    # Uncomment the following parts to obtain setups used in the paper
    # ---------------------------------
    # Scenario hyperparameter tuning
    # for scenario_name, NN_type in itertools.product(
    #     *(["A-D", "E"], ["NN", "dtNN", "PINN"])
    # ):
    #     sweep_config = create_config_hyperparameter_exploration_scenarios(
    #         NN_type_=NN_type, scenario_name_=scenario_name
    #     )
    #     wandb.sweep(
    #         sweep_config,
    #         entity=definitions.wandb_entity,
    #         project=definitions.wandb_project,
    #     )
    #
    # # ---------------------------------
    # # Accuracy NN size - Fig. 3
    # for dataset_name in ["kundur", "ieee39"]:
    #     sweep_config = create_config_hyperparameter_accuracy(dataset_name)
    #     wandb.sweep(
    #         sweep_config,
    #         entity=definitions.wandb_entity,
    #         project=definitions.wandb_project,
    #     )
    #
    # # ---------------------------------
    # # Scenarios A-E - Fig. 4
    # for scenario_name, NN_type in itertools.product(
    #     *(["A-D", "E"], ["NN", "dtNN", "PINN"])
    # ):
    #     sweep_config = create_config_scenarios(
    #         NN_type_=NN_type, scenario_name_=scenario_name
    #     )
    #     wandb.sweep(
    #         sweep_config,
    #         entity=definitions.wandb_entity,
    #         project=definitions.wandb_project,
    #     )

    # --------------------------------
    # L-BFGS sweep - Fig. 5
    sweep_config = create_lbfgs_hyperparameters()
    sweep_id = wandb.sweep(
        sweep_config, entity=definitions.wandb_entity, project=definitions.wandb_project
    )

    run_agent_by_sweep_id(sweep_id)
