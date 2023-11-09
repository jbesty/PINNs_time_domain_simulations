from types import SimpleNamespace

from definitions import wandb_entity, wandb_project


def create_training_config():
    """
    A basic configuration file to train a single NN with the specified parameters.
    :return: config dictionary
    """

    parameters_dict = {
        "project": wandb_project,
        "entity": wandb_entity,
        "hidden_layer_size": 32,
        "n_hidden_layers": 5,
        "pytorch_init_seed": None,
        "epochs": 50,
        "learning_rate": 1.6,
        "lr_decay": 1.0,
        "tolerance_change": 1e-10,
        "tolerance_grad": 1e-7,
        "history_size": 120,
        "line_search": True,
        "max_iterations": 22,
        "dt_regulariser": 0.0,
        "physics_regulariser_max": 0.0,
        "physics_regulariser_epochs_to_tenfold": 20,
        "dataset_name": "kundur",
        "time_increment_training": 2.0,
        "power_increment_training": 2.0,
        "time_increment_collocation": 0.2,
        "power_increment_collocation": 0.2,
        "threads": 16,
        "validate_with_testset": False,
    }
    config = SimpleNamespace(**parameters_dict)

    return config
