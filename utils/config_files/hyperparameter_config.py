import numpy as np


def check_required_config_keys(sweep_config_):
    required_keys = [
        "hidden_layer_size",
        "n_hidden_layers",
        "pytorch_init_seed",
        "epochs",
        "learning_rate",
        "lr_decay",
        "tolerance_change",
        "tolerance_grad",
        "history_size",
        "line_search",
        "max_iterations",
        "dt_regulariser",
        "physics_regulariser_max",
        "physics_regulariser_epochs_to_tenfold",
        "dataset_name",
        "time_increment_training",
        "power_increment_training",
        "time_increment_collocation",
        "power_increment_collocation",
        "threads",
        "validate_with_testset",
    ]

    for required_key in required_keys:
        assert (
            required_key in sweep_config_["parameters"].keys()
        ), f"Please specify the parameter {required_key}"

    pass


def default_sweep_config(search_type, NN_type, scenario):
    if search_type == "grid":
        method = "grid"
        metric_name = "loss_validation_best"
    elif search_type == "random":
        method = "random"
        metric_name = "loss_validation_best"
    else:
        raise NotImplementedError

    if scenario != "E":
        if NN_type == "NN":
            learning_rate = 1.0
            dt_regulariser = 0.0
            physics_regulariser_max = 0.0
            history_size = 140
            max_iterations = 22
        elif NN_type == "dtNN":
            learning_rate = 0.5
            dt_regulariser = 0.3
            physics_regulariser_max = 0.0
            history_size = 120
            max_iterations = 23
        elif NN_type == "PINN":
            learning_rate = 1.2
            dt_regulariser = 0.01
            physics_regulariser_max = 0.5
            history_size = 120
            max_iterations = 20
        else:
            raise NotImplementedError
    else:
        if NN_type == "NN":
            learning_rate = 1.6
            dt_regulariser = 0.0
            physics_regulariser_max = 0.0
            history_size = 140
            max_iterations = 22
        elif NN_type == "dtNN":
            learning_rate = 2.0
            dt_regulariser = 1.0
            physics_regulariser_max = 0.0
            history_size = 120
            max_iterations = 20
        elif NN_type == "PINN":
            learning_rate = 1.0
            dt_regulariser = 0.01
            physics_regulariser_max = 0.01
            history_size = 120
            max_iterations = 19
        else:
            raise NotImplementedError

    parameters_dict = {
        "epochs": {"value": 5000},
        "hidden_layer_size": {"value": 32},
        "n_hidden_layers": {"value": 5},
        "pytorch_init_seed": {"value": None},
        "dataset_name": {"value": "kundur"},
        "learning_rate": {"value": learning_rate},
        "lr_decay": {"value": 1.0},
        "tolerance_change": {"value": 1e-10},
        "tolerance_grad": {"value": 1e-7},
        "history_size": {"value": history_size},
        "line_search": {"value": True},
        "max_iterations": {"value": max_iterations},
        "dt_regulariser": {"value": dt_regulariser},
        "physics_regulariser_max": {"value": physics_regulariser_max},
        "physics_regulariser_epochs_to_tenfold": {"value": 20.0},
        "time_increment_training": {"value": 0.2},
        "power_increment_training": {"value": 0.2},
        "time_increment_collocation": {"value": 0.2},
        "power_increment_collocation": {"value": 0.2},
        "threads": {"value": 4},
        "validate_with_testset": {"value": False},
    }

    sweep_config_ = {
        "program": "training_workflow.py",
        "method": method,
        "name": "default",
    }

    metric = {"name": metric_name, "goal": "minimize"}

    sweep_config_["parameters"] = parameters_dict
    sweep_config_["metric"] = metric

    return sweep_config_


def create_config_hyperparameter_accuracy(dataset_name_, seed=None):
    sweep_config_ = default_sweep_config(search_type="grid", NN_type="NN", scenario="E")

    if seed is None:
        sweep_config_["parameters"]["pytorch_init_seed"] = {
            "values": np.arange(20).tolist()
        }
        name = f"accuracy_size_{dataset_name_}"
    else:
        assert type(seed) == int
        sweep_config_["parameters"]["pytorch_init_seed"] = {"value": seed}
        name = f"accuracy_size_{dataset_name_}_single_seed"

    sweep_config_["name"] = name

    sweep_config_["parameters"]["dataset_name"] = {"value": dataset_name_}
    sweep_config_["parameters"]["hidden_layer_size"] = {"values": [16, 32, 64, 128]}
    sweep_config_["parameters"]["n_hidden_layers"] = {"values": [2, 3, 4, 5]}
    sweep_config_["parameters"]["validate_with_testset"] = {"value": True}

    check_required_config_keys(sweep_config_=sweep_config_)
    return sweep_config_


def create_config_hyperparameter_exploration_scenarios(NN_type_, scenario_name_="E"):
    sweep_config_ = default_sweep_config(
        search_type="random", NN_type=NN_type_, scenario=scenario_name_
    )

    sweep_config_["name"] = f"tuning_scenario_{scenario_name_}_{NN_type_}"

    if scenario_name_ != "E":
        sweep_config_["parameters"]["time_increment_training"] = {"value": 0.0}
        sweep_config_["parameters"]["power_increment_training"] = {"value": 0.0}

    sweep_config_["parameters"].update(
        learning_rate={"distribution": "q_uniform", "min": 0.1, "max": 2.0, "q": 0.1},
        history_size={"distribution": "q_uniform", "min": 100, "max": 150, "q": 10},
        max_iterations={"distribution": "q_uniform", "min": 18, "max": 28, "q": 1},
    ),
    if NN_type_ != "NN":
        sweep_config_["parameters"].update(
            dt_regulariser={
                "distribution": "q_log_uniform_values",
                "min": 0.01,
                "max": 2.0,
                "q": 0.01,
            }
        ),
    if NN_type_ == "PINN":
        sweep_config_["parameters"].update(
            physics_regulariser_max={
                "values": [10, 5.0, 1.0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3]
            },
            physics_regulariser_epochs_to_tenfold={"values": [10, 15, 20, 30, 40, 50]},
        ),

    check_required_config_keys(sweep_config_=sweep_config_)
    return sweep_config_


def create_lbfgs_hyperparameters():
    sweep_config_ = default_sweep_config(
        search_type="random", NN_type="PINN", scenario="E"
    )

    sweep_config_["name"] = f"lbfgs_exploration_PINN"
    sweep_config_["method"] = "random"

    sweep_config_["parameters"].update(
        learning_rate={"distribution": "q_uniform", "min": 0.1, "max": 2.0, "q": 0.1},
        history_size={"distribution": "q_uniform", "min": 50, "max": 200, "q": 10},
        max_iterations={"distribution": "q_uniform", "min": 15, "max": 35, "q": 1},
        dt_regulariser={
            "distribution": "q_log_uniform_values",
            "min": 0.01,
            "max": 2.0,
            "q": 0.01,
        },
        physics_regulariser_max={
            "values": [10, 5.0, 1.0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3]
        },
        physics_regulariser_epochs_to_tenfold={"values": [10, 15, 20, 30, 40, 50]},
        tolerance_change={"values": [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]},
        tolerance_grad={"values": [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]},
        line_search={"values": [True, False]},
    ),

    check_required_config_keys(sweep_config_=sweep_config_)
    return sweep_config_


def create_config_scenarios(NN_type_, scenario_name_="E", seed=None):
    sweep_config_ = default_sweep_config(
        search_type="grid", NN_type=NN_type_, scenario=scenario_name_
    )

    if seed is None:
        sweep_config_["parameters"]["pytorch_init_seed"] = {
            "values": np.arange(20).tolist()
        }
        name = f"scenario_{scenario_name_}_{NN_type_}"
    else:
        assert type(seed) == int
        sweep_config_["parameters"]["pytorch_init_seed"] = {"value": seed}
        name = f"scenario_{scenario_name_}_{NN_type_}_single_seed"

    sweep_config_["name"] = name

    if scenario_name_ != "E":
        sweep_config_["parameters"]["time_increment_training"] = {"values": [1.0, 2.0]}
        sweep_config_["parameters"]["power_increment_training"] = {"values": [1.0, 2.0]}

    check_required_config_keys(sweep_config_=sweep_config_)
    return sweep_config_
