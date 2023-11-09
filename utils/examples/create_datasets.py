import pickle

import numpy as np

from definitions import PROJECT_PATH
from src import PowerSystem
from utils.config_files.dataset_config import create_dataset_config
from utils.config_files.power_system_config import create_power_system_config


def single_trajectory_simulation(power_system, y0, yd0, dataset_config, P_disturbance):
    power_system.update_disturbance(P_disturbance_pu=P_disturbance)

    t, y, yd = power_system.simulate_trajectory(
        y0,
        yd0,
        t0=dataset_config.time_min,
        tfinal=dataset_config.time_max,
        dt=dataset_config.time_increment,
        atol=dataset_config.absolute_tolerance,
        rtol=dataset_config.relative_tolerance,
    )

    y0_extended = y0.reshape((1, -1)).repeat(repeats=len(t), axis=0)
    control_input_extended = P_disturbance.reshape((1, -1)).repeat(
        repeats=len(t), axis=0
    )
    return t, y, yd, y0_extended, control_input_extended


if __name__ == "__main__":
    dataset_names = ["kundur", "ieee39"]
    for dataset_name in dataset_names:
        dataset_config = create_dataset_config(dataset_name=dataset_name)

        test_system = PowerSystem(
            create_power_system_config(dataset_config.power_system_name)
        )

        y0, yd0 = test_system.calculate_initial_condition()

        n_power_steps = (
            int(
                (dataset_config.power_max - dataset_config.power_min)
                / dataset_config.power_increment
            )
            + 1
        )
        P_disturbance_bus = np.linspace(
            start=dataset_config.power_min,
            stop=dataset_config.power_max,
            num=n_power_steps,
            endpoint=True,
        )
        P_disturbances = np.zeros((len(P_disturbance_bus), test_system.n_buses))
        P_disturbances[:, dataset_config.bus_disturbance] = P_disturbance_bus

        dataset_results = [
            single_trajectory_simulation(
                test_system, y0, yd0, dataset_config, P_disturbance
            )
            for P_disturbance in P_disturbances
        ]

        dataset = dict(
            {
                "time": np.concatenate(
                    [dataset_result[0] for dataset_result in dataset_results],
                    axis=0,
                ),
                "states_results": np.concatenate(
                    [dataset_result[1] for dataset_result in dataset_results],
                    axis=0,
                ),
                "states_dt_results": np.concatenate(
                    [dataset_result[2] for dataset_result in dataset_results],
                    axis=0,
                ),
                "states_initial": np.concatenate(
                    [dataset_result[3] for dataset_result in dataset_results],
                    axis=0,
                ),
                "control_input": np.concatenate(
                    [dataset_result[4] for dataset_result in dataset_results],
                    axis=0,
                ),
                "system_parameters": test_system.parameter_file,
                "dataset_config": dataset_config,
            }
        )

        with open(
            PROJECT_PATH
            / "utils"
            / "datasets"
            / f"{dataset_config.dataset_name}.pickle",
            "wb",
        ) as file_path:
            pickle.dump(dataset, file_path)

        print(f"Simulated dataset {dataset_name}")

    # check that the dataset size is correct
    from utils.datasets.load_dataset import load_dataset_by_name

    print()
    print("Dataset sizes")
    for dataset_name in dataset_names:
        test = load_dataset_by_name(dataset_name=dataset_name)
        print(test["time"].shape[0])
