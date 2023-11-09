from types import SimpleNamespace


def create_dataset_config(dataset_name):
    power_min = 0.0
    time_min = 0.0
    absolute_tolerance = 1.0e-10
    relative_tolerance = 1.0e-10

    if dataset_name == "kundur":
        power_system_name = "kundur"
        power_max = 10.0
        time_max = 20.0
        bus_disturbance = 6

        time_increment = 0.05
        power_increment = 0.05

    elif dataset_name == "kundur_short":
        power_system_name = "kundur"
        power_max = 10.0
        time_max = 5.0
        bus_disturbance = 6

        time_increment = 0.05
        power_increment = 0.05

    elif dataset_name == "ieee39":
        power_system_name = "ieee39"
        power_max = 10.0
        time_max = 20.0
        bus_disturbance = 19

        time_increment = 0.05
        power_increment = 0.05

    elif dataset_name == "ieee39_short":
        power_system_name = "ieee39"
        power_max = 10.0
        time_max = 5.0
        bus_disturbance = 19

        time_increment = 0.05
        power_increment = 0.05

    else:
        raise Exception(f"Dataset {dataset_name} is not a valid dataset name!")

    dataset_config_dict = dict(
        {
            "dataset_name": dataset_name,
            "power_system_name": power_system_name,
            "time_min": time_min,
            "time_max": time_max,
            "power_min": power_min,
            "power_max": power_max,
            "bus_disturbance": bus_disturbance,
            "time_increment": time_increment,
            "power_increment": power_increment,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        }
    )

    dataset_config = SimpleNamespace(**dataset_config_dict)

    return dataset_config
