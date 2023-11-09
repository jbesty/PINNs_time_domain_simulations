from collections import ChainMap

import numpy as np
import torch
import torch.utils.data
from matplotlib import patches
from matplotlib import pyplot as plt


def linspace_by_increment(value_min, value_max, increment, decimals=5, offset=False):
    n_steps = int((value_max - value_min) / increment) + 1
    assert value_max == value_min + (n_steps - 1) * increment
    if offset:
        vector = torch.linspace(value_min, value_max, n_steps) + increment / 2
    else:
        vector = torch.linspace(value_min, value_max, n_steps)

    return torch.round(vector, decimals=decimals)


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Simple class that defines the dataset structure in the training process.
    Can then be used with DataLoader.
    """

    def __init__(self, dataset, PowerSystem):
        # config files
        self.dataset_config = dataset["dataset_config"]
        self.bus_disturbance = dataset["dataset_config"].bus_disturbance
        self.time_min = dataset["dataset_config"].time_min
        self.time_max = dataset["dataset_config"].time_max
        self.time_increment = dataset["dataset_config"].time_increment
        self.power_min = dataset["dataset_config"].power_min
        self.power_max = dataset["dataset_config"].power_max
        self.power_increments = dataset["dataset_config"].power_increment
        # TODO: set accuracy based on increments
        self.time_digits = 4
        self.power_digits = 4

        # data and place holders
        self.time = torch.round(
            torch.from_numpy(dataset["time"]).float(), decimals=self.time_digits
        )
        self.control_input = torch.round(
            torch.from_numpy(dataset["control_input"]).float(),
            decimals=self.power_digits,
        )
        self.states_initial = torch.from_numpy(dataset["states_initial"]).float()
        self.states_results = torch.from_numpy(dataset["states_results"]).float()
        self.states_dt_results = torch.from_numpy(dataset["states_dt_results"]).float()

        # number of data point in time and power
        self.n_time_steps = len(self.time.unique())
        self.n_power_steps = len(self.control_input[:, self.bus_disturbance].unique())

        # additional fields, to be filled later on
        self.states_prediction = torch.zeros(self.states_results.shape)
        self.states_dt_prediction = torch.zeros(self.states_results.shape)
        self.physics_prediction = torch.zeros(self.states_results.shape)
        self.error_states_absolute = torch.zeros(self.states_results.shape)
        self.error_states_dt_absolute = torch.zeros(self.states_results.shape)
        self.error_physics_absolute = torch.zeros(self.states_results.shape)
        self.error_states_full = torch.zeros(self.states_results.shape)
        self.error_states_full = torch.zeros(self.states_results.shape)
        self.error_states_full = torch.zeros(self.states_results.shape)
        self.error_states_point_wise = torch.zeros(self.states_results.shape)
        self.trajectory_error = torch.zeros((self.n_power_steps, 1))
        self.error_states_dt_full = torch.zeros(self.states_results.shape)
        self.error_states_dt_point_wise = torch.zeros(self.states_results.shape)
        self.trajectory_dt_error = torch.zeros((self.n_power_steps, 1))
        self.error_physics_full = torch.zeros(self.states_results.shape)
        self.error_physics_point_wise = torch.zeros(self.states_results.shape)
        self.trajectory_physics_error = torch.zeros((self.n_power_steps, 1))

        # dataset specific structure - return values for training (__getitem__)
        self.input_data = torch.concat(
            [
                self.time,
                self.control_input[:, self.bus_disturbance : self.bus_disturbance + 1],
            ],
            dim=1,
        )
        self.output_data = self.states_results
        self.output_data_dt = self.states_dt_results

        # data subset specifics files
        self.training_indices = list()
        self.validation_indices = list()
        self.testing_indices = list()
        self.collocation_indices = list()

        # dataset subset statistics
        self.mean_input_data_training = torch.zeros((1, self.input_data.shape[1]))
        self.std_input_data_training = torch.ones((1, self.input_data.shape[1]))

        self.mean_output_data_training = torch.zeros((1, self.output_data.shape[1]))
        self.std_output_data_training = torch.ones((1, self.output_data.shape[1]))

        self.mean_output_data_dt_training = torch.zeros(
            (1, self.output_data_dt.shape[1])
        )
        self.std_output_data_dt_training = torch.ones((1, self.output_data_dt.shape[1]))

        self.mean_output_data_testing = torch.zeros((1, self.output_data.shape[1]))
        self.std_output_data_testing = torch.ones((1, self.output_data.shape[1]))

        self.mean_output_data_dt_testing = torch.zeros(
            (1, self.output_data_dt.shape[1])
        )
        self.std_output_data_dt_testing = torch.ones((1, self.output_data_dt.shape[1]))

        self.std_output_data_training_state_type = torch.ones(
            (1, self.output_data_dt.shape[1])
        )
        self.std_output_data_testing_state_type = torch.ones(
            (1, self.output_data_dt.shape[1])
        )

        # Power system data
        self.A_T = torch.from_numpy(PowerSystem.A.T).float()
        self.B_T = torch.from_numpy(PowerSystem.B.T).float()
        self.F_T = torch.from_numpy(PowerSystem.F.T).float()
        self.M_T = torch.from_numpy(PowerSystem.M.T).float()
        self.U_T = torch.from_numpy(PowerSystem.U.T).float()
        self.V_magnitude_pu_T = torch.from_numpy(
            PowerSystem.V_magnitude_pu.reshape((1, -1))
        )
        self.P_set_point_generators_pu_T = torch.from_numpy(
            PowerSystem.P_set_point_generators_pu.reshape((1, -1))
        ).float()
        self.P_set_point_loads_pu_T = torch.from_numpy(
            PowerSystem.P_set_point_loads_pu.reshape((1, -1))
        ).float()
        self.Y_bus_pu = torch.complex(
            torch.from_numpy(np.real(PowerSystem.Y_bus_pu)),
            torch.from_numpy(np.imag(PowerSystem.Y_bus_pu)),
        )

        # plotting information (state space)
        self.n_states = PowerSystem.n_states
        self.n_buses = PowerSystem.n_buses
        self.state_delta_bus = PowerSystem.state_delta_bus
        self.state_delta_load = PowerSystem.state_delta_load
        self.state_delta_generator = PowerSystem.state_delta_generator
        self.state_omega_generator = PowerSystem.state_omega_generator

        self.differential_states = PowerSystem.differential_states
        self.state_names = PowerSystem.state_names
        self.states_bus_index = PowerSystem.states_bus_index
        self.bus_colour_map = PowerSystem.bus_colour_map

        self.delta_states = [
            not omega_state for omega_state in self.state_omega_generator
        ]

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return (
            self.input_data[idx, :],
            self.output_data[idx, :],
            self.output_data_dt[idx, :],
            self.time[idx, :],
            self.control_input[idx, :],
        )

    def filter_by_time(self, time_vector):
        filter_boolean = torch.isin(self.time[:, 0], time_vector)
        return filter_boolean

    def filter_by_control_input(self, control_input_vector):
        filter_boolean = torch.isin(
            self.control_input[:, self.bus_disturbance], control_input_vector
        )
        return filter_boolean

    def divide_dataset(self, config):
        self.dataset_indices = torch.arange(0, len(self))
        # nasty work-around to have the option for hyperparameter tuning for different dataset size
        # only critical if the increment is set to 0
        if config.time_increment_training == 0.0:
            time_increment_training = np.random.choice(np.array([1.0, 2.0]), size=1)[0]
        else:
            time_increment_training = config.time_increment_training

        if config.power_increment_training == 0.0:
            power_increment_training = np.random.choice(np.array([1.0, 2.0]), size=1)[0]
        else:
            power_increment_training = config.power_increment_training

        time_vector_training = linspace_by_increment(
            self.time_min, self.time_max, time_increment_training
        )
        power_vector_training = linspace_by_increment(
            self.power_min, self.power_max, power_increment_training
        )

        time_vector_validation = linspace_by_increment(
            self.time_min, self.time_max, time_increment_training, offset=True
        )
        power_vector_validation = linspace_by_increment(
            self.power_min, self.power_max, power_increment_training, offset=True
        )

        time_vector_collocation = linspace_by_increment(
            self.time_min, self.time_max, config.time_increment_collocation
        )
        power_vector_collocation = linspace_by_increment(
            self.power_min, self.power_max, config.power_increment_collocation
        )

        self.training_indices = self.dataset_indices[
            torch.logical_and(
                self.filter_by_time(time_vector_training),
                self.filter_by_control_input(power_vector_training),
            )
        ]
        self.validation_indices = self.dataset_indices[
            torch.logical_and(
                self.filter_by_time(time_vector_validation),
                self.filter_by_control_input(power_vector_validation),
            )
        ]

        self.collocation_indices = self.dataset_indices[
            torch.logical_and(
                self.filter_by_time(time_vector_collocation),
                self.filter_by_control_input(power_vector_collocation),
            )
        ]
        self.testing_indices = self.dataset_indices

        assert len(self.training_indices) == len(time_vector_training) * len(
            power_vector_training
        )
        assert len(self.collocation_indices) == len(time_vector_collocation) * len(
            power_vector_collocation
        )
        assert len(self.validation_indices) == (len(time_vector_training) - 1) * (
            len(power_vector_training) - 1
        )

        self.update_dataset_statistics()
        pass

    def get_training_data(self):
        return (
            self.input_data[self.training_indices, :],
            self.output_data[self.training_indices, :],
            self.output_data_dt[self.training_indices, :],
        )

    def get_validation_data(self):
        return (
            self.input_data[self.validation_indices, :],
            self.output_data[self.validation_indices, :],
        )

    def get_testing_data(self):
        return (
            self.input_data[self.testing_indices, :],
            self.output_data[self.testing_indices, :],
        )

    def get_collocation_data(self):
        return (
            self.input_data[self.collocation_indices, :],
            self.control_input[self.collocation_indices, :],
        )

    def update_dataset_statistics(self):
        self.mean_input_data_training = torch.mean(
            self.input_data[self.training_indices, :], dim=0, keepdim=True
        )
        self.std_input_data_training = torch.std(
            self.input_data[self.training_indices, :], dim=0, keepdim=True
        )

        self.mean_output_data_training = torch.mean(
            self.output_data[self.training_indices, :], dim=0, keepdim=True
        )
        self.std_output_data_training = torch.std(
            self.output_data[self.training_indices, :], dim=0, keepdim=True
        )

        self.mean_output_data_dt_training = torch.mean(
            self.output_data_dt[self.training_indices, :], dim=0, keepdim=True
        )
        self.std_output_data_dt_training = torch.std(
            self.output_data_dt[self.training_indices, :], dim=0, keepdim=True
        )

        self.mean_output_data_testing = torch.mean(
            self.output_data[self.testing_indices, :], dim=0, keepdim=True
        )
        self.std_output_data_testing = torch.std(
            self.output_data[self.testing_indices, :], dim=0, keepdim=True
        )

        self.mean_output_data_dt_testing = torch.mean(
            self.output_data_dt[self.testing_indices, :], dim=0, keepdim=True
        )
        self.std_output_data_dt_testing = torch.std(
            self.output_data_dt[self.testing_indices, :], dim=0, keepdim=True
        )

        delta_states = torch.reshape(torch.tensor(self.delta_states), (1, -1))
        omega_states = torch.reshape(torch.tensor(self.state_omega_generator), (1, -1))
        self.std_output_data_training_state_type = (
            torch.mean(self.std_output_data_training[:, self.delta_states])
            * delta_states
            + torch.mean(self.std_output_data_training[:, self.state_omega_generator])
            * omega_states
        )
        self.std_output_data_testing_state_type = (
            torch.mean(self.std_output_data_testing[:, self.delta_states])
            * delta_states
            + torch.mean(self.std_output_data_testing[:, self.state_omega_generator])
            * omega_states
        )
        pass

    def calculate_state_update(self, states, states_dt, control_input):
        angle_complex = torch.complex(
            real=(states @ self.U_T) * 0.0, imag=states @ self.U_T
        )
        V_complex = self.V_magnitude_pu_T * torch.exp(angle_complex)
        FCX_first = torch.matmul(torch.diag_embed(V_complex), torch.conj(self.Y_bus_pu))
        FCX_second = torch.matmul(FCX_first, torch.conj(V_complex[:, :, None]))
        FCX = torch.real(FCX_second[:, :, 0]).float()
        state_update_left_hand_side = states_dt @ self.M_T
        state_update_right_hand_side = (
            states @ self.A_T
            + FCX @ self.F_T
            + (
                self.P_set_point_generators_pu_T
                + self.P_set_point_loads_pu_T
                + control_input
            )
            @ self.B_T
        )

        return state_update_left_hand_side, state_update_right_hand_side

    def plot_trajectory(self, trajectory_indices, states_true=True, prediction=False):
        n_subplots = 2

        fig = plt.figure(figsize=(8, n_subplots * 4))
        axes = fig.subplots(nrows=n_subplots, ncols=1, sharex="col")

        axes[0].set_ylabel("delta_i [deg]")
        axes[1].set_ylabel("Delta omega_i1 [rad/s]")
        axes[1].set_xlabel("Time [s]")

        handles = [
            patches.Patch(
                color=self.bus_colour_map.colors[bus_index],
                label=f"Bus {bus_index + 1}",
            )
            for bus_index in range(self.n_buses)
        ]

        fig.legend(handles=handles, loc="upper center", ncol=6)

        time = self.time[trajectory_indices, :].numpy()

        if states_true:
            linestyle_plot = "dashed" if prediction else "solid"
            states_plot = self.states_results[trajectory_indices, :].numpy()
            for state_index, state_plot in enumerate(states_plot.T):
                bus_index = self.states_bus_index[state_index]
                if self.delta_states[state_index]:
                    axes[0].plot(
                        time,
                        np.rad2deg(state_plot),
                        color=self.bus_colour_map.colors[bus_index],
                        linestyle=linestyle_plot,
                    )
                elif self.state_omega_generator[state_index]:
                    axes[1].plot(
                        time,
                        state_plot,
                        color=self.bus_colour_map.colors[bus_index],
                        linestyle=linestyle_plot,
                    )
                else:
                    pass

        if prediction:
            linestyle_plot = "solid"
            states_plot = self.states_prediction[trajectory_indices, :].numpy()
            for state_index, state_plot in enumerate(states_plot.T):
                bus_index = self.states_bus_index[state_index]
                if self.delta_states[state_index]:
                    axes[0].plot(
                        time,
                        np.rad2deg(state_plot),
                        color=self.bus_colour_map.colors[bus_index],
                        linestyle=linestyle_plot,
                    )
                elif self.state_omega_generator[state_index]:
                    axes[1].plot(
                        time,
                        state_plot,
                        color=self.bus_colour_map.colors[bus_index],
                        linestyle=linestyle_plot,
                    )
                else:
                    pass

        return fig
