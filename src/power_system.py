import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import linalg

from .component_matrices import compute_matrix_for_bus_type


class PowerSystem:
    """
    The update function is constructed as

    M \frac{dx}{dt} = A x + F FCX + B u

    where FCX computes the non-linear parts of the update equation, namely the power flows between the buses.
    """

    def __init__(self, power_system_config):
        self.n_generators = power_system_config.n_generators
        self.n_buses = power_system_config.n_buses
        self.H_generators_pu = power_system_config.H_generators_pu
        self.H_total_pu = sum(self.H_generators_pu)
        self.D_generators_pu = power_system_config.D_generators_pu
        self.D_loads_pu = power_system_config.D_loads_pu
        self.f_0_Hz = power_system_config.f_0_Hz
        self.omega_0_rad_s = power_system_config.omega_0_rad_s
        self.V_magnitude_pu = power_system_config.V_magnitude_pu
        self.V_magnitude_dc_pu = np.ones(self.V_magnitude_pu.shape)
        self.Y_bus_pu = power_system_config.Y_bus_pu
        self.Y_bus_pu_original = power_system_config.Y_bus_pu
        self.P_set_point_generators_pu = power_system_config.P_set_point_generators_pu
        self.P_set_point_loads_pu = power_system_config.P_set_point_loads_pu
        self.delta_initial = power_system_config.delta_initial.reshape((-1, 1))
        self.generator_indices = power_system_config.generator_indices
        self.load_indices = power_system_config.load_indices
        self.P_disturbance_pu = np.zeros((self.n_buses, 1))
        self.slack_bus = power_system_config.slack_bus
        self.bus_colour_map = power_system_config.cmap_buses
        self.parameter_file = power_system_config

        self.bus_type_list = self.prepare_bus_types()

        (
            self.A,
            self.B,
            self.F,
            self.M,
            self.U,
            self.differential_states,
            self.state_names,
            self.states_bus_index,
            self.delta_initial,
        ) = self.create_system_matrices()

        self.n_states = self.A.shape[0]
        self.state_delta_bus = [
            state == "delta_bus" for state in self.state_names.tolist()
        ]
        self.state_delta_load = [
            state == "delta_load" for state in self.state_names.tolist()
        ]
        self.state_delta_generator = [
            state == "delta_generator" for state in self.state_names.tolist()
        ]
        self.state_omega_generator = [
            state == "omega_generator" for state in self.state_names.tolist()
        ]

        self.delta_states = [
            not omega_state for omega_state in self.state_omega_generator
        ]

    def prepare_bus_types(self):
        bus_type_list = list()
        for bus in range(self.n_buses):
            if bus in self.generator_indices:
                n_th_generator = np.where(bus == self.generator_indices)[0][0]
                bus_dict = {
                    "type": 0,
                    "H_pu": self.H_generators_pu[n_th_generator],
                    "D_pu": self.D_generators_pu[n_th_generator],
                }
            elif bus in self.load_indices:
                n_th_load = np.where(bus == self.load_indices)[0][0]
                bus_dict = {"type": 1, "D_pu": self.D_loads_pu[n_th_load]}
            else:
                bus_dict = {"type": 2}

            bus_type_list.append(bus_dict)

        return bus_type_list

    def create_system_matrices(self):
        matrices = [
            compute_matrix_for_bus_type(bus_type, self.omega_0_rad_s)
            for bus_type in self.bus_type_list
        ]

        A = linalg.block_diag(*([matrix_element[0] for matrix_element in matrices]))
        B = linalg.block_diag(*([matrix_element[1] for matrix_element in matrices]))
        F = linalg.block_diag(*([matrix_element[2] for matrix_element in matrices]))
        M = linalg.block_diag(*([matrix_element[3] for matrix_element in matrices]))
        U = linalg.block_diag(*([matrix_element[4] for matrix_element in matrices]))
        differential_states = np.hstack(
            [matrix_element[5] for matrix_element in matrices]
        )
        state_names = np.hstack([matrix_element[6] for matrix_element in matrices])
        states_per_bus = np.hstack(
            [len(matrix_element[6]) for matrix_element in matrices]
        )
        states_bus_index = np.repeat(np.arange(len(states_per_bus)), states_per_bus)

        n_states = A.shape[0]
        bus_index_to_remove = self.slack_bus
        bus_indices_to_keep = np.delete(
            np.arange(self.n_buses), bus_index_to_remove
        ).tolist()

        index_to_remove = np.where(states_bus_index == self.slack_bus)[0][0]
        indices_to_keep = np.delete(np.arange(n_states), index_to_remove).tolist()
        A_shifted = A.copy()
        A_shifted[:, index_to_remove + 1 : index_to_remove + 2] -= U.T @ (
            np.ones((self.n_buses, 1)) * self.omega_0_rad_s
        )
        A_reduced = A_shifted[:, indices_to_keep][indices_to_keep, :]
        B_reduced = B[indices_to_keep, :]
        F_reduced = F[indices_to_keep, :]
        M_reduced = M[:, indices_to_keep][indices_to_keep, :]
        U_reduced = U[:, indices_to_keep]
        differential_states_reduced = differential_states[indices_to_keep]
        state_names_reduced = state_names[indices_to_keep]
        states_bus_index_reduced = states_bus_index[indices_to_keep]
        delta_initial_reduced = self.delta_initial[bus_indices_to_keep, :]
        return (
            A_reduced,
            B_reduced,
            F_reduced,
            M_reduced,
            U_reduced,
            differential_states_reduced,
            state_names_reduced,
            states_bus_index_reduced,
            delta_initial_reduced,
        )

    def calculate_initial_condition(self):
        bus_index_to_remove = self.slack_bus
        bus_indices_to_keep = np.delete(
            np.arange(self.n_buses), bus_index_to_remove
        ).tolist()
        y0 = (
            np.zeros((self.n_states, 1))
            + self.U[bus_indices_to_keep, :].T @ self.delta_initial
        )
        yd = np.zeros((self.n_states, 1))

        return y0, yd

    def residual(self, t, y, yd):
        y_vector = y.reshape((-1, 1))
        yd_vector = yd.reshape((-1, 1))

        V_complex = self.V_magnitude_pu * np.exp(1j * self.U @ y_vector)
        FCX = np.real(
            np.diag(V_complex[:, 0]) @ np.conj(self.Y_bus_pu) @ np.conj(V_complex)
        )
        residual_vector = self.M @ yd_vector - (
            self.A @ y_vector
            + self.F @ FCX
            + self.B
            @ (
                self.P_set_point_generators_pu
                + self.P_set_point_loads_pu
                + self.P_disturbance_pu
            )
        )

        return residual_vector.flatten()

    def right_hand_side(self, t, y):
        y_vector = y.reshape((-1, 1))

        V_complex = self.V_magnitude_pu * np.exp(1j * self.U @ y_vector)
        FCX = np.real(
            np.diag(V_complex[:, 0]) @ np.conj(self.Y_bus_pu) @ np.conj(V_complex)
        )
        rhs = (
            self.A @ y_vector
            + self.F @ FCX
            + self.B
            @ (
                self.P_set_point_generators_pu
                + self.P_set_point_loads_pu
                + self.P_disturbance_pu
            )
        )

        return rhs.flatten()

    def power_flows(self, y):
        y_vector = y.reshape((-1, 1))

        V_complex = self.V_magnitude_pu * np.exp(1j * self.U @ y_vector)
        FCX = np.real(
            np.diag(V_complex[:, 0]) @ np.conj(self.Y_bus_pu) @ np.conj(V_complex)
        )
        return FCX

    def update_disturbance(self, P_disturbance_pu):
        self.P_disturbance_pu = P_disturbance_pu.reshape((-1, 1))
        pass

    def adjust_line_admittance(self, bus1, bus2, factor):
        self.Y_bus_pu[bus1, bus2] = self.Y_bus_pu[bus1, bus2] * factor
        self.Y_bus_pu[bus2, bus1] = self.Y_bus_pu[bus2, bus1] * factor
        pass

    def restore_original_line_admittance(self, bus1, bus2):
        self.Y_bus_pu[bus1, bus2] = self.Y_bus_pu_original[bus1, bus2]
        self.Y_bus_pu[bus2, bus1] = self.Y_bus_pu_original[bus2, bus1]
        pass

    def simulate_trajectory(
        self, y0, yd0, t0, tfinal, dt=0.01, atol=1.0e-6, rtol=1.0e-6, ncp=None
    ):
        model = Implicit_Problem(
            self.residual, y0, yd0, t0
        )  # Create an Assimulo problem
        sim = IDA(model)
        sim.algvar = np.diag(self.M) > 0
        # print(sim.algvar)
        sim.atol = atol
        sim.rtol = rtol
        n_time_steps = int(np.round((tfinal - t0) / dt)) + 1
        if ncp is None:
            ncp_list = np.linspace(
                start=t0, stop=tfinal, num=n_time_steps, endpoint=True
            )

            ncp = 0
        else:
            ncp_list = None

        # This function is affected by the tolerance setting -> with low tolerance setting the
        # initial condition might not yield a residual of 0.
        sim.make_consistent("IDA_YA_YDP_INIT")

        sim.verbosity = 40
        sim.suppress_alg = True
        # t, y, yd = sim.simulate(tfinal, ncp=ncp, ncp_list=ncp_list)
        t, y, yd = sim.simulate(tfinal, ncp=n_time_steps - 1)

        if np.isclose(t[-2] + dt, t[-1]):
            return np.vstack(t), y, yd
        else:
            state_difference = y[-2, :] - y[-1, :]
            # if the solution reached a constant value, the solver skips over time steps
            # assert np.allclose(y[-3, :], y[-2, :], atol=2.0e-5)
            # assert np.allclose(yd[-3, :], yd[-2, :], atol=2.0e-5)
            print(f"Max state difference: {np.max(np.abs(state_difference)):.3e}")

            missing_time_steps = [time_value > t[-2] for time_value in ncp_list]
            t_missing = ncp_list[np.stack(missing_time_steps)]
            y_missing = y[-2:-1, :].repeat(t_missing.shape[0], axis=0)
            yd_missing = yd[-2:-1, :].repeat(t_missing.shape[0], axis=0)
            t_completed = np.vstack([np.vstack(t[:-1]), t_missing.reshape((-1, 1))])
            y_completed = np.vstack([y[:-1,], y_missing])
            yd_completed = np.vstack([yd[:-1,], yd_missing])
            assert (
                t_completed.shape[0]
                == y_completed.shape[0]
                == yd_completed.shape[0]
                == n_time_steps
            )
            return t_completed, y_completed, yd_completed

    def plot_trajectory(self, time, y):
        n_subplots = 2

        fig = plt.figure(figsize=(8, n_subplots * 4))
        axes = fig.subplots(nrows=n_subplots, ncols=1, sharex="col")

        axes[0].set_ylabel("delta_i [deg]")
        axes[1].set_ylabel("Delta f_i [Hz]")
        axes[1].set_xlabel("Time [s]")

        handles = [
            patches.Patch(
                color=self.bus_colour_map.colors[bus_index],
                label=f"Bus {bus_index + 1}",
            )
            for bus_index in range(self.n_buses)
        ]

        fig.legend(handles=handles, loc="upper center", ncol=6)

        linestyle_plot = "solid"
        for state_index, state_plot in enumerate(y.T):
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
                    state_plot * self.omega_0_rad_s / (2 * np.pi),
                    color=self.bus_colour_map.colors[bus_index],
                    linestyle=linestyle_plot,
                )
            else:
                pass

        omega_pu_center_of_inertia = np.sum(
            y[:, self.state_omega_generator]
            * self.H_generators_pu.reshape((1, -1))
            / self.H_total_pu,
            axis=1,
        )
        axes[1].plot(
            time,
            omega_pu_center_of_inertia * self.omega_0_rad_s / (2 * np.pi),
            linestyle="dashed",
            color="black",
        )

        axes[-1].set_xlim([time.min(), time.max()])
        return fig
