import numpy as np


def generator_matrices(H_generator, D_generator, omega_0):
    A = np.array([[0.0, omega_0], [0.0, -D_generator * omega_0]])

    B = np.array([[0.0], [1.0]])

    F = np.array([[0.0], [-1.0]])

    M = np.array([[1.0, 0.0], [0.0, 2 * H_generator * omega_0]])

    U = np.array([[1.0, 0.0]])

    state_type = [True, True]

    state_name = ["delta_generator", "omega_generator"]

    return A, B, F, M, U, state_type, state_name


def load_bus_matrices(D_load, omega_0):
    A = np.array([[0.0]])

    B = np.array([[1.0]])

    F = np.array([[-1.0]])

    M = np.array([[D_load * omega_0]])

    U = np.array([[1.0]])

    state_type = [True]

    state_name = ["delta_load"]

    return A, B, F, M, U, state_type, state_name


def simple_bus_matrices(omega_0):
    A = np.array([[0.0]])

    B = np.array([[1.0]])

    F = np.array([[-1.0]])

    M = np.array([[0.0]])

    U = np.array([[1.0]])

    state_type = [False]

    state_name = ["delta_bus"]

    return A, B, F, M, U, state_type, state_name


def compute_matrix_for_bus_type(bus_type, omega_0):
    if bus_type["type"] == 0:
        A, B, F, M, U, state_type, state_name = generator_matrices(
            bus_type["H_pu"], bus_type["D_pu"], omega_0
        )
    elif bus_type["type"] == 1:
        A, B, F, M, U, state_type, state_name = load_bus_matrices(
            bus_type["D_pu"], omega_0
        )
    elif bus_type["type"] == 2:
        A, B, F, M, U, state_type, state_name = simple_bus_matrices(omega_0)
    else:
        raise Exception("Invalid bus type!")

    return A, B, F, M, U, state_type, state_name
