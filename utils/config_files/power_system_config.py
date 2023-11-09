from collections import ChainMap
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

from definitions import PROJECT_PATH


def create_power_system_config(power_system_name):
    implemented_cases = ["ieee39", "kundur"]
    assert (
        power_system_name in implemented_cases
    ), f"Specified case is currently not implemented! Choose among {implemented_cases}"

    f_0_Hz: float = 60.0
    omega_0_rad_s: float = f_0_Hz * 2 * np.pi

    dict_config_parameters_static = create_config_parameters_static_from_matpower_case(
        power_system_name
    )

    generator_set_points = np.array(
        dict_config_parameters_static["P_set_point_generators_pu"][
            dict_config_parameters_static["generator_indices"], 0
        ]
    )
    load_set_points = -np.array(
        dict_config_parameters_static["P_set_point_loads_pu"][
            dict_config_parameters_static["load_indices"], 0
        ]
    )

    if power_system_name == "kundur":
        H_generators_pu = np.array([6.5, 6.5, 6.175, 6.175])
        alpha_load = 1.0
        alpha_generator = 0.05
    elif power_system_name == "ieee39":
        alpha_load = 0.2
        alpha_generator = 0.05
        H_generators_pu = np.array(
            [500.0, 30.3, 35.8, 38.6, 26.0, 34.8, 26.4, 24.3, 34.5, 42.0]
        )
    else:
        raise Exception("Values for damping and inertia are not provided!")

    D_generators_pu = (alpha_generator * omega_0_rad_s) / generator_set_points
    D_loads_pu = load_set_points / (alpha_load * omega_0_rad_s)

    if power_system_name == "kundur":
        display_name = "11-bus system"
        cmap_buses = matplotlib.colors.ListedColormap(
            [
                "red",
                "orangered",
                "blue",
                "cornflowerblue",
                "tomato",
                "chocolate",
                "gold",
                "black",
                "green",
                "teal",
                "dodgerblue",
            ]
        )
    elif power_system_name == "ieee39":
        display_name = "39-bus system"
        colors_tab20b = matplotlib.colormaps.get("tab20b").colors
        colors_tab20c = matplotlib.colormaps.get("tab20c").colors
        cmap_buses = matplotlib.colors.ListedColormap(
            colors_tab20b + colors_tab20c[:-1]
        )
    else:
        raise Exception("Please specify a color scheme for the system!")

    assert cmap_buses.N == dict_config_parameters_static["n_buses"]
    assert len(H_generators_pu) == dict_config_parameters_static["n_generators"]
    assert len(D_generators_pu) == dict_config_parameters_static["n_generators"]
    assert len(D_loads_pu) == dict_config_parameters_static["n_loads"]

    dict_config_parameters_dynamic = dict(
        {
            "H_generators_pu": H_generators_pu,
            "D_generators_pu": D_generators_pu,
            "D_loads_pu": D_loads_pu,
            "f_0_Hz": f_0_Hz,
            "omega_0_rad_s": omega_0_rad_s,
        }
    )

    dict_config_parameters_visualisation = dict(
        {"cmap_buses": cmap_buses, "display_name": display_name}
    )

    dict_config_parameters = SimpleNamespace(
        **dict(
            ChainMap(
                *[
                    dict_config_parameters_static,
                    dict_config_parameters_dynamic,
                    dict_config_parameters_visualisation,
                ]
            )
        )
    )

    return dict_config_parameters


def create_config_parameters_static_from_matpower_case(power_system_name):
    buses_mat, generators_mat, branches_mat = import_matpower_acopf_case(
        power_system_name
    )
    n_buses: int = len(buses_mat)
    n_generators: int = len(generators_mat)

    baseMVA: float = 100.0

    generator_indices = (
        generators_mat["bus"].values - 1
    )  # Matlab indexing starts with 1!!!
    slack_bus = np.where(buses_mat["type"].values == 3)[0][0]
    load_indices = np.setdiff1d(
        np.where(buses_mat["Pd"].values > 0.0)[0], generator_indices
    )

    P_set_point_loads_pu = -buses_mat["Pd"].values.reshape((-1, 1)) / baseMVA
    n_loads: int = len(load_indices)

    Ybus, _, _ = compute_admittance_matrix(
        branches=branches_mat, buses=buses_mat, baseMVA=baseMVA
    )

    V_complex = np.reshape(
        buses_mat["Vm"].values * np.exp(1j * np.deg2rad(buses_mat["Va"].values)),
        (-1, 1),
    )

    S_bus = np.diag(V_complex[:, 0]) @ np.conj(Ybus) @ np.conj(V_complex)

    delta_initial = np.angle(V_complex)[:, 0] - np.angle(V_complex)[slack_bus, 0]

    P_set_point_generators_pu = np.real(S_bus) - P_set_point_loads_pu

    config_parameters_static = dict(
        {
            "n_buses": n_buses,
            "n_generators": n_generators,
            "n_loads": n_loads,
            "generator_indices": generator_indices,
            "load_indices": load_indices,
            "Y_bus_pu": Ybus,
            "delta_initial": delta_initial,
            "P_set_point_generators_pu": P_set_point_generators_pu,
            "P_set_point_loads_pu": P_set_point_loads_pu,
            "V_magnitude_pu": np.abs(V_complex),
            "slack_bus": slack_bus,
        }
    )
    return config_parameters_static


def import_matpower_acopf_case(case_name):
    available_cases = ["ieee39", "kundur"]
    if case_name not in available_cases:
        raise Exception(
            f"Case not available. Choose among the following:\n {available_cases}"
        )

    bus_labels = [
        "bus",
        "type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area",
        "Vm",
        "Va",
        "baseKV",
        "zone",
        "Vmax",
        "Vmin",
        "lambda_vmax",
        "lambda_vmin",
        "mu_vmax",
        "mu_vmin",
    ]
    generator_labels = [
        "bus",
        "Pg",
        "Qg",
        "Qmax",
        "Qmin",
        "Vg",
        "mBase",
        "status",
        "Pmax",
        "Pmin",
        "Pc1",
        "Pc2",
        "Qc1min",
        "Qc1max",
        "Qc2min",
        "Qc2max",
        "ramp_agc",
        "ramp_10",
        "ramp_30",
        "ramp_q",
        "apf",
        "mu_pmax",
        "mu_pmin",
        "mu_qmax",
        "mu_qmin",
    ]
    branch_labels = [
        "fbus",
        "tbus",
        "r",
        "x",
        "b",
        "rateA",
        "rateB",
        "rateC",
        "ratio",
        "angle",
        "status",
        "angmin",
        "angmax",
        "pf",
        "qf",
        "pt",
        "qt",
        "mu_sf",
        "mu_st",
        "mu_angmin",
        "mu_angmax",
    ]

    case_path = PROJECT_PATH / "utils" / "matpower_case_files"

    matlab_buses = pd.read_csv(case_path / f"{case_name}_bus.csv", names=bus_labels)
    matlab_generators = pd.read_csv(
        case_path / f"{case_name}_gen.csv", names=generator_labels
    )
    matlab_branches = pd.read_csv(
        case_path / f"{case_name}_branch.csv", names=branch_labels
    )

    return matlab_buses, matlab_generators, matlab_branches


def compute_admittance_matrix(branches, buses, baseMVA):
    n_buses = len(buses["bus"].values)
    n_branches = len(branches["fbus"].values)
    # for each branch, compute the elements of the branch admittance matrix where
    #
    #      | If |   | Yff  Yft |   | Vf |
    #      |    | = |          | * |    |
    #      | It |   | Ytf  Ytt |   | Vt |
    #
    branch_status = branches["status"].values  # ones at in-service branches
    Ys = branch_status / (
        branches["r"].values + 1j * branches["x"].values
    )  # series admittance
    Bc = branch_status * branches["b"].values  # line charging susceptance
    tap = branches["ratio"].values.copy()
    tap[tap == 0] = 1  # set all zero tap ratios (lines) to 1
    tap_complex = tap * np.exp(
        1j * np.pi / 180 * branches["angle"].values
    )  # add phase shifters
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap_complex * np.conj(tap_complex))
    Yft = -Ys / np.conj(tap_complex)
    Ytf = -Ys / tap_complex

    Ysh = (
        buses["Gs"].values + 1j * buses["Bs"].values
    ) / baseMVA  # vector of shunt admittances

    Cf = np.zeros((n_branches, n_buses))
    Ct = np.zeros((n_branches, n_buses))
    for ii, (bus_from, bus_to) in enumerate(
        zip(branches["fbus"].values - 1, branches["tbus"].values - 1)
    ):
        Cf[ii, bus_from] = 1
        Ct[ii, bus_to] = 1

    # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    # at each branch's 'from' bus, and Yt is the same for the 'to' bus end

    Yf = np.diag(Yff) @ Cf + np.diag(Yft) @ Ct
    Yt = np.diag(Ytf) @ Cf + np.diag(Ytt) @ Ct

    Ybus = Cf.T @ Yf + Ct.T @ Yt + np.diag(Ysh)

    return Ybus, Yf, Yt
