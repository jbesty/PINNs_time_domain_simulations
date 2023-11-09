import numpy as np

from src import PowerSystem
from utils.config_files.power_system_config import create_power_system_config

if __name__ == "__main__":
    test_system = PowerSystem(create_power_system_config("kundur"))

    y0, yd = test_system.calculate_initial_condition()
    t0 = 0.0
    tfinal = 20

    P_disturbance = np.zeros((test_system.n_buses, 1))
    if test_system.n_buses == 11:
        P_disturbance[6, 0] = 10.0
    elif test_system.n_buses == 39:
        P_disturbance[19, 0] = 10.0
    else:
        raise Exception

    test_system.update_disturbance(P_disturbance)

    results = test_system.simulate_trajectory(
        y0, yd0=yd, t0=t0, tfinal=tfinal, atol=1.0e-10, rtol=1.0e-10
    )
    test_system.plot_trajectory(results[0], results[1]).show()
