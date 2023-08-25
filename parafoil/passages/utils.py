import numpy as np

def get_wall_distance(rho: float, Uf: float, mu: float, L: float, y_plus_desired: float):
    # TODO: add docs
    Re = (rho * Uf * L) / mu
    C_f = 0.026 * np.power(Re, -(1.0 / 7.0))
    # C_f = 0.0576 * np.pow(Re, -1.0 / 5.0)
    # C_f = 0.370 * np.pow(np.log(Re) / np.log(10), -2.584)
    # C_f = np.pow((2 * np.log(Re) / np.log(10) - 0.65), -2.3)

    tau_w = C_f * 0.5 * rho * Uf **2 
    U_f = np.sqrt(tau_w / rho)

    return (y_plus_desired * mu) / (U_f * rho)