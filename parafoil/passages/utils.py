import numpy as np
from paraflow import FlowState
from parafoil.airfoils.airfoil import Airfoil

# def yPlus2Ds(flow_state: FlowState, airfoil: Airfoil):
#     # Compute a wall spacing value 
#     # for given flow conditions and a desired Y+

#     Uinf = flow_state.freestream_velocity
#     rho = flow_state.rho_mass()
#     L = airfoil.chord_length
#     mu = flow_state.mu()

#     # Compute Ds
#     Rex = (rho * Uinf * L) / mu
#     Cf = 0.026 * np.power(Rex, -(1.0/7.0))
#     tauWall = Cf * rho * Uinf * Uinf * 0.5
#     Ufric = np.sqrt(tauWall / rho)
#     Ds = (yPlus * mu) / (Ufric * rho)

# %%
import math

# rho = 1.225  # density
# u_freestream = 10.0  # freestream velocity
# L = 1.0  # characteristic length
# mu = 1.7894e-5  # dynamic viscosity
# y_plus_desired = 1.0  # desired wall distance

rho = 1.4  # density
u_freestream = 200.0  # freestream velocity
L = 1.0  # characteristic length
mu = 2E-5  # dynamic viscosity
y_plus_desired = 1.0  # desired wall distance



Re = (rho * u_freestream * L) / mu
C_f = 0.026 * math.pow(Re, -(1.0 / 7.0))
C_f = 0.0576 * math.pow(Re, -1.0 / 5.0)
C_f = 0.370 * math.pow(math.log(Re) / math.log(10), -2.584)
C_f = math.pow((2 * math.log(Re) / math.log(10) - 0.65), -2.3)

tau_w = C_f * 0.5 * rho * u_freestream * u_freestream
U_f = math.sqrt(tau_w / rho)

wall_distance_estimation = (y_plus_desired * mu) / (U_f * rho)

Re_exp = "{:.1e}".format(Re)
wall_distance_estimation_exp = "{:.1e}".format(wall_distance_estimation)

print("Re: ", Re_exp)
print("Wall Distance Estimation: ", wall_distance_estimation_exp)
# %%
