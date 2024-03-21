import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

# Constants
MIN_VALUE = -80
MAX_VALUE = 40
MIN_VM = -80  # Minimum physiologically reasonable value for Vm
MAX_VM = 40   # Maximum physiologically reasonable value for Vm

# Safe logarithm function
@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)

# Exponential functions with @njit for performance
@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
    # Constants within the function
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y

    # Calculating currents
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))

    # Application current adjustment
    I_app = np.zeros_like(Vm)
    I_app[99] = I_app_val if 100 <= t <= 400 else 0.0

    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])

    return dVm_dt

def run_simulation(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index):
    y0 = np.ones(Ng) * (-33)  # Initial condition

    # Solving the ODE with specified method
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val), method='Radau')

    # Extract the voltage for the specified time and cell.
    time_idx = np.argmin(np.abs(sol.t - time_in_ms))  # Find index of the closest time point to time_in_ms
    voltage = sol.y[cell_index, time_idx]  # Extract voltage

    return voltage

# Example usage
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
time_in_ms = 300  # Example time in ms
cell_index = 100  # Example cell index

voltage = run_simulation(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=1, K_o=4.502039575569403, I_app_val=-70, t_span=t_span, Ng=Ng, time_in_ms=time_in_ms, cell_index=cell_index)
print(f"Voltage at time {time_in_ms}ms for cell index {cell_index}: {voltage}")



