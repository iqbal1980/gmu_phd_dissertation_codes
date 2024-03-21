import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.extending import overload
from scipy.optimize import curve_fit
from numba import generated_jit
from numba import types
from gplearn.genetic import SymbolicRegressor

# Random number not needed really
random_number = 1

global_dx = 1  # No spatial discretisation in network model

# Constants for safe_log and safe_exponential

MIN_VALUE = -80
MAX_VALUE = 40

@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)
 
 
@njit
def exponential_function(x, a):
    return np.exp(a * x) 
 



@njit
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)



MIN_VM = -80  # Minimum physiologically reasonable value for Vm
MAX_VM = 40  # Maximum physiologically reasonable value for Vm





njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, dt, loop):    
    #print(f"Time Step (dt): {dt}")
    #print(f"Loop: {loop}")

    F = 9.6485e4
    R = 8.314e3
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value

    # Calculate the scaling factor based on the time step
    scaling_factor = dt / 0.001  # Assuming the original time step was 0.001

    #print(scaling_factor)
    eki1 = ((g_gap * dt) / (dx**2 * cm)) * scaling_factor
    eki2 = (dt / cm) * scaling_factor

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)
    distance_m = np.zeros(Ng)
    vstims = np.zeros(Ng)
    vresps = np.zeros(Ng)
    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)

    for j in range(loop):
        t = j * dt

        if 100 <= t <= 400:
            I_app[99] = I_app_val
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o / 150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm += random_number * 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                new_Vm += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VM), MIN_VM)

        distance_m[:] = np.arange(Ng) * dx
        vstims[99] = Vm[99]
        vresps[:] = Vm[:]
        vresps[99] = 0.0

        A[j, 0] = t
        A[j, 1:] = Vm

    return A

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o,  I_app_val, dt, loop):
    ggapval = ggap
    print(f"ggapval={ggapval}")
    A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, dt, loop)
  
    x = A[:, 0]
    y = A[:, 101:135]

    # Plot
    plt.figure()
    plt.plot(x, y, linewidth=3)
    plt.title(f"G gap = {ggapval}")
    plt.savefig(f"Image{ggapval}.png")
    print("saved 1")
	
Ibg_init_val = 0.7 * 0.94
simulation_time = 600
dt_init = 0.05
loop_init = int(simulation_time / dt_init)



HK_deltas_vstim_vresponse_graph_modified_v2(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=global_dx, K_o=4.502039575569403, I_app_val=-70, dt=dt_init, loop=loop_init)

 