import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from numba.extending import overload
from scipy.optimize import curve_fit
from numba import generated_jit
from numba import types
import random

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

MIN_VM = -80 # Minimum physiologically reasonable value for Vm
MAX_VM = 40 # Maximum physiologically reasonable value for Vm

@njit(parallel=False)
def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index):
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))
    I_app = np.zeros_like(Vm)
    I_app[activated_cell_index] = I_app_val if stimulation_time_start <= t <= stimulation_time_end else 0.0
    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
    return dVm_dt








def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, what_cell_is_activated):
    y0 = np.ones(Ng) * y0_init # Initial condition
    #sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='Radau')#, max_step=0.01)
    #sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='RK23')#, max_step=0.01)
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='RK45')#, max_step=0.01)
    return sol







    
    
def generate_random_params(param_bounds):
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    time_in_s = np.random.choice(param_bounds[5])
    y0_init = np.random.choice(param_bounds[6])
    Ng = np.random.choice(param_bounds[7])
    activated_cell_index = int(np.random.choice(param_bounds[8]))  # Convert to integer

    print(y0_init)
    print(time_in_s)
    stimulation_time_start = random.uniform(0, int(0.75*time_in_s))
    stimulation_time_end = random.uniform(stimulation_time_start + 85, min(stimulation_time_start + 90 + random.uniform(0, 300), time_in_s))
    
    print("stimulation_time_start="+str(int(stimulation_time_start)))
    print("stimulation_time_end="+str(int(stimulation_time_end)))
    print("activated_cell_index="+str(activated_cell_index))

    t_span = (0, time_in_s)
    Ng = 200

    return ggap, Ikir_coef, cm, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index

# Example usage
param_bounds = [
    np.linspace(0.1, 35, 10),  # ggap (10 values) 0
    np.linspace(0.90, 0.96, 5),  # Ikir_coef (5 values) 1
    np.linspace(8, 11, 4),  # cm (4 values) 2
    np.linspace(1, 8, 8),  # K_o (8 values) 3
    np.linspace(-70, 70, 35),  # I_app_val (15 values) 4
    np.linspace(600, 600, 1),  # time_in_s (100 values) 5
    np.linspace(-33, -33, 1),  # y0_init (20 values) 6
    np.linspace(200, 200, 1),  # ng number of cells (200) 7
    np.arange(85, 115)  # activated_cell_index (all values from 10 to 195) 8  
]

Ibg_init_val = 0.7 * 0.94
dx = 1

ggap, Ikir_coef, cm, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index = generate_random_params(param_bounds)

sol = HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index)





#print("sol.t="+str(sol.t))
#print("sol.y="+str(sol.y))


print("len(sol.t)="+str(len(sol.t)))
print("len(sol.y)="+str(len(sol.y)))



# Plotting the random simulation
plt.figure(figsize=(10, 6))
start_plot_cell = int(activated_cell_index)
end_plot_cell = int(activated_cell_index+34)
print(len(sol.t))
print(len(sol.y))
plt.plot(sol.t, sol.y[start_plot_cell:end_plot_cell].T, linewidth=1)
plt.title(f"Random Simulation\nG gap = {ggap:.2f}, Ikir_coef = {Ikir_coef:.2f}, cm = {cm:.2f}, K_o = {K_o:.2f}, I_app_val = {I_app_val:.2f}")
plt.xlabel("Time (s)")
plt.ylabel("Vm (mV)")
plt.grid(True)
plt.xlim(0, 600)  # Set the x-axis to display from 0 to 600 seconds
plt.show()