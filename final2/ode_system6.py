import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def safe_log(x):
    return np.log(np.clip(x, 1e-80, None))

def exponential_function(x, a):
    return np.exp(np.where(a * x < 700, a * x, 700))

def exponential_decay_function(x, A, B):
    return A * np.exp(np.where(B * x < 700, B * x, 700))

def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y

    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))

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

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng):
    y0 = np.ones(Ng) * (-33)  # Initial condition

    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val), method='Radau', dense_output=True)

    t_eval = np.linspace(t_span[0], t_span[1], 300)  # Create an array of time points where solution is evaluated
    y_eval = sol.sol(t_eval)

    # Adjusted plotting section
    plt.figure()
    for curve_idx in range(y_eval.shape[0]):  # Iterate over each curve (each cell in this context)
        if 100 <= curve_idx < 134:  # Adjust this range as necessary
            plt.plot(t_eval, y_eval[curve_idx, :], linewidth=3, label=f'Cell {curve_idx}')
    plt.title(f"G gap = {ggap}")
    plt.xlabel("Time")
    plt.ylabel("Vm")
    plt.legend()
    plt.show()

    # Return a function to evaluate solution at new points
    def solution_function(new_t_eval):
        return sol.sol(new_t_eval)

    return solution_function

# Example usage
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
solution_func = HK_deltas_vstim_vresponse_graph_modified_v2(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=1, K_o=4.502039575569403, I_app_val=-70, t_span=t_span, Ng=Ng)

# Evaluate the solution at specific time points# Evaluate the solution at t=250
t_eval_single = 250
y_eval_single = solution_func(np.array([t_eval_single]))

# Since y_eval_single will have shape (Ng, 1), we can plot a subset or all depending on interest
plt.figure()
for curve_idx in range(y_eval_single.shape[0]):
    if 100 <= curve_idx < 134:  # Adjust this range as necessary
        plt.plot(t_eval_single, y_eval_single[curve_idx, 0], 'o', label=f'Cell {curve_idx}')
plt.title("Solution at t=250")
plt.xlabel("Cell Index")
plt.ylabel("Vm")
plt.legend()
plt.show()



# Evaluate the solution at multiple time points
t_eval_multiple = np.array([100, 200, 300, 400, 500])
y_eval_multiple = solution_func(t_eval_multiple)

# Plotting each curve at the specified time points
plt.figure()
for curve_idx in range(y_eval_multiple.shape[0]):
    if 100 <= curve_idx < 134:  # Adjust this range as necessary
        plt.plot(t_eval_multiple, y_eval_multiple[curve_idx, :], '-o', label=f'Cell {curve_idx}')

plt.title("Solution at Multiple Time Points")
plt.xlabel("Time")
plt.ylabel("Vm")
plt.legend()
plt.show()



# Select a subset of cells for clearer visualization
selected_cells = range(10, 180)  # Adjust this as necessary

# Increase the plot size for better clarity
plt.figure(figsize=(10, 6))

# Plotting only the selected subset of cells
for curve_idx in selected_cells:
    plt.plot(t_eval_multiple, y_eval_multiple[curve_idx, :], '-o', label=f'Cell {curve_idx}')

plt.title("Solution at Multiple Time Points")
plt.xlabel("Time")
plt.ylabel("Vm")
plt.legend()
plt.show()



# Define a higher resolution time array
t_eval_multiple = np.linspace(50, 150, 200)  # For example, 200 points between 100 and 500
y_eval_multiple = solution_func(t_eval_multiple)

# Plotting each curve at the specified time points
plt.figure(figsize=(10, 6))  # Increase figure size for better visibility
for curve_idx in range(10, 34):  # This is the range of cells you appear to be plotting
    plt.plot(t_eval_multiple, y_eval_multiple[curve_idx, :], '-o', label=f'Cell {curve_idx}')

plt.title("Solution at Multiple Time Points")
plt.xlabel("Time")
plt.ylabel("Vm")
plt.legend()
plt.show()

