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

    # Extract the equations from the solution
    equations = sol.sol

    # Create a mathematical function using the equations
    def solution_function(t):
        return equations(t)[:, 100:134]



    # Evaluate the solution function at the same time points as the original solution
    t_eval = sol.t
    y_eval = solution_function(t_eval)

    # Plot the solution
    plt.figure()
    plt.plot(t_eval, y_eval, linewidth=3)
    plt.title(f"G gap = {ggap}")
    plt.xlabel("Time")
    plt.ylabel("Vm")
    plt.show()

    return solution_function

# Example usage
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
solution_func = HK_deltas_vstim_vresponse_graph_modified_v2(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=1, K_o=4.502039575569403, I_app_val=-70, t_span=t_span, Ng=Ng)



t_eval = 250  # Evaluate the solution at t=250
y_eval = solution_func(t_eval)
print(f"Solution at t={t_eval}: {y_eval}")


t_eval = [100, 200, 300, 400, 500]  # Evaluate the solution at multiple time points
y_eval = solution_func(t_eval)
plt.figure()
plt.plot(t_eval, y_eval, marker='o')
plt.xlabel("Time")
plt.ylabel("Vm")
plt.show()