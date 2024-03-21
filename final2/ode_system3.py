import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def safe_log(x):
    return np.log(np.clip(x, 1e-80, None))

def exponential_function(x, a):
    return np.exp(np.where(a * x < 700, a * x, 700))

def exponential_decay_function(x, A, B):
    return A * np.exp(np.where(B * x < 700, B * x, 700))

def ode_system(y, t, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
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
    t = np.linspace(t_span[0], t_span[1], 1000)  # Fixed time points

    sol = odeint(ode_system, y0, t, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val))

    x = t
    y = sol[:, 100:134]

    plt.figure()
    plt.plot(x, y, linewidth=3)
    plt.title(f"G gap = {ggap}")
    plt.xlabel("Time")
    plt.ylabel("Vm")
    plt.show()

# Example usage
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=1, K_o=4.502039575569403, I_app_val=-70, t_span=t_span, Ng=Ng)