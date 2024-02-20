import numpy as np
from numba import njit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

@njit
def safe_log(x):
    epsilon = 1e-10
    if x <= 0:
        return np.log(epsilon)
    else:
        return np.log(x)

@njit
def exponential_func(x, a):
    max_exp = safe_log(np.finfo(np.float64).max)
    min_exp = safe_log(np.finfo(np.float64).tiny)

    exponent = a * x

    if exponent > max_exp:
        return np.float64(np.inf)
    elif exponent < min_exp:
        return 0.0
    else:
        return np.exp(exponent)

@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)
    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = 50.0
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o/150)

            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_func((Vm[kk] - E_K - 25) / 7, 1)))

            if np.isnan(Vm[kk]) or np.isinf(Vm[kk]):
                print(f"NaN or Inf found in Vm at iteration {j}, cell {kk}")
                return A

            if kk == 0:
                Vm[kk] += 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                Vm[kk] += eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in [98, 99, 100]:
                Vm[kk] += eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                Vm[kk] += eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

        A[j, 0] = t
        A[j, 1:] = Vm

    return A

def plot_data2_modified(A, ggap, withReference=False):  
    cellLength = 60  # in Microns
    end_values = A[399998, 101:135]
    start_values = A[99000, 101:135]

    if np.any(start_values == 0) or np.any(np.isnan(start_values)):
        print("Zero or NaN encountered in start values. Skipping curve fitting.")
        return None

    D = np.abs(end_values - start_values) / np.abs(start_values[0])
    distance_m = cellLength * np.arange(34)

    if np.any(np.isnan(D)):
        valid_indices = ~np.isnan(D)
        D = D[valid_indices]
        distance_m = distance_m[valid_indices]

    if len(D) == 0:
        print("Empty D array after NaN removal. Skipping curve fitting.")
        return None

    try:
        popt, pcov = curve_fit(exponential_func, distance_m, D, maxfev=1000)

        residuals = D - exponential_func(distance_m, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((D - np.mean(D))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        plt.figure()
        plt.ylim(-0.5, 12)
        plt.plot(distance_m, D, '.', markersize=8)
        
        x_fit = np.linspace(min(distance_m), max(distance_m), 100)
        y_fit = exponential_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r--', linewidth=2)
        
        if withReference:
            y_ref = np.exp(-0.003 * distance_m)
            plt.plot(distance_m, y_ref, 'b-', linewidth=2)
            plt.legend(['Data', f'Fit: y = e^({popt[0]:.3f}x), $R^2$ = {r_squared:.4f}', 
                        'Reference: y = e^(-0.003x)'], loc='upper left')

        plt.title(f'Exponential Fit for Ggap = {ggap}')
        plt.xlabel('Distance (µm)')
        plt.ylabel('Delta Value')
        plt.savefig(f"Image2_{ggap}{'_ref' if withReference else ''}.png")

    except Exception as e:
        print(f"Error in curve fitting: {e}")
        return None


def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o):
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)
    x = A[:, 0]
    y = A[:, 101:135]

    plt.figure()
    plt.plot(x, y, linewidth=3)
    plt.title(f"G gap = {ggap}")
    plt.savefig(f"Image{ggap}.png")
    print("saved 1")
        
    plot_data2_modified(A, ggap, False)
    plot_data2_modified(A, ggap, True)
 
    print("saved 2")

    return

Ibg_init_val = 0.7 * 0.94 
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.3015830801507125, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=1, K_o=3)
