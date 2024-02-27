import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.extending import overload
from scipy.optimize import curve_fit
from numba import generated_jit
from numba import types

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



@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o_left, K_o_right):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(eki1))
    
    eki2 = dt / cm
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(eki1))

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)
    distance_m = np.zeros(Ng)
    vstims = np.zeros(Ng)
    vresps = np.zeros(Ng)

    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)
    
    K_o_array = np.ones(Ng) * K_o_left
    K_o_array[Ng//2:] = K_o_right  # Change [K]o in the right half of the domain
    #print(K_o_array)

    for j in range(loop):
        t = j * dt
        #if 100 <= t <= 400:
        I_app[100:200] = 5.0
        #else:
        #    I_app[99] = 0.0

        for kk in range(Ng):
            K_o = K_o_array[kk]
            E_K = (R * 293 / F) * safe_log(K_o/150)
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

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

        # Debugging: Check for NaNs in Vm
        #if np.any(np.isnan(Vm)):
        #    print(f"NaN detected in Vm at iteration {j}")
        #    print(f"Vm: {Vm}")

    return A

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o_left, K_o_right):
    ggapval = ggap
    print(f"ggapval={ggapval}")
    
    # Call the simulation function to get data
    A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, cm, dx, K_o_left, K_o_right)
    

    x = A[:, 0]
    y = A[:, 60:140:10]
    z = A[599999, :]
    
    plt.figure()
    legends = np.arange(60, 140, 10) 
    for i in range(len(legends)):
        plt.plot(x, y[:, i], linewidth=3, label=f'Plot {legends[i]}')
    
    plt.title(f"G gap = {ggapval}")
    plt.legend()
    plt.savefig(f"Image{ggapval}.png")
    print("saved 1")


    plot_data2_modified(A, ggap, False)
    plot_data2_modified(A, ggap, True)

    print("saved 2")
    
    np.savetxt(f"x_data_{ggapval}.csv", x, delimiter=",")
    np.savetxt(f"y_data_{ggapval}.csv", y, delimiter=",")

    xy_combined = np.column_stack((x, y))
    
    np.savetxt(f"z_data_{ggapval}.csv", z, delimiter=",")
    
    np.savetxt(f"xy_combined_data_{ggapval}.csv", xy_combined, delimiter=",")
    
    return ["Image" + str(ggapval) + ".png", "Image2_" + str(ggap) + ".png", "Image2_" + str(ggap) + "_ref.png"]



def plot_data2_modified(A, ggap, withReference=False):
    cellLength = 60  # in Microns
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / np.abs(A[99000, 101:135])[0]
    
    print(D)
    # Set any value of D that is greater than 1 to 1
    #D = np.where(D > 10, 0.8, D)
    
    #print(D)
    distance_m = cellLength * np.arange(102-102, 136-102)
    
    A_initial = D[0]
    B_initial = np.log(D[1] / D[0]) / (distance_m[1] - distance_m[0])
    
    print(distance_m)

    # Debugging: Check for NaNs in D and distance_m before curve_fit
    if np.any(np.isnan(D)) or np.any(np.isnan(distance_m)):
        print("NaN detected in D or distance_m")
        print(f"D: {D}, distance_m: {distance_m}")
        return None

    try:
        popt, pcov = curve_fit(exponential_decay_function, distance_m, D, p0=[A_initial, B_initial])
    except RuntimeError as e:
        print("Error in curve fitting:", e)
        return None

    A, B = popt
    residuals = D - exponential_decay_function(distance_m, A, B)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((D - np.mean(D))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.figure()
    plt.ylim(-0.5, 3)
    plt.plot(distance_m, D, '.', markersize=8)

    # Plot the fitted exponential curve
    x_fit = np.linspace(min(distance_m), max(distance_m), 100)
    y_fit = exponential_decay_function(x_fit, A, B)
    plt.plot(x_fit, y_fit, 'r--', linewidth=2)

    if withReference:
        # Plot the reference line
        y_ref = np.exp(-0.003 * distance_m)
        plt.plot(distance_m, y_ref, 'b-', linewidth=2)
        plt.legend(['Data', f'Fit: y = e^({popt[0]:.3f}x), $R^2$ = {r_squared:.4f}',
                    'Reference: y = e^(-0.003x)'], loc='upper left')

    plt.title(f'Chart for Ggap = {ggap}')
    plt.savefig(f"Image2_{ggap}{'_ref' if withReference else ''}.png")

    return f"Image2_{ggap}{'_ref' if withReference else ''}.png"

#Dr Jafri email about g*10000/36
Ibg_init_val = 0.7 * 0.94

HK_deltas_vstim_vresponse_graph_modified_v2(ggap=4, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.001, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.01, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.1, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.5, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=5, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=10, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=15, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=20, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=25, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=30, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=35, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=100, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=500, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1000, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o_left=8, K_o_right=4)

