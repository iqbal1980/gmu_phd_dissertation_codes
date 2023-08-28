import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import time
from numba import njit
import random

#Random number not needed really!
random_number = 1

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.0, Ibg_init=0.0, Ikir_coef=0.94, dt=0.001, 
                                                cm=9.4, a=0.01, dx=0.06, F=9.6485e4, R=8.314e3, K_o=5):
    max_val = 0.51
    min_val = 0.5
    images = []

    for counter in np.arange(min_val, max_val, 0.01):
        ggapval = counter * ggap
        print(f"ggapval={ggapval}")
        A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
        x = A[:, 0]
        y = A[:, 98:135]

        # Plot
        plt.figure()
        plt.plot(x, y, linewidth=3)
        plt.title(f"G gap = {ggapval}")
        images.append(f"Image{ggapval}.png")
        plt.savefig(f"Image{ggapval}.png")
        print("saved 1")
        
        plot_data2_modified(A)
        
        print("saved 2")

    return images

@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o):
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

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
            I_app[99] = 50.0
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * np.log(K_o/150)

            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))

            if kk == 0:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                Vm[kk] += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 98:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 99:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 100:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

    return A

def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]

    distance_m = dx * np.arange(99, 136)
    plt.figure()
    plt.plot(distance_m, D, '.', markersize=8)
    c = np.polyfit(distance_m, D, 1)
    y_est = np.polyval(c, distance_m)
    plt.plot(distance_m, y_est, 'r--', linewidth=2)
    plt.savefig(f"Image2.png")

#Bayesian Optimization Code
def objective(params):
    ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o = params
    
    #Run the simulation with the provided parameters
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
    
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    
    #Compute the polynomial coefficients from the model's result
    coefficients = np.polyfit(distance_m, D, 1)
    
    #Compute the loss as the squared difference between the model's coefficients and the target coefficients
    loss = (coefficients[0] - 2)**2 + (coefficients[1] - 3.2)**2
    
    return loss

#Define the Parameter Space, TODO: need to discuss the value ranges here!
space = [
    Real(0.5, 1.5, name="ggap"),
    Real(0, 1, name="Ibg_init"),
    Real(0.8, 1.0, name="Ikir_coef"),
    Real(0.0005, 0.005, name="dt"),
    Real(8, 11, name="cm"),
    Real(0, 0.05, name="a"),
    Real(0.05, 0.07, name="dx"),
    Real(9.5e4, 1e5, name="F"),
    Real(8.2e3, 8.4e3, name="R"),
    Real(4, 6, name="K_o")
]

#Apply Bayesian Optimization
result = gp_minimize(objective, space, n_calls=50, random_state=0)

# Extract the best parameters
best_parameters = result.x

print("Best parameters:", best_parameters)

