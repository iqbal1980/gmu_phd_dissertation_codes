import numpy as np
from deap import base, creator, tools, algorithms
from skopt.space import Real
from numba import njit

random_number = 1

 
param_bounds = [
    (0.1, 35),  # ggap
    (0.1, 1.5),      # Ibg_init
    (0.8, 1.2),  # Ikir_coef
    (8, 11),     # cm
    (0.01, 0.09),  # dx
    (1, 8)       # K_o
]


@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o):
    dt=0.001
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
            elif 98 <= kk <= 100:
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

 
 

import random
import pandas as pd


def generate_dataset(param_bounds, num_samples=100):
    columns = ['ggap', 'Ibg_init', 'Ikir_coef', 'cm', 'dx', 'K_o', 'Output']
    data = []
    i=0;
    for _ in range(num_samples):
        i+=1
       
        # Randomly select parameters within the specified bounds
        g_gap_value = random.uniform(*param_bounds[0])
        Ibg_init = random.uniform(*param_bounds[1])
        Ikir_coef = random.uniform(*param_bounds[2])
        cm = random.uniform(*param_bounds[3])
        dx = random.uniform(*param_bounds[4])
        K_o = random.uniform(*param_bounds[5])

        # Run the simulation
        output = simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o)

        # Assuming the output we're interested in is the last voltage value of the last simulation step
        final_output = output[-1, -1]

        # Append the parameters and output to the data list
        data.append([g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, final_output])
        print(i)

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(data, columns=columns)
    return df

# Example usage:
df = generate_dataset(param_bounds)
df.to_csv('simulation_data.csv', index=False)



 