import numpy as np
from deap import base, creator, tools, algorithms
from skopt.space import Real
from numba import njit
from scipy.optimize import dual_annealing

random_number = 1


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

def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    c = np.polyfit(distance_m, D, 1)


 
def objective(params):
    print("Starting objective function...")
    
    # Clip each parameter to lie within its bounds.
    for i in range(len(params)):
        low, high = param_bounds[i]
        params[i] = np.clip(params[i], low, high)
        
    ggap, Ibg_init, Ikir_coef, cm, dx, K_o = params
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)
    dx = 0.06
    D = np.abs(A[399998, 98:135] - A[99000, 98:135]) / np.abs(A[99000, 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    coefficients = np.polyfit(distance_m, D, 1)
    loss = (coefficients[0] + 0.0000000000001)**2 + (coefficients[1] - 0.6)**2
    
    if np.isnan(loss) or np.isinf(loss):
        print("NAN loss detected with params:", params)
        loss = 1e10
        
    print("Completed objective function!")
    return loss  # Return a scalar value instead of a tuple


 
param_bounds = [
    (0.1, 35),  # ggap
    (0.1, 1.5),      # Ibg_init
    (0.8, 1.2),  # Ikir_coef
    (8, 11),     # cm
    (0.01, 0.09),  # dx
    (1, 8)       # K_o
]

 

def main():
    print("Starting Simulated Annealing...")
    result = dual_annealing(objective, bounds=param_bounds, maxiter=1000)
    print("Optimization completed!")

    print("Best parameters are: ", result.x, " with loss: ", result.fun)

    return result

if __name__ == "__main__":
    main()





