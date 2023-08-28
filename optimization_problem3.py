import numpy as np
from skopt import Optimizer
from skopt.space import Real
from skopt.acquisition import gaussian_ei
from numba import njit
import random
from sklearn.ensemble import RandomForestRegressor

random_number = 1

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.0, Ibg_init=0.0, Ikir_coef=0.94, dt=0.001, 
                                                cm=9.4, a=0.01, dx=0.06, F=9.6485e4, R=8.314e3, K_o=5):
    max_val = 0.51
    min_val = 0.5
    images = []
    for counter in np.arange(min_val, max_val, 0.01):
        ggapval = counter * ggap
        A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
        plot_data2_modified(A)
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

@njit(parallel=False)
def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    c = np.polyfit(distance_m, D, 1)

def objective(params):
    ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o = params
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    coefficients = np.polyfit(distance_m, D, 1)
    loss = (coefficients[0] - 2)**2 + (coefficients[1] - 3.2)**2
    return loss

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

# Custom RandomForest that can return standard deviation
class RandomForestWithUncertainty(RandomForestRegressor):
    def predict(self, X, return_std=False):
        preds = []
        for estimator in self.estimators_:
            preds.append(estimator.predict(X))
        preds = np.array(preds)
        
        if return_std:
            return preds.mean(axis=0), preds.std(axis=0)
        return preds.mean(axis=0)

# Bayesian optimization with custom random forest regressor
optimizer = Optimizer(space, base_estimator=RandomForestWithUncertainty(), acq_func="EI", acq_optimizer="sampling")
for i in range(300):
    next_x = optimizer.ask()
    f_val = objective(next_x)
    optimizer.tell(next_x, f_val)

best_parameters = optimizer.Xi[np.argmin(optimizer.yi)]
print("Best parameters:", best_parameters)

