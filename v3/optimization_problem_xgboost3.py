import numpy as np
from skopt import Optimizer
from skopt.space import Real
from skopt.acquisition import gaussian_ei
from numba import njit
import random
import xgboost as xgb

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


def objective(params):
    ggap, Ibg_init, Ikir_coef, cm, dx, K_o = params
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)
    dx = 0.06
    #D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    #D = np.abs(A[399998, 98:135] - A[99000, 98:135]) / np.abs(A[99000, 98:135])[0]
    D = np.abs(A[99000, 98:135])[0] / np.abs(A[399998, 98:135] - A[99000, 98:135])
    distance_m = dx * np.arange(99, 136)
    coefficients = np.polyfit(distance_m, D, 1)
    #loss = (coefficients[0] - 2)**2 + (coefficients[1] - 3.2)**2
    loss = (coefficients[0] + 0.0000000000001)**2 + (coefficients[1] - 0.6)**2 #  
    
    if np.isnan(loss) or np.isinf(loss):
        print("Invalid loss value:", loss, "for parameters:", params)
        loss = 1e10  # Replace with a large but manageable value
    
    return loss

space = [
    Real(0.1, 35, name="ggap"),
    Real(0.1, 1.5, name="Ibg_init"),
    Real(0.3, 1.2, name="Ikir_coef"),
    Real(8, 11, name="cm"),
    Real(0.01, 0.09, name="dx"),
    Real(1, 8, name="K_o")
]





# Custom XGBoost regressor that can return standard deviation
class XGBoostWithUncertainty(xgb.XGBRegressor):
    def predict(self, X, return_std=False):
        preds = super().predict(X, output_margin=True)
        
        if return_std:
            # Convert margin prediction to probability using logistic function
            prob = 1.0 / (1.0 + np.exp(-preds))
            std_dev = prob * (1.0 - prob)  # For binary logistic regression
            
            # Clip std_dev to prevent extreme values
            std_dev = np.clip(std_dev, 1e-5, 1e5)
            
            return preds, std_dev
        return preds

#Optimization with custom XGBoost regressor
optimizer = Optimizer(space, base_estimator=XGBoostWithUncertainty(), acq_func="EI", acq_optimizer="sampling")
for i in range(2400):#300
    next_x = optimizer.ask()
    f_val = objective(next_x)
    optimizer.tell(next_x, f_val)

best_parameters = optimizer.Xi[np.argmin(optimizer.yi)]
print("Best parameters:", best_parameters)