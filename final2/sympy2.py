# First, ensure you have the necessary packages installed:
# numpy, matplotlib, scipy, numba, pandas, and pysr

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from numba import njit
from pysr import PySRRegressor

# Define the necessary functions and constants from your simulation setup
@njit
def safe_log(x):
    if x <= 0:
        return -80  # Using -80 as the minimum value as per your setup
    return np.log(x)

@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
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

def run_simulation(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index):
    y0 = np.ones(Ng) * (-33)
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val), method='Radau')
    time_idx = np.argmin(np.abs(sol.t - time_in_ms))
    voltage = sol.y[cell_index, time_idx]
    return voltage

# Generate a dataset
# Note: In a real application, consider generating a more comprehensive dataset
data = []
for ggap in np.linspace(0.1, 35, 10):  # Simplified example; adjust as needed
    for Ikir_coef in np.linspace(0.90, 0.96, 5):
        for cm in np.linspace(8, 11, 4):
            for K_o in np.linspace(1, 8, 8):
                for I_app_val in np.linspace(-70, 70, 5):
                    # Fixed example values for time_in_ms and cell_index for simplification
                    voltage = run_simulation(ggap, 0.7 * 0.94, Ikir_coef, cm, 1, K_o, I_app_val, (0, 600), 200, 300, 100)
                    data.append([ggap, Ikir_coef, cm, K_o, I_app_val, voltage])
                
# Convert the list to a DataFrame for easier manipulation
columns = ['ggap', 'Ikir_coef', 'cm', 'K_o', 'I_app_val', 'voltage']
df = pd.DataFrame(data, columns=columns)

# Now, let's set up symbolic regression with PySRRegressor
# Note: Adjust parameters according to your specific needs
model = PySRRegressor(
    niterations=5,  # Number of iterations of the algorithm (increase for real applications)
    binary_operators=["+", "*", "/", "-"],  # Basic arithmetic operations
    unary_operators=[
        "sin", "cos", "exp", "log",  # Some common unary operations; add or remove as needed
        "sqrt", "abs",  # Include more based on your requirements
    ],
    extra_sympy_mappings={},  # Add any additional functions you need to consider
    model_selection="best",  # Strategy to select the best model
    loss="loss(x, y) = (x - y)^2",  # Loss function, here it's MSE
    verbosity=1  # Increase verbosity to see more information during the run
)

# Fit the model to your dataset
X = df[['ggap', 'Ikir_coef', 'cm', 'K_o', 'I_app_val']]
y = df['voltage']
model.fit(X, y)

# Check the best equation found
print("Best equation:", model)

