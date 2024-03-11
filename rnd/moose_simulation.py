import moose
import numpy as np
from scipy.optimize import curve_fit

# Constants for safe_log and safe_exponential
MIN_VALUE = -80  # Minimum physiologically reasonable value for Vm
MAX_VALUE = 40   # Maximum physiologically reasonable value for Vm

# Function for safe_log
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return moose.functions.log(x)

# Function for exponential_function
def exponential_function(x, a):
    return moose.functions.exp(a * x)

# Function for exponential_decay_function
def exponential_decay_function(x, A, B):
    return A * moose.functions.exp(B * x)

# Simulate process function
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * -33
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = I_app[99]
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o / 150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm += 3 * (Vm[kk + 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng - 1:
                new_Vm += eki1 * (Vm[kk - 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm += eki1 * 0.6 * (Vm[kk + 1] + Vm[kk - 1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm += eki1 * (Vm[kk + 1] + Vm[kk - 1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VALUE), MIN_VALUE)

    return Vm

# Objective function
def objective(params):
    print("Starting objective function...")
    
    # Clip each parameter to lie within its bounds.
    for i in range(len(params)):
        low, high = param_bounds[i]
        params[i] = np.clip(params[i], low, high)
    
    ggap, Ikir_coef, cm, K_o = params
    
    dx = 1
    Ibg_init = 0.7 * 0.94
    
    # Run the simulation with the provided parameters
    Vm = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, np.zeros(200))

    cellLength = 60  # in Microns
    D = np.abs(Vm[101:135] - Vm[101]) / np.abs(Vm[101])
    
    distance_m = cellLength * np.arange(0, 34)

    A_initial = D[0]
    B_initial = np.log(D[1] / D[0]) / (distance_m[1] - distance_m[0])
    
    # Check for NaNs in D and distance_m before curve_fit
    if np.any(np.isnan(D)) or np.any(np.isnan(distance_m)):
        print("NaN detected in D or distance_m")
        print(f"D: {D}, distance_m: {distance_m}")
        return None

    try:
        # Fit the experimental data to the exponential decay function
        popt, _ = curve_fit(exponential_decay_function, distance_m, D, p0=[A_initial, B_initial])
        A, B = popt

        # Generate simulated exponential decay with fitted parameters
        simulated_decay = exponential_decay_function(distance_m, A, B)

        # Reference exponential decay function
        reference_decay = 1 * np.exp(-0.003 * distance_m)

        # Calculate the loss as sum of squared differences
        loss = np.sum((simulated_decay - reference_decay) ** 2)
        
        if loss <= 0.3:
            with open('good_genes.txt', 'a') as file:
                file.write(f'Params: {params}, Loss: {loss}\n')
                
    except RuntimeError as e:
        print("Error in curve fitting:", e)
        return None

    print("Completed objective function with loss:", loss)
    return (loss,)

# Parameter bounds
param_bounds = [
    (0.1, 35),  # ggap
    (0.90, 0.96),  # Ikir_coef
    (8, 11),     # cm
    (1, 8)       # K_o
]

def main():
    # Create a MOOSE compartment
    compartment = moose.Compartment('compartment')
    compartment.Em = -70e-3  # Resting membrane potential
    compartment.Rm = 1e12    # Membrane resistance
    compartment.Cm = 1e-12   # Membrane capacitance
    
    # Create a pulse generator
    pulse = moose.PulseGen('pulse')
    pulse.delay[0] = 100.0
    pulse.width[0] = 300.0
    pulse.level[0] = 70.0
    pulse.delay[1] = 1e9
    
    # Connect the pulse generator to the compartment
    moose.connect(pulse, 'output', compartment, 'injectMsg')
    
    # Run the optimization
    params = [np.random.uniform(low, high) for low, high in param_bounds]
    loss = objective(params)
    print("Optimization completed with loss:", loss)

if __name__ == "__main__":
    main()