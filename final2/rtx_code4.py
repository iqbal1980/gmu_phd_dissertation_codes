import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from multiprocessing import Pool
import time

# Constants
F = 96485  # Faraday constant (C/mol)
R = 8.314  # Gas constant (J/(mol*K))
T = 310    # Temperature (K)

# Cell parameters
C_m = 15e-12  # Membrane capacitance (F)
vol_cell = 1e-15  # Cell volume (L)

# Ion concentrations (mM)
K_o, K_i = 5.4, 140
Ca_o, Ca_i = 2.0, 0.0001
Na_o, Na_i = 140, 10
Cl_o, Cl_i = 120, 30

# Conductances (nS)
g_Kir61 = 0.1
g_TRPC1 = 0.005
g_CaL = 0.01
g_CaCC = 0.05
g_leak = 0.01

# ATP-related parameters
ATP_i = 5  # Initial intracellular ATP (mM)
K_atp = 0.5  # Half-maximal ATP concentration for Kir6.1 (mM)

# Calcium handling parameters
v_serca = 0.5  # SERCA pump rate (μM/ms)
k_serca = 0.25  # SERCA pump affinity (μM)
k_leak = 0.0005  # ER leak rate (ms^-1)
IP3 = 0.1  # IP3 concentration (μM)
k_ip3r = 0.2  # IP3R affinity (μM)
Ca_er = 400  # ER calcium concentration (μM)

# Normalization factors
V_norm = 100  # mV
Ca_norm = 1  # mM
K_norm = 140  # mM
Na_norm = 140  # mM
Cl_norm = 120  # mM
ATP_norm = 10  # mM

def nernst_potential(z, c_out, c_in):
    return (R * T / (z * F)) * np.log(max(c_out, 1e-10) / max(c_in, 1e-10)) * 1000  # mV

def model(t, y, params):
    V, Ca_i, K_i, Na_i, Cl_i, ATP = y
    g_Kir61, g_TRPC1, g_CaL, g_CaCC, k_ip3r, ATP_production = params
    
    # Denormalize variables
    V = V * V_norm
    Ca_i = Ca_i * Ca_norm
    K_i = K_i * K_norm
    Na_i = Na_i * Na_norm
    Cl_i = Cl_i * Cl_norm
    ATP = ATP * ATP_norm
    
    # Enforce bounds on variables
    V = np.clip(V, -100, 50)
    Ca_i = np.clip(Ca_i, 1e-7, 10)
    K_i = np.clip(K_i, 1, 160)
    Na_i = np.clip(Na_i, 1, 20)
    Cl_i = np.clip(Cl_i, 1, 40)
    ATP = np.clip(ATP, 0.1, 10)
    
    E_K = nernst_potential(1, K_o, K_i)
    E_Ca = nernst_potential(2, Ca_o, Ca_i)
    E_Na = nernst_potential(1, Na_o, Na_i)
    E_Cl = nernst_potential(-1, Cl_o, Cl_i)
    
    I_Kir61 = g_Kir61 * (ATP / (ATP + K_atp)) * (V - E_K)
    I_TRPC1 = g_TRPC1 * (V - E_Ca)
    I_CaL = g_CaL * (V - E_Ca)
    I_CaCC = g_CaCC * (Ca_i / (Ca_i + 0.5)) * (V - E_Cl)
    I_leak = g_leak * (V - E_K)
    
    dV_dt = -(I_Kir61 + I_TRPC1 + I_CaL + I_CaCC + I_leak) / C_m
    
    J_serca = v_serca * (Ca_i**2 / (Ca_i**2 + k_serca**2))
    J_leak = k_leak * (Ca_er - Ca_i)
    J_ip3r = k_ip3r * (IP3 / (IP3 + k_ip3r)) * (Ca_i / (Ca_i + 0.3)) * (Ca_er - Ca_i)
    
    dCa_dt = (-I_CaL / (2 * F * vol_cell) + J_leak + J_ip3r - J_serca) * 1e3  # Convert to μM/ms
    dK_dt = -I_Kir61 / (F * vol_cell)
    dNa_dt = 0  # Simplified, assuming Na+ is constant
    dCl_dt = I_CaCC / (F * vol_cell)
    dATP_dt = ATP_production - 0.1 * ATP
    
    # Normalize rates of change
    dV_dt /= V_norm
    dCa_dt /= Ca_norm
    dK_dt /= K_norm
    dNa_dt /= Na_norm
    dCl_dt /= Cl_norm
    dATP_dt /= ATP_norm
    
    return [dV_dt, dCa_dt, dK_dt, dNa_dt, dCl_dt, dATP_dt]

# Define the parameter ranges for Sobol analysis
problem = {
    'num_vars': 6,
    'names': ['g_Kir61', 'g_TRPC1', 'g_CaL', 'g_CaCC', 'k_ip3r', 'ATP_production'],
    'bounds': [[0.05, 0.2], [0.001, 0.01], [0.005, 0.02], [0.01, 0.1], [0.1, 0.5], [0.1, 1.0]]
}

def run_model(params):
    y0 = [-70/V_norm, Ca_i/Ca_norm, K_i/K_norm, Na_i/Na_norm, Cl_i/Cl_norm, ATP_i/ATP_norm]
    t_end = 1000
    
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', with_jacobian=True, atol=1e-8, rtol=1e-6)
    solver.set_f_params(params)
    solver.set_initial_value(y0, 0)
    
    dt = 1.0  # Increased time step for faster computation
    t = []
    sol = []
    
    try:
        while solver.successful() and solver.t < t_end:
            solver.integrate(solver.t + dt)
            t.append(solver.t)
            sol.append(solver.y)
        
        sol = np.array(sol)
        return np.array([
            sol[-1, 0] * V_norm,  # Final membrane potential
            np.mean(sol[:, 1]) * Ca_norm,  # Mean intracellular calcium
            np.max(sol[:, 1]) * Ca_norm,  # Peak intracellular calcium
            sol[-1, 5] * ATP_norm  # Final ATP concentration
        ])
    except Exception as e:
        print(f"Error in simulation: {e}")
        return np.array([np.nan, np.nan, np.nan, np.nan])

def parallel_run_model(param_values):
    print(f"Starting parallel simulations for {len(param_values)} parameter sets...")
    start_time = time.time()
    with Pool() as pool:
        results = pool.map(run_model, param_values)
    end_time = time.time()
    print(f"Parallel simulations completed in {end_time - start_time:.2f} seconds")
    return np.array(results)

def main():
    start_time = time.time()
    print("Generating Sobol samples...")
    param_values = sobol_sample.sample(problem, 1024)  # Using 8 samples as per your test
    print(f"Generated {len(param_values)} parameter sets")

    print("Running model simulations...")
    Y = parallel_run_model(param_values)

    print("Processing results...")
    valid_indices = ~np.isnan(Y).any(axis=1)
    Y_valid = Y[valid_indices]
    param_values_valid = param_values[valid_indices]
    print(f"Valid simulations: {len(Y_valid)} out of {len(Y)}")

    if len(Y_valid) >= 64:  # Minimum number of samples for Sobol analysis
        print("Performing Sobol analysis...")
        try:
            Si = sobol.analyze(problem, Y_valid)
            print("\nSobol Sensitivity Indices:")
            for i, name in enumerate(problem['names']):
                print(f"{name}:")
                print(f"  S1 (First-order): {Si['S1'][i]:.3f}")
                print(f"  ST (Total-order): {Si['ST'][i]:.3f}")
        except Exception as e:
            print(f"Sobol analysis failed: {e}")
            print("Proceeding without Sobol indices.")
            Si = None
    else:
        print("Insufficient samples for Sobol analysis. Skipping.")
        Si = None

    print("\nRunning simulation with mean parameters...")
    mean_params = np.mean(param_values_valid, axis=0)
    run_and_plot_simulation(mean_params)

    if Si is not None:
        print("\nAdjusting parameters...")
        target_voltage = -70  # mV
        target_calcium = 0.1  # μM
        adjusted_params = adjust_parameters(target_voltage, target_calcium, param_values_valid, Si)

        print("\nRunning simulation with adjusted parameters...")
        run_and_plot_simulation(adjusted_params)

        print("\nAdjusted parameters:")
        for name, value in zip(problem['names'], adjusted_params):
            print(f"{name}: {value:.6f}")
    else:
        print("\nSkipping parameter adjustment due to insufficient data for Sobol analysis.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

def run_and_plot_simulation(params):
    y0 = [-70/V_norm, Ca_i/Ca_norm, K_i/K_norm, Na_i/Na_norm, Cl_i/Cl_norm, ATP_i/ATP_norm]
    t_end = 1000
    
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', with_jacobian=True, atol=1e-8, rtol=1e-6)
    solver.set_f_params(params)
    solver.set_initial_value(y0, 0)
    
    dt = 1.0
    t = []
    sol = []
    
    while solver.successful() and solver.t < t_end:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)
    
    sol = np.array(sol)
    t = np.array(t)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Pericyte Model Simulation', fontsize=16)
    
    ax1.plot(t, sol[:, 0] * V_norm)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Membrane Potential Over Time')
    
    ax2.plot(t, sol[:, 1] * Ca_norm, label='Ca2+')
    ax2.plot(t, sol[:, 2] * K_norm, label='K+')
    ax2.plot(t, sol[:, 3] * Na_norm, label='Na+')
    ax2.plot(t, sol[:, 4] * Cl_norm, label='Cl-')
    ax2.set_ylabel('Ion Concentration (mM)')
    ax2.set_title('Intracellular Ion Concentrations')
    ax2.legend()
    ax2.set_yscale('log')
    
    ax3.plot(t, sol[:, 5] * ATP_norm)
    ax3.set_ylabel('ATP Concentration (mM)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Intracellular ATP Concentration')
    
    plt.tight_layout()
    plt.show()

def adjust_parameters(target_voltage, target_calcium, param_values_valid, Si):
    params = np.mean(param_values_valid, axis=0)
    
    for i in range(5):  # Perform 5 iterations of adjustment
        print(f"Adjustment iteration {i+1}")
        result = run_model(params)
        if np.isnan(result).any():
            print("Parameter adjustment failed due to numerical instability")
            return params
        
        voltage_error = result[0] - target_voltage
        calcium_error = result[1] - target_calcium
        
        print(f"Current voltage: {result[0]:.2f} mV, Target: {target_voltage} mV, Error: {voltage_error:.2f} mV")
        print(f"Current calcium: {result[1]:.6f} mM, Target: {target_calcium} mM, Error: {calcium_error:.6f} mM")
        
        if Si is not None:
            # Adjust parameters based on Sobol indices and errors
            params[0] += -np.sign(voltage_error) * Si['ST'][0] * 0.01  # g_Kir61
            params[1] += -np.sign(calcium_error) * Si['ST'][1] * 0.001  # g_TRPC1
            params[2] += -np.sign(calcium_error) * Si['ST'][2] * 0.001  # g_CaL
            params[3] += -np.sign(calcium_error) * Si['ST'][3] * 0.01  # g_CaCC
        else:
            # Simple adjustment without Sobol indices
            params[0] += -np.sign(voltage_error) * 0.01  # g_Kir61
            params[1] += -np.sign(calcium_error) * 0.001  # g_TRPC1
            params[2] += -np.sign(calcium_error) * 0.001  # g_CaL
            params[3] += -np.sign(calcium_error) * 0.01  # g_CaCC
        
        # Ensure parameters stay within bounds
        params = np.clip(params, problem['bounds'][:,0], problem['bounds'][:,1])
        
        print("Adjusted parameters:")
        for name, value in zip(problem['names'], params):
            print(f"{name}: {value:.6f}")
        print()
    
    return params

if __name__ == "__main__":
    main()