import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering

from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample  # Correct import for sampling
from SALib.analyze import sobol  # Correct import for analysis
import pandas as pd

# Constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)
membrane_capacitance = 0.94  # Membrane capacitance (pF)
cell_volume = 2e-12  # Cell volume (L)

# Define the model parameters
default_params = {
    'K_out': 4.5, 
    'K_in': 150.0,
    'Ca_out': 2.0,
    'Ca_in': 0.0001,
    'Na_out': 140.0,
    'Na_in': 10.0,
    'Cl_out': 120.0,
    'Cl_in': 30.0,
    'conductance_Kir61': 0.025,
    'conductance_TRPC1': 0.01,
    'conductance_CaCC': 0.001,
    'conductance_CaL': 0.01,
    'conductance_leak': 0.01,
    'calcium_extrusion_rate': 5.0,
    'resting_calcium': 0.0001,
    'calcium_activation_threshold_CaCC': 0.0005,
    'reference_K': 5.4,
    'voltage_slope_Kir61': 6,
    'voltage_shift_Kir61': 15,
    'activation_midpoint_CaL': -40,
    'activation_slope_CaL': 4,
    'inactivation_midpoint_CaL': -45,
    'inactivation_slope_CaL': 5,
    'voltage_shift_CaL': 50,
    'slope_factor_CaL': 20,
    'amplitude_factor_CaL': 0.6,
    'vu': 1540000.0,
    'vm': 2200000.0,
    'aku': 0.303,
    'akmp': 0.14,
    'aru': 1.8,
    'arm': 2.1,
    'initial_dpmca': 1.0,
    'simulation_duration': 50,  # Reduced for faster simulation
    'time_points': 500,  # Reduced for faster simulation
    'initial_voltage': -70,
    'initial_calcium': 0.0001,
    'initial_atp': 4.4
}

# Define the parameter space for sensitivity analysis at the global scope
problem = {
    'num_vars': 10,
    'names': ['conductance_Kir61', 'conductance_TRPC1', 'conductance_CaCC', 'conductance_CaL',
              'conductance_leak', 'calcium_extrusion_rate', 'activation_midpoint_CaL',
              'activation_slope_CaL', 'voltage_slope_Kir61', 'voltage_shift_Kir61'],
    'bounds': [[0.01, 0.05], [0.0, 0.05], [0.0001, 0.01], [0.0001, 0.05],
               [0.001, 0.05], [1.0, 10.0], [-50, -30], [2, 6], [4, 8], [10, 20]]
}

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (1 * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (2 * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (1 * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (-1 * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_currents(V, Ca_i, atp, dpmca, params):
    # Avoid negative or zero values for Ca_i in power calculations
    if Ca_i <= 0:
        Ca_i = 1e-6  # Set to a small positive value to prevent invalid power calculations

    try:
        I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
        I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
        I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
        d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
        f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
        I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
        I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
        u4 = (params['vu'] * (Ca_i**params['aru']) / (Ca_i**params['aru'] + params['aku']**params['aru'])) / (6.6253e5)
        u5 = (params['vm'] * (Ca_i**params['arm']) / (Ca_i**params['arm'] + params['akmp']**params['arm'])) / (6.6253e5)
        cJpmca = (dpmca * u4 + (1 - dpmca) * u5)
        I_PMCA = 0*(cJpmca * 2 * faraday_constant * cell_volume * 1e6) * (1 - 2 / 2)  # Corrects for the electroneutral exchange
    except (ZeroDivisionError, OverflowError, ValueError):
        # Handle cases where calculations fail, set currents to zero or other default
        I_Kir61 = I_TRPC1 = I_CaCC = I_CaL = I_leak = I_PMCA = 0.0
        d_inf = f_inf = u4 = u5 = 0.0

    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, u4, u5

def model(t, y, params):
    V, Ca_i, atp, dpmca = y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, _, _ = calculate_currents(V, Ca_i, atp, dpmca, params)
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA) / membrane_capacitance
    dCa_dt = -I_CaL / (2 * faraday_constant * cell_volume) - params['calcium_extrusion_rate'] * (Ca_i - params['resting_calcium'])
    datp_dt = 0  # ATP concentration remains constant
    w1 = 0.1 * Ca_i
    w2 = 0.01
    taom = 1 / (w1 + w2)
    dpmcainf = w2 / (w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca) / taom
    return [dV_dt, dCa_dt, datp_dt, ddpmca_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca']]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='Radau', args=(params,), dense_output=True)
    return sol

def perform_sensitivity_analysis():
    # Generate parameter samples using SALib.sample.sobol
    param_values = sobol_sample.sample(problem, 256)  # Use `sobol_sample.sample`

    # Prepare storage for model outputs
    Y = []

    # Run the model for each parameter set
    for i, sample in enumerate(param_values):
        params = default_params.copy()
        for j, param in enumerate(problem['names']):
            params[param] = sample[j]

        # Update reversal potentials for each sample
        reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl = calculate_reversal_potentials(
            params['K_out'], params['K_in'], params['Ca_out'], params['Ca_in'], 
            params['Na_out'], params['Na_in'], params['Cl_out'], params['Cl_in']
        )
        params.update({
            'reversal_potential_K': reversal_potential_K,
            'reversal_potential_Ca': reversal_potential_Ca,
            'reversal_potential_Na': reversal_potential_Na,
            'reversal_potential_Cl': reversal_potential_Cl
        })

        # Run simulation and collect output data (e.g., final membrane potential)
        try:
            sol = run_simulation(params)
            final_membrane_potential = sol.y[0][-1]
        except:
            final_membrane_potential = np.nan  # Handle simulation failures

        Y.append(final_membrane_potential)

    # Convert results to numpy array for analysis
    Y = np.array(Y)

    # Remove any NaN values from results before analysis
    valid_indices = ~np.isnan(Y)
    param_values = param_values[valid_indices]
    Y = Y[valid_indices]

    # Perform Sobol sensitivity analysis
    Si = sobol.analyze(problem, Y)
    return Si

def plot_sensitivity_indices(Si, problem):
    # Extract parameter names from the problem definition
    parameter_names = problem['names']
    
    if 'S1' in Si and 'ST' in Si:
        # First Order Sensitivity Plot
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(Si['S1'])), Si['S1'], tick_label=parameter_names)
        plt.title('First Order Sensitivity Indices (S1)')
        plt.xlabel('Parameter')
        plt.ylabel('Sensitivity Index')
        plt.xticks(rotation=45, ha="right")
        plt.grid()
        plt.savefig("s1_sensitivity.png")  # Save the plot as an image
        plt.close()

        # Total Order Sensitivity Plot
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(Si['ST'])), Si['ST'], tick_label=parameter_names)
        plt.title('Total Order Sensitivity Indices (ST)')
        plt.xlabel('Parameter')
        plt.ylabel('Sensitivity Index')
        plt.xticks(rotation=45, ha="right")
        plt.grid()
        plt.savefig("st_sensitivity.png")  # Save the plot as an image
        plt.close()
    else:
        print("Error: Sensitivity indices 'S1' or 'ST' are not present in the analysis results.")



def main():
    # Perform sensitivity analysis
    Si = perform_sensitivity_analysis()

    # Plot the results, passing the problem definition
    plot_sensitivity_indices(Si, problem)

    # Display results in a tabular form
    sensitivity_df = pd.DataFrame({
        'Parameter': problem['names'],
        'First Order (S1)': Si['S1'],
        'Total Order (ST)': Si['ST']
    })
    print(sensitivity_df)

if __name__ == "__main__":
    main()
