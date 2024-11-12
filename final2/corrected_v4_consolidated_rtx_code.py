# Consolidated model for pericyte ion channels based on rtx_code2.py with improvements

# Nernst potential function (from rtx_code4)
def nernst_potential(z, c_out, c_in):
    return (R * T / (z * F)) * np.log(max(c_out, 1e-10) / max(c_in, 1e-10)) * 1000  # mV

# Base dynamics from rtx_code2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)

# Cell parameters
membrane_capacitance = 0.94  # Membrane capacitance (pF)
cell_volume = 2e-12  # Cell volume (L)
Vcyto = cell_volume  # Cytoplasmic volume (L)

# Ion valences
valence_K = 1  # Potassium ion valence
valence_Ca = 2  # Calcium ion valence
valence_Na = 1  # Sodium ion valence
valence_Cl = -1  # Chloride ion valence
z = 2  # Valence of calcium ions

# Hardcoded ion concentrations based on earlier findings
K_out = 6.26  # Extracellular K+ (mM)
K_in = 140  # Intracellular K+ (mM)
Ca_out = 2.0  # Extracellular Ca2+ (mM)
Ca_in = 0.0001  # Intracellular Ca2+ (mM)
Na_out = 140  # Extracellular Na+ (mM)
Na_in = 15.38  # Intracellular Na+ (mM)
Cl_out = 110  # Extracellular Cl- (mM)
Cl_in = 9.65  # Intracellular Cl- (mM)

# Other hardcoded parameters
conductance_Kir61 = 0.025  # Kir6.1 conductance (nS)
conductance_TRPC1 = 0.001  # TRPC1 conductance (nS)
conductance_CaL = 0.0005  # L-type calcium channel conductance (nS)
conductance_CaCC = 0.001  # Calcium-activated chloride channel conductance (nS)
conductance_leak = 0.01  # Leak conductance (nS)
conductance_IP3R1 = 0.1  # IP3R1 conductance (nS)
conductance_IP3R2 = 0.05  # IP3R2 conductance (nS)
calcium_extrusion_rate = 100.0  # Ca2+ extrusion rate (ms^-1)
resting_calcium = 0.001  # Resting calcium concentration (mM)
calcium_activation_threshold_CaCC = 0.0005  # CaCC activation threshold (mM)

# Time simulation parameters
simulation_duration = 100  # Simulation duration (ms)
time_points = 1000  # Number of time points

# Initial conditions
initial_voltage = -70  # Initial membrane potential (mV)
initial_calcium = Ca_in  # Initial intracellular calcium (mM)
initial_atp = 4.4  # Initial ATP level (unitless)
initial_dpmca = 1.0  # Initial PMCA state (unitless)

# Function to calculate reversal potentials
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

# Function to calculate ion currents
def calculate_currents(V, Ca_i, atp, dpmca, params):
    I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
    I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])  # Assuming potassium leak
    u4 = (params['vu'] * (Ca_i**params['aru']) / (Ca_i**params['aru'] + params['aku']**params['aru'])) / (6.6253e5)
    u5 = (params['vm'] * (Ca_i**params['arm']) / (Ca_i**params['arm'] + params['akmp']**params['arm'])) / (6.6253e5)
    cJpmca = (dpmca * u4 + (1 - dpmca) * u5)
    I_PMCA = cJpmca * z * faraday_constant * Vcyto * 1e6  # convert flux (μM/s) to current (pA/s)

    # Calculate IP3R currents but do not affect the main dynamics
    I_IP3R1 = params['conductance_IP3R1'] * (V - params['reversal_potential_Ca'])
    I_IP3R2 = params['conductance_IP3R2'] * (V - params['reversal_potential_Ca'])

    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, I_IP3R1, I_IP3R2

# ODE model function
def model(t, y, params):
    V, Ca_i, atp, dpmca = y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, I_IP3R1, I_IP3R2 = calculate_currents(V, Ca_i, atp, dpmca, params)
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA) / membrane_capacitance
    Ca_influx = -I_CaL / (2 * faraday_constant * cell_volume)
    Ca_efflux = params['calcium_extrusion_rate'] * (Ca_i - params['resting_calcium'])
    dCa_dt = Ca_influx - Ca_efflux
    datp_dt = 0  # ATP concentration remains constant
    w1 = 0.1 * Ca_i
    w2 = 0.01
    taom = 1 / (w1 + w2)
    dpmcainf = w2 / (w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca) / taom

    if t % 10 == 0:  # Print every 10 ms
        print(f"Time: {t:.2f} ms, V: {V:.2f} mV, Ca_i: {Ca_i:.6f} mM")
        print(f"I_Kir61: {I_Kir61:.4f}, I_TRPC1: {I_TRPC1:.4f}, I_CaCC: {I_CaCC:.4f}, I_CaL: {I_CaL:.4f}, I_leak: {I_leak:.4f}, I_PMCA: {I_PMCA:.4f}, I_IP3R1: {I_IP3R1:.4f}, I_IP3R2: {I_IP3R2:.4f}")
        print(f"dCa_dt: {dCa_dt:.6f}, Ca influx: {Ca_influx:.6f}, Ca efflux: {Ca_efflux:.6f}")

    return [dV_dt, dCa_dt, datp_dt, ddpmca_dt]

# Function to run the simulation
def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca']]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
    return sol

def plot_results_with_two_figures(sol, params):
    # Figure 1: Membrane Potential, Calcium Concentrations, and ATP
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 15))
    fig1.suptitle('Cell Dynamics - Part 1', fontsize=16, fontweight='bold')

    # Membrane Potential
    axs1[0].plot(sol.t, sol.y[0], color='blue', linewidth=2)
    axs1[0].set_title('Membrane Potential Over Time', fontsize=14, fontweight='bold')
    axs1[0].set_ylabel('Membrane\nPotential (mV)', fontsize=12, fontweight='bold')
    axs1[0].grid(True)

    # Intracellular Calcium (Regular Scale)
    axs1[1].plot(sol.t, sol.y[1], color='orange', linewidth=2)
    axs1[1].set_title('Intracellular Calcium Concentration (Regular Scale)', fontsize=14, fontweight='bold')
    axs1[1].set_ylabel('Intracellular\nCa2+ (mM)', fontsize=12, fontweight='bold')
    axs1[1].grid(True)

    # Intracellular Calcium (Log Scale)
    axs1[2].plot(sol.t, sol.y[1], color='green', linewidth=2)
    axs1[2].set_yscale('log')
    axs1[2].set_title('Intracellular Calcium Concentration (Log Scale)', fontsize=14, fontweight='bold')
    axs1[2].set_ylabel('Intracellular\nCa2+ (mM)', fontsize=12, fontweight='bold')
    axs1[2].grid(True)

    fig1.tight_layout()
    plt.show()

    # Figure 2: Currents, Gating Variables, and dPMCA
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 15))
    fig2.suptitle('Cell Dynamics - Part 2', fontsize=16, fontweight='bold')

    # Currents
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, I_IP3R1, I_IP3R2 = calculate_currents(sol.y[0], sol.y[1], sol.y[2], sol.y[3], params)
    axs2[0].plot(sol.t, I_Kir61, label='I_Kir61', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_TRPC1, label='I_TRPC1', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_CaCC, label='I_CaCC', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_CaL, label='I_CaL', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_leak, label='I_leak', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_PMCA, label='I_PMCA', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_IP3R1, label='I_IP3R1', alpha=0.8, linewidth=2, color='magenta')
    axs2[0].plot(sol.t, I_IP3R2, label='I_IP3R2', alpha=0.8, linewidth=2, color='pink')
    axs2[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[0].set_title('Currents Over Time', fontsize=14, fontweight='bold')
    axs2[0].set_ylabel('Current (pA)', fontsize=12, fontweight='bold')
    axs2[0].grid(True)

    # Gating Variables
    axs2[1].plot(sol.t, d_inf, label='d_inf', linestyle='--', color='purple', linewidth=2)
    axs2[1].plot(sol.t, f_inf, label='f_inf', linestyle='--', color='brown', linewidth=2)
    axs2[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[1].set_title('Gating Variables Over Time', fontsize=14, fontweight='bold')
    axs2[1].set_ylabel('Gating Variables', fontsize=12, fontweight='bold')
    axs2[1].grid(True)

    # dPMCA
    axs2[2].plot(sol.t, sol.y[3], label='dPMCA', color='cyan', linewidth=2)
    axs2[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[2].set_title('dPMCA Over Time', fontsize=14, fontweight='bold')
    axs2[2].set_ylabel('dPMCA\nConcentration', fontsize=12, fontweight='bold')
    axs2[2].grid(True)

    fig2.tight_layout()
    plt.show()

# Parameters dictionary
params = {
    'K_out': K_out, 'K_in': K_in, 'Ca_out': Ca_out, 'Ca_in': Ca_in,
    'Na_out': Na_out, 'Na_in': Na_in, 'Cl_out': Cl_out, 'Cl_in': Cl_in,
    'conductance_Kir61': conductance_Kir61, 'conductance_TRPC1': conductance_TRPC1,
    'conductance_CaCC': conductance_CaCC, 'conductance_CaL': conductance_CaL,
    'conductance_leak': conductance_leak,
    'conductance_IP3R1': conductance_IP3R1,
    'conductance_IP3R2': conductance_IP3R2,
    'calcium_extrusion_rate': calcium_extrusion_rate,
    'resting_calcium': resting_calcium,
    'calcium_activation_threshold_CaCC': calcium_activation_threshold_CaCC,
    'reference_K': 5.4,  # Reference K+ (mM)
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
    'initial_dpmca': initial_dpmca,
    'simulation_duration': simulation_duration,
    'time_points': time_points,
    'initial_voltage': initial_voltage,
    'initial_calcium': initial_calcium,
    'initial_atp': initial_atp
}

# Calculate reversal potentials
reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl = calculate_reversal_potentials(
    params['K_out'], params['K_in'], params['Ca_out'], params['Ca_in'],
    params['Na_out'], params['Na_in'], params['Cl_out'], params['Cl_in'])

params.update({
    'reversal_potential_K': reversal_potential_K,
    'reversal_potential_Ca': reversal_potential_Ca,
    'reversal_potential_Na': reversal_potential_Na,
    'reversal_potential_Cl': reversal_potential_Cl
})

# Run simulation and plot results
sol = run_simulation(params)
plot_results_with_two_figures(sol, params)
