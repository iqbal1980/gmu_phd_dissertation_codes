# Consolidated model for pericyte ion channels with extended dynamics

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
ER_volume = cell_volume * 0.2  # ER volume (L)

# Ion valences
valence_K = 1  # Potassium ion valence
valence_Ca = 2  # Calcium ion valence
valence_Na = 1  # Sodium ion valence
valence_Cl = -1  # Chloride ion valence
z = 2  # Valence of calcium ions

# Hardcoded ion concentrations
K_out = 6.26  # Extracellular K+ (mM)
K_in = 140  # Intracellular K+ (mM)
Ca_out = 2.0  # Extracellular Ca2+ (mM)
Ca_in = 0.0001  # Intracellular Ca2+ (mM)
Na_out = 140  # Extracellular Na+ (mM)
Na_in = 15.38  # Intracellular Na+ (mM)
Cl_out = 110  # Extracellular Cl- (mM)
Cl_in = 9.65  # Intracellular Cl- (mM)

# Other parameters
conductance_Kir61 = 0.025
conductance_TRPC1 = 0.001
conductance_CaL = 0.0005
conductance_CaCC = 0.001
conductance_leak = 0.01
conductance_IP3R1 = 0.1
conductance_IP3R2 = 0.05
conductance_RyR = 0.01
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
calcium_activation_threshold_CaCC = 0.0005
k_serca = 0.1
Km_serca = 0.5
leak_rate_er = 0.05
IP3_concentration = 0.1
k_ncx = 0.001

# Buffer parameters
Bscyt = 225.0  # Total cytosolic stationary buffer (μM)
aKscyt = 0.1  # Cytosolic buffer dissociation constant
Bser = 2000.0  # Total ER stationary buffer (μM)
aKser = 1.0  # ER buffer dissociation constant

# Time simulation parameters
simulation_duration = 100  # Simulation duration (ms)
time_points = 1000

# Initial conditions
initial_voltage = -70  # Initial membrane potential (mV)
initial_calcium = Ca_in  # Initial intracellular calcium (mM)
initial_atp = 4.4  # Initial ATP level (unitless)
initial_dpmca = 1.0  # Initial PMCA state (unitless)
Ca_ER_initial = 0.5  # Initial ER calcium (mM)

# Function to calculate reversal potentials
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_buffering_factors(Ca_i, Ca_ER):
    beta_cyt = 1.0 / (1.0 + (Bscyt * aKscyt) / ((aKscyt + Ca_i)**2))
    beta_er = 1.0 / (1.0 + (Bser * aKser) / ((aKser + Ca_ER)**2))
    return beta_cyt, beta_er

# Function to calculate ion currents and fluxes
def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, params):
    I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
    I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    tau_d = params['activation_slope_CaL'] / (1 + np.exp((V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    tau_f = params['inactivation_slope_CaL'] / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
    
    # PMCA current
    u4 = (params['vu'] * (Ca_i**params['aru']) / (Ca_i**params['aru'] + params['aku']**params['aru'])) / (6.6253e5)
    u5 = (params['vm'] * (Ca_i**params['arm']) / (Ca_i**params['arm'] + params['akmp']**params['arm'])) / (6.6253e5)
    cJpmca = (dpmca * u4 + (1 - dpmca) * u5)
    I_PMCA = cJpmca * z * faraday_constant * Vcyto * 1e6

    # SERCA flux
    J_SERCA = params['k_serca'] * (Ca_i**2) / (Ca_i**2 + params['Km_serca']**2)

    # ER Leak flux
    J_ER_leak = params['leak_rate_er'] * (Ca_ER - Ca_i)

    # Na⁺/Ca²⁺ Exchanger (NCX)
    I_NCX = params['k_ncx'] * (params['Na_in']**3 / (params['Na_in']**3 + 87.5**3)) * (params['Ca_out'] / (params['Ca_out'] + 1))

    # IP3R flux
    J_IP3R = params['conductance_IP3R1'] * (Ca_i / (Ca_i + 0.2)) * (params['IP3_concentration'] / (params['IP3_concentration'] + 0.1)) * (Ca_ER - Ca_i)
    J_IP3R2 = params['conductance_IP3R2'] * (Ca_i / (Ca_i + 0.15)) * (params['IP3_concentration'] / (params['IP3_concentration'] + 0.1)) * (Ca_ER - Ca_i)

    # RyR flux
    J_RyR = params['conductance_RyR'] * (Ca_i / (Ca_i + 0.3)) * (Ca_ER - Ca_i)

    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_IP3R2, J_RyR

# ODE model function
def model(t, y, params):
    V, Ca_i, atp, dpmca, Ca_ER = y
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_IP3R2, J_RyR = calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, params)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX) / membrane_capacitance
    
    # Convert membrane currents to fluxes and combine with internal fluxes
    Ca_influx = -I_CaL / (2 * faraday_constant * cell_volume)
    Ca_efflux = -I_PMCA / (2 * faraday_constant * cell_volume) - I_NCX / (2 * faraday_constant * cell_volume)
    
    dCa_dt = beta_cyt * (Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_IP3R2 + J_RyR - J_SERCA)
    
    # ER calcium ODE with volume ratio correction
    dCa_ER_dt = beta_er * (Vcyto/ER_volume) * (J_SERCA - J_ER_leak - J_IP3R - J_IP3R2 - J_RyR)
    
    datp_dt = 0
    
    w1 = 0.1 * Ca_i
    w2 = 0.01
    taom = 1 / (w1 + w2)
    dpmcainf = w2 / (w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca) / taom

    if t % 10 == 0:
        print(f"Time: {t:.2f} ms, V: {V:.2f} mV, Ca_i: {Ca_i:.6f} mM, Ca_ER: {Ca_ER:.6f} mM")
        print(f"I_Kir61: {I_Kir61:.4f}, I_TRPC1: {I_TRPC1:.4f}, I_CaCC: {I_CaCC:.4f}, I_CaL: {I_CaL:.4f}, I_leak: {I_leak:.4f}")
        print(f"J_SERCA: {J_SERCA:.6f}, J_ER_leak: {J_ER_leak:.6f}, J_IP3R: {J_IP3R:.6f}, J_RyR: {J_RyR:.6f}")

    return [dV_dt, dCa_dt, datp_dt, ddpmca_dt, dCa_ER_dt]
    
# Function to run the simulation
def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca'], params['Ca_ER_initial']]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
    return sol

def plot_results_with_two_figures(sol, params):
    # Calculate currents over the entire time range
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_IP3R2, J_RyR = [
        np.zeros_like(sol.t) for _ in range(14)
    ]

    # Evaluate currents at each time point
    for i, (V, Ca_i, atp, dpmca, Ca_ER) in enumerate(zip(sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])):
        (
            I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i], d_inf[i], f_inf[i], I_PMCA[i],
            J_SERCA[i], J_ER_leak[i], I_NCX[i], J_IP3R[i], J_IP3R2[i], J_RyR[i]
        ) = calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, params)

    # Figure 1: Membrane Potential, Calcium Concentrations, and ATP
    fig1, axs1 = plt.subplots(4, 1, figsize=(12, 20))
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

    # ER Calcium
    axs1[2].plot(sol.t, sol.y[4], color='red', linewidth=2)
    axs1[2].set_title('ER Calcium Concentration', fontsize=14, fontweight='bold')
    axs1[2].set_ylabel('ER Ca2+ (mM)', fontsize=12, fontweight='bold')
    axs1[2].grid(True)

    # Intracellular Calcium (Log Scale)
    axs1[3].plot(sol.t, sol.y[1], color='green', linewidth=2)
    axs1[3].set_yscale('log')
    axs1[3].set_title('Intracellular Calcium Concentration (Log Scale)', fontsize=14, fontweight='bold')
    axs1[3].set_ylabel('Intracellular\nCa2+ (mM)', fontsize=12, fontweight='bold')
    axs1[3].grid(True)

    fig1.tight_layout()
    plt.show()

    # Figure 2: Currents, Fluxes, Gating Variables, and dPMCA
    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 20))
    fig2.suptitle('Cell Dynamics - Part 2', fontsize=16, fontweight='bold')

    # Membrane Currents
    axs2[0].plot(sol.t, I_Kir61, label='I_Kir61', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_TRPC1, label='I_TRPC1', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_CaCC, label='I_CaCC', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_CaL, label='I_CaL', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_leak, label='I_leak', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_PMCA, label='I_PMCA', alpha=0.8, linewidth=2)
    axs2[0].plot(sol.t, I_NCX, label='I_NCX', alpha=0.8, linewidth=2, color='magenta')
    axs2[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[0].set_title('Membrane Currents Over Time', fontsize=14, fontweight='bold')
    axs2[0].set_ylabel('Current (pA)', fontsize=12, fontweight='bold')
    axs2[0].grid(True)

    # Calcium Fluxes
    axs2[1].plot(sol.t, J_SERCA, label='J_SERCA', alpha=0.8, linewidth=2, color='purple')
    axs2[1].plot(sol.t, J_ER_leak, label='J_ER_leak', alpha=0.8, linewidth=2, color='cyan')
    axs2[1].plot(sol.t, J_IP3R, label='J_IP3R', alpha=0.8, linewidth=2, color='pink')
    axs2[1].plot(sol.t, J_RyR, label='J_RyR', alpha=0.8, linewidth=2, color='brown')
    axs2[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[1].set_title('Calcium Fluxes Over Time', fontsize=14, fontweight='bold')
    axs2[1].set_ylabel('Flux (mM/s)', fontsize=12, fontweight='bold')
    axs2[1].grid(True)

    # Gating Variables
    axs2[2].plot(sol.t, d_inf, label='d_inf', linestyle='--', color='purple', linewidth=2)
    axs2[2].plot(sol.t, f_inf, label='f_inf', linestyle='--', color='brown', linewidth=2)
    axs2[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[2].set_title('Gating Variables Over Time', fontsize=14, fontweight='bold')
    axs2[2].set_ylabel('Gating Variables', fontsize=12, fontweight='bold')
    axs2[2].grid(True)

    # dPMCA
    axs2[3].plot(sol.t, sol.y[3], label='dPMCA', color='cyan', linewidth=2)
    axs2[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs2[3].set_title('dPMCA Over Time', fontsize=14, fontweight='bold')
    axs2[3].set_ylabel('dPMCA\nConcentration', fontsize=12, fontweight='bold')
    axs2[3].grid(True)

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
    'conductance_RyR': conductance_RyR,
    'k_serca': k_serca,
    'Km_serca': Km_serca,
    'leak_rate_er': leak_rate_er,
    'IP3_concentration': IP3_concentration,
    'k_ncx': k_ncx,
    'calcium_extrusion_rate': calcium_extrusion_rate,
    'resting_calcium': resting_calcium,
    'calcium_activation_threshold_CaCC': calcium_activation_threshold_CaCC,
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
    'initial_dpmca': initial_dpmca,
    'simulation_duration': simulation_duration,
    'time_points': time_points,
    'initial_voltage': initial_voltage,
    'initial_calcium': initial_calcium,
    'initial_atp': initial_atp,
    'Ca_ER_initial': Ca_ER_initial
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