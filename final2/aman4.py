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
valence_K = 1  
valence_Ca = 2  
valence_Na = 1  
valence_Cl = -1  
z = 2  

# Hardcoded ion concentrations
K_out = 6.26  # Extracellular K+ (mM)
K_in = 140  # Intracellular K+ (mM)
Ca_out = 2.0  # Extracellular Ca2+ (mM)
Ca_in = 0.0001  # Intracellular Ca2+ (mM)
Na_out = 140  # Extracellular Na+ (mM)
Na_in = 15.38  # Intracellular Na+ (mM)
Cl_out = 110  # Extracellular Cl- (mM)
Cl_in = 9.65  # Intracellular Cl- (mM)

# De Young-Keizer IP3R Parameters from T-cell model
a1 = 400.0    # IP3 binding constant (μM^-1 s^-1)
a2 = 0.2      # Ca2+ inhibitory binding constant (μM^-1 s^-1)
a3 = 400.0    # IP3 binding constant (μM^-1 s^-1)
a4 = 0.2      # Ca2+ inhibitory binding constant (μM^-1 s^-1)
a5 = 20.0     # Ca2+ activation binding constant (μM^-1 s^-1)
d1 = 0.13     # IP3 dissociation constant (μM)
d2 = 1.049    # Ca2+ inhibitory dissociation constant (μM)
d3 = 0.9434   # IP3 dissociation constant (μM)
d4 = 0.1445   # Ca2+ inhibitory dissociation constant (μM)
d5 = 82.34e-3 # Ca2+ activation dissociation constant (μM)
v1 = 90.0     # Maximum IP3R flux rate

# Calculate binding products for IP3R
b1 = a1 * d1
b2 = a2 * d2
b3 = a3 * d3
b4 = a4 * d4
b5 = a5 * d5

# PMCA parameters from T-cell model
vu = 1.0 * 1540000.0  # Unmodulated PMCA velocity (ions/s)
vm = 1.0 * 2200000.0  # Modulated PMCA velocity (ions/s)
aku = 0.303   # Unmodulated Ca2+ binding constant
akmp = 0.14   # Modulated Ca2+ binding constant
aru = 1.8     # Unmodulated Hill coefficient
arm = 2.1     # Modulated Hill coefficient
p1 = 0.1      # PMCA activation rate constant (μM^-1)
p2 = 0.01     # PMCA deactivation rate constant

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
aKscyt = 0.1   # Cytosolic buffer dissociation constant
Bser = 2000.0  # Total ER stationary buffer (μM)
aKser = 1.0    # ER buffer dissociation constant

# Time simulation parameters
simulation_duration = 100  # Simulation duration (ms)
time_points = 1000

# Initial conditions
initial_voltage = -70  
initial_calcium = Ca_in  
initial_atp = 4.4  
initial_dpmca = 1.0  
Ca_ER_initial = 0.5  

# Initial conditions for IP3R states
x000_initial = 0.27    # Unbound state
x100_initial = 0.039   # IP3 bound
x010_initial = 0.29    # Ca2+ activation site bound
x001_initial = 0.17    # Ca2+ inhibition site bound
x110_initial = 0.042   # IP3 and Ca2+ activation bound
x101_initial = 0.0033  # IP3 and Ca2+ inhibition bound
x011_initial = 0.18    # Both Ca2+ sites bound
x111_initial = 0.0035  # All sites bound

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

def calculate_ip3r_states(Ca_i, x000, x100, x010, x001, x110, x101, x011, x111):
   """
   Calculate IP3R state transitions based on De Young-Keizer model
   Returns the derivatives for each state
   """
   # IP3R state transitions
   f1 = b5*x010 - a5*Ca_i*x000
   f2 = b1*x100 - a1*IP3_concentration*x000
   f3 = b4*x001 - a4*Ca_i*x000
   f4 = b5*x110 - a5*Ca_i*x100
   f5 = b2*x101 - a2*Ca_i*x100
   f6 = b1*x110 - a1*IP3_concentration*x010
   f7 = b4*x011 - a4*Ca_i*x010
   f8 = b5*x011 - a5*Ca_i*x001
   f9 = b3*x101 - a3*IP3_concentration*x001
   f10 = b2*x111 - a2*Ca_i*x110
   f11 = b5*x111 - a5*Ca_i*x101
   f12 = b3*x111 - a3*IP3_concentration*x011

   # State derivatives
   dx000_dt = f1 + f2 + f3
   dx100_dt = f4 + f5 - f2
   dx010_dt = -f1 + f6 + f7
   dx001_dt = f8 - f3 + f9
   dx110_dt = -f4 - f6 + f10
   dx101_dt = f11 - f9 - f5
   dx011_dt = -f8 - f7 + f12
   dx111_dt = -f11 - f12 - f10

   return [dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt]

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
   """
   Calculate IP3R flux using open probability from De Young-Keizer model
   """
   return v1 * (x110**3) * (Ca_ER - Ca_i)

def calculate_pmca_flux(Ca_i, dpmca):
   """
   Calculate PMCA flux using T-cell model formulation with modulated and unmodulated states
   """
   # Unmodulated state flux
   u4 = (vu * (Ca_i**aru)) / (Ca_i**aru + aku**aru)
   
   # Modulated state flux
   u5 = (vm * (Ca_i**arm)) / (Ca_i**arm + akmp**arm)
   
   # Combined flux based on PMCA state
   cJpmca = (dpmca * u4 + (1 - dpmca) * u5) / (6.6253e5)
   
   return cJpmca, u4, u5
   
   
def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, params):
   I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
   I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
   I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
   d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
   tau_d = params['activation_slope_CaL'] / (1 + np.exp((V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
   f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
   tau_f = params['inactivation_slope_CaL'] / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL']))
   I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
   I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
   
   # PMCA current using T-cell model formulation
   cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
   I_PMCA = cJpmca * z * faraday_constant * Vcyto * 1e6

   # Other fluxes
   J_SERCA = params['k_serca'] * (Ca_i**2) / (Ca_i**2 + params['Km_serca']**2)
   J_ER_leak = params['leak_rate_er'] * (Ca_ER - Ca_i)
   I_NCX = params['k_ncx'] * (params['Na_in']**3 / (params['Na_in']**3 + 87.5**3)) * (params['Ca_out'] / (params['Ca_out'] + 1))
   J_IP3R = calculate_ip3r_flux(x110, Ca_ER, Ca_i)
   J_RyR = params['conductance_RyR'] * (Ca_i / (Ca_i + 0.3)) * (Ca_ER - Ca_i)

   return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, 0, J_RyR
   
# ODE model function
def model(t, y, params):
   # Unpack all state variables including IP3R states
   V, Ca_i, atp, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111 = y
   
   beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
   
   # Calculate all currents and fluxes
   I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, _, J_RyR = calculate_currents(
       V, Ca_i, atp, dpmca, Ca_ER, x110, params)
   
   # Membrane potential ODE
   dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX) / membrane_capacitance
   
   # Convert membrane currents to fluxes and combine with internal fluxes
   Ca_influx = -I_CaL / (2 * faraday_constant * cell_volume)
   Ca_efflux = -I_PMCA / (2 * faraday_constant * cell_volume) - I_NCX / (2 * faraday_constant * cell_volume)
   
   # Calcium ODEs with new IP3R flux
   dCa_dt = beta_cyt * (Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
   dCa_ER_dt = beta_er * (Vcyto/ER_volume) * (J_SERCA - J_ER_leak - J_IP3R - J_RyR)
   
   # ATP ODE (unchanged)
   datp_dt = 0
   
   # PMCA state ODE from T-cell model
   w1 = p1 * Ca_i  # Ca2+-dependent activation rate
   taom = 1 / (w1 + p2)  # Time constant for state transition
   dpmcainf = p2 / (w1 + p2)  # Steady state value
   ddpmca_dt = (dpmcainf - dpmca) / taom

   # IP3R state ODEs
   [dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt] = calculate_ip3r_states(
       Ca_i, x000, x100, x010, x001, x110, x101, x011, x111)

   if t % 10 == 0:
       print(f"Time: {t:.2f} ms, V: {V:.2f} mV, Ca_i: {Ca_i:.6f} mM, Ca_ER: {Ca_ER:.6f} mM")
       print(f"IP3R open probability (x110^3): {x110**3:.6f}")
       print(f"IP3R flux: {J_IP3R:.6f}")
       print(f"PMCA state: {dpmca:.6f}")

   return [dV_dt, dCa_dt, datp_dt, ddpmca_dt, dCa_ER_dt, 
           dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt]
           
def run_simulation(params):
   t_span = (0, params['simulation_duration'])
   t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
   
   # Initial conditions including IP3R states
   y0 = [params['initial_voltage'], 
         params['initial_calcium'], 
         params['initial_atp'], 
         params['initial_dpmca'], 
         params['Ca_ER_initial'],
         x000_initial,
         x100_initial,
         x010_initial,
         x001_initial,
         x110_initial,
         x101_initial,
         x011_initial,
         x111_initial]
   
   sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
   return sol

def plot_results_with_two_figures(sol, params):
   # Calculate currents over the entire time range
   I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_IP3R2, J_RyR = [
       np.zeros_like(sol.t) for _ in range(14)
   ]

   # Evaluate currents at each time point with IP3R states and PMCA modulation
   for i in range(len(sol.t)):
       V = sol.y[0][i]
       Ca_i = sol.y[1][i]
       atp = sol.y[2][i]
       dpmca = sol.y[3][i]
       Ca_ER = sol.y[4][i]
       x110 = sol.y[9][i]  # IP3R open state
       
       (
           I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i], d_inf[i], f_inf[i], I_PMCA[i],
           J_SERCA[i], J_ER_leak[i], I_NCX[i], J_IP3R[i], J_IP3R2[i], J_RyR[i]
       ) = calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, params)

   # Figure 1: Membrane Potential, Calcium Concentrations, and IP3R States
   fig1, axs1 = plt.subplots(5, 1, figsize=(12, 25))
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

   # IP3R States
   axs1[3].plot(sol.t, sol.y[9], label='x110 (Open State)', color='purple', linewidth=2)
   axs1[3].plot(sol.t, sol.y[5], label='x000', color='gray', alpha=0.5)
   axs1[3].set_title('IP3R States', fontsize=14, fontweight='bold')
   axs1[3].set_ylabel('State\nProbability', fontsize=12, fontweight='bold')
   axs1[3].legend()
   axs1[3].grid(True)

   # PMCA State
   axs1[4].plot(sol.t, sol.y[3], color='green', linewidth=2)
   axs1[4].set_title('PMCA Modulation State', fontsize=14, fontweight='bold')
   axs1[4].set_ylabel('PMCA\nModulation', fontsize=12, fontweight='bold')
   axs1[4].grid(True)

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

   # PMCA Modulation
   axs2[3].plot(sol.t, sol.y[3], label='dPMCA', color='cyan', linewidth=2)
   axs2[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
   axs2[3].set_title('PMCA Modulation State Over Time', fontsize=14, fontweight='bold')
   axs2[3].set_ylabel('PMCA\nModulation', fontsize=12, fontweight='bold')
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
   'vu': vu,
   'vm': vm,
   'aku': aku,
   'akmp': akmp,
   'aru': aru,
   'arm': arm,
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