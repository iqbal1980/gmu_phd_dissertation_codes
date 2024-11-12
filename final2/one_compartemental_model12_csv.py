import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import csv

# Physical constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)

# Cell parameters
membrane_capacitance = 0.94  # Membrane capacitance (pF)
cell_volume = 2e-12  # Cell volume (L)

# Ion concentrations (mM)
potassium_out = 4.5  # Extracellular potassium concentration
potassium_in = 150  # Intracellular potassium concentration
calcium_out = 2.0  # Extracellular calcium concentration (increased)
calcium_in = 0.0001  # Initial intracellular calcium concentration
sodium_out = 140  # Extracellular sodium concentration
sodium_in = 10  # Intracellular sodium concentration
chloride_out = 120  # Extracellular chloride concentration
chloride_in = 30  # Intracellular chloride concentration

# Channel conductances (nS)
conductance_Kir61 = 0.015  # Reduced to decrease hyperpolarization
conductance_TRPC1 = 0.0001  # Slightly increased from 0
conductance_CaCC = 0.007  # Slightly increased
conductance_CaL = 0.0001  # Increased to allow more calcium influx
conductance_leak = 0.002  # Slightly increased

# Ion valences
valence_K = 1  # Potassium ion valence
valence_Ca = 2  # Calcium ion valence
valence_Na = 1  # Sodium ion valence
valence_Cl = -1  # Chloride ion valence

# Reversal potentials (mV)
reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(potassium_out / potassium_in)
reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(calcium_out / calcium_in)
reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(sodium_out / sodium_in)
reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(chloride_out / chloride_in)

# Gating variable parameters for L-type calcium channels
activation_midpoint_CaL = -20  # Half-activation voltage for d_inf (mV) (adjusted)
activation_slope_CaL = 6.24  # Slope factor for d_inf (mV)
inactivation_midpoint_CaL = -35.06  # Half-inactivation voltage for f_inf (mV)
inactivation_slope_CaL = 8.6  # Slope factor for f_inf (mV)
voltage_shift_CaL = 50  # Voltage shift for second component of f_inf (mV)
slope_factor_CaL = 20  # Slope factor for second component of f_inf (mV)
amplitude_factor_CaL = 0.6  # Amplitude factor for second component of f_inf

# Calcium dynamics parameters
calcium_extrusion_rate = 0.5  # Reduced to allow calcium accumulation
resting_calcium = 0.0001  # Kept the same

# Kir6.1 channel parameters
reference_potassium = 5.4  # Reference extracellular potassium concentration (mM)
voltage_slope_Kir61 = 6  # Voltage slope factor for Kir6.1 (mV)
voltage_shift_Kir61 = 15  # Voltage shift for Kir6.1 (mV)

# SERCA pump parameters
serca_pump_rate = 0.1  # Reduced to allow calcium accumulation
serca_km = 0.001  # Km for SERCA pump (mM)

# CaCC channel parameters
calcium_activation_threshold_CaCC = 0.0005  # Reduced to increase CaCC activation at lower calcium levels

def calculate_currents(V, Ca_i, atp):
    I_Kir61 = conductance_Kir61 * atp * np.sqrt(potassium_out / reference_potassium) * (V - reversal_potential_K) / (1 + np.exp((V - reversal_potential_K - voltage_shift_Kir61) / voltage_slope_Kir61))
    
    I_TRPC1 = conductance_TRPC1 * (V - reversal_potential_Ca)
    
    I_CaCC = conductance_CaCC * (Ca_i**2 / (Ca_i**2 + calcium_activation_threshold_CaCC**2)) * (V - reversal_potential_Cl)
    
    d_inf = 1 / (1 + np.exp(-(V - activation_midpoint_CaL) / activation_slope_CaL))
    f_inf = 1 / (1 + np.exp((V - inactivation_midpoint_CaL) / inactivation_slope_CaL)) + amplitude_factor_CaL / (1 + np.exp((voltage_shift_CaL - V) / slope_factor_CaL))
    I_CaL = conductance_CaL * d_inf * f_inf * (V - reversal_potential_Ca)
    
    I_leak = conductance_leak * (V - reversal_potential_K)  # Assuming potassium leak
    
    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf

def model(t, y):
    V, Ca_i, atp = y
    
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _ = calculate_currents(V, Ca_i, atp)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak) / membrane_capacitance
    
    # Modified calcium dynamics equation
    dCa_dt = -I_CaL / (2 * faraday_constant * cell_volume) - calcium_extrusion_rate * (Ca_i - resting_calcium) - serca_pump_rate * (Ca_i**2 / (Ca_i**2 + serca_km**2))
    
    datp_dt = 0  # ATP concentration remains constant

    return [dV_dt, dCa_dt, datp_dt]

# Simulation parameters
simulation_duration = 1000  # Total simulation time (ms)
time_points = 10000  # Number of time points for simulation

# Initial conditions
initial_voltage = -70  # Initial membrane potential (mV)
initial_calcium = 0.0001  # Initial intracellular calcium concentration (mM)
initial_atp = 4.4  # Initial ATP level (unitless)

# Solve ODE
t_span = (0, simulation_duration)
t_eval = np.linspace(0, simulation_duration, time_points)
y0 = [initial_voltage, initial_calcium, initial_atp]
sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA', dense_output=True)

# Function to save data to CSV
def save_to_csv(filename, sol):
    t = sol.t
    V, Ca_i, atp = sol.y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf = calculate_currents(V, Ca_i, atp)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (ms)', 'Membrane Potential (mV)', 'Intracellular Ca2+ (mM)', 'ATP',
                         'I_Kir61 (pA)', 'I_TRPC1 (pA)', 'I_CaCC (pA)', 'I_CaL (pA)', 'I_leak (pA)',
                         'd_inf', 'f_inf'])
        for i in range(len(t)):
            writer.writerow([t[i], V[i], Ca_i[i], atp[i],
                             I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i],
                             d_inf[i], f_inf[i]])

# Save data to CSV
save_to_csv('simulation_results_with_currents_updated.csv', sol)

# Plot results
fig, axs = plt.subplots(5, 1, figsize=(10, 25))
axs[0].plot(sol.t, sol.y[0])
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].plot(sol.t, sol.y[1])
axs[1].set_ylabel('Intracellular Ca2+ (mM)')
axs[1].set_yscale('log')
axs[2].plot(sol.t, sol.y[2])
axs[2].set_ylabel('ATP')

I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf = calculate_currents(sol.y[0], sol.y[1], sol.y[2])
axs[3].plot(sol.t, I_Kir61, label='I_Kir61')
axs[3].plot(sol.t, I_TRPC1, label='I_TRPC1')
axs[3].plot(sol.t, I_CaCC, label='I_CaCC')
axs[3].plot(sol.t, I_CaL, label='I_CaL')
axs[3].plot(sol.t, I_leak, label='I_leak')
axs[3].set_ylabel('Current (pA)')
axs[3].legend()

axs[4].plot(sol.t, d_inf, label='d_inf')
axs[4].plot(sol.t, f_inf, label='f_inf')
axs[4].set_ylabel('Gating variables')
axs[4].legend()

for ax in axs:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()

print("Updated simulation results with currents have been saved to 'simulation_results_with_currents_updated.csv'")

# Analyze currents
print("\nAnalysis of currents:")
print(f"Mean I_Kir61: {np.mean(I_Kir61):.4f} pA")
print(f"Mean I_TRPC1: {np.mean(I_TRPC1):.4f} pA")
print(f"Mean I_CaCC: {np.mean(I_CaCC):.4f} pA")
print(f"Mean I_CaL: {np.mean(I_CaL):.4f} pA")
print(f"Mean I_leak: {np.mean(I_leak):.4f} pA")

# Check current balance at rest
rest_index = -1  # Using the last point as approximate steady state
total_inward = I_CaCC[rest_index] + I_CaL[rest_index] + I_TRPC1[rest_index]
total_outward = I_Kir61[rest_index] + I_leak[rest_index]
balance = total_inward + total_outward

print(f"\nCurrent balance at steady state:")
print(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward:.4f} pA")
print(f"Total outward current (I_Kir61 + I_leak): {total_outward:.4f} pA")
print(f"Balance (should be close to 0): {balance:.4f} pA")