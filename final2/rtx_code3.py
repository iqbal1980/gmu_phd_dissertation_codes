import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

faraday_constant = 96485
gas_constant = 8.314
temperature = 310

membrane_capacitance = 0.94
cell_volume = 2e-12
Vcyto = cell_volume

valence_K, valence_Ca, valence_Na, valence_Cl = 1, 2, 1, -1

K_out, K_in = 6.26, 140
Ca_out, Ca_in = 2.0, 0.0001
Na_out, Na_in = 140, 15.38
Cl_out, Cl_in = 110, 9.65

conductance_Kir61 = 0.025
conductance_TRPC1 = 0.001
conductance_CaL = 0.0005
conductance_CaCC = 0.001
conductance_leak = 0.01

ip3_concentration = 0.1
ka_ip3r, ki_ip3r, kact_ip3r = 0.3, 0.5, 0.2

v_serca, k_serca = 0.9, 0.1

er_volume = 0.1 * cell_volume
er_ca_concentration = 400

k_atp, h_atp = 0.1, 2

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_currents(V, Ca_i, K_i, Na_i, Cl_i, atp, ip3, params):
    I_Kir61 = params['conductance_Kir61'] * (1 / (1 + (atp / params['k_atp'])**params['h_atp'])) * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K'])
    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
    I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
    p_open_ip3r = (ip3 / (ip3 + params['ka_ip3r'])) * (Ca_i / (Ca_i + params['kact_ip3r'])) * (1 - Ca_i / (Ca_i + params['ki_ip3r']))
    J_ip3r = p_open_ip3r * (params['er_ca_concentration'] - Ca_i)
    J_serca = params['v_serca'] * (Ca_i**2 / (Ca_i**2 + params['k_serca']**2))
    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, J_ip3r, J_serca

def model(t, y, params):
    V, Ca_i, K_i, Na_i, Cl_i, atp, ip3 = y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, J_ip3r, J_serca = calculate_currents(V, Ca_i, K_i, Na_i, Cl_i, atp, ip3, params)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak) / params['membrane_capacitance']
    
    Ca_influx = -I_CaL / (2 * faraday_constant * params['cell_volume'])
    dCa_dt = Ca_influx + J_ip3r - J_serca - params['ca_extrusion_rate'] * (Ca_i - params['resting_calcium'])
    
    dK_dt = -I_Kir61 / (faraday_constant * params['cell_volume'])
    dNa_dt = 0  # Assuming no significant Na+ flux for simplicity
    dCl_dt = -I_CaCC / (faraday_constant * params['cell_volume'])
    
    datp_dt = -0.01 * atp
    dip3_dt = 0.01 - 0.05 * ip3
    
    return [dV_dt, dCa_dt, dK_dt, dNa_dt, dCl_dt, datp_dt, dip3_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_potassium'], 
          params['initial_sodium'], params['initial_chloride'], params['initial_atp'], params['initial_ip3']]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
    return sol

def plot_results(sol, params):
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 10))
    fig1.suptitle('Pericyte Model Simulation - Part 1', fontsize=16, fontweight='bold')

    axs1[0].plot(sol.t, sol.y[0], color='blue', linewidth=2)
    axs1[0].set_ylabel('Membrane\nPotential (mV)', fontsize=12, fontweight='bold')
    axs1[0].set_title('Membrane Potential Over Time', fontsize=14, fontweight='bold', x=0.1, ha='left')

    axs1[1].plot(sol.t, sol.y[1], color='red', label='Ca2+')
    axs1[1].plot(sol.t, sol.y[2], color='orange', label='K+')
    axs1[1].plot(sol.t, sol.y[3], color='green', label='Na+')
    axs1[1].plot(sol.t, sol.y[4], color='purple', label='Cl-')
    axs1[1].set_ylabel('Ion\nConcentration (mM)', fontsize=12, fontweight='bold')
    axs1[1].set_title('Intracellular Ion Concentrations', fontsize=14, fontweight='bold')
    axs1[1].legend()
    axs1[1].set_yscale('log')

    for ax in axs1:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)

    fig1.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 10))
    fig2.suptitle('Pericyte Model Simulation - Part 2', fontsize=16, fontweight='bold')

    axs2[0].plot(sol.t, sol.y[5], color='green', linewidth=2)
    axs2[0].set_ylabel('ATP\nConcentration (mM)', fontsize=12, fontweight='bold')
    axs2[0].set_title('ATP Concentration Over Time', fontsize=14, fontweight='bold')

    axs2[1].plot(sol.t, sol.y[6], color='orange', linewidth=2)
    axs2[1].set_ylabel('IP3\nConcentration (μM)', fontsize=12, fontweight='bold')
    axs2[1].set_title('IP3 Concentration Over Time', fontsize=14, fontweight='bold')

    for ax in axs2:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)

    fig2.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

params = {
    'K_out': K_out, 'K_in': K_in, 'Ca_out': Ca_out, 'Ca_in': Ca_in,
    'Na_out': Na_out, 'Na_in': Na_in, 'Cl_out': Cl_out, 'Cl_in': Cl_in,
    'conductance_Kir61': conductance_Kir61, 'conductance_TRPC1': conductance_TRPC1,
    'conductance_CaCC': conductance_CaCC, 'conductance_CaL': conductance_CaL,
    'conductance_leak': conductance_leak,
    'membrane_capacitance': membrane_capacitance,
    'cell_volume': cell_volume,
    'er_volume': er_volume,
    'er_ca_concentration': er_ca_concentration,
    'calcium_activation_threshold_CaCC': 0.0005,
    'reference_K': 5.4,
    'activation_midpoint_CaL': -40,
    'activation_slope_CaL': 4,
    'inactivation_midpoint_CaL': -45,
    'inactivation_slope_CaL': 5,
    'voltage_shift_CaL': 50,
    'slope_factor_CaL': 20,
    'amplitude_factor_CaL': 0.6,
    'k_atp': k_atp,
    'h_atp': h_atp,
    'ka_ip3r': ka_ip3r,
    'ki_ip3r': ki_ip3r,
    'kact_ip3r': kact_ip3r,
    'v_serca': v_serca,
    'k_serca': k_serca,
    'simulation_duration': 100,  # Changed to 100 ms to match original
    'time_points': 1000,
    'initial_voltage': -70,
    'initial_calcium': Ca_in,
    'initial_potassium': K_in,
    'initial_sodium': Na_in,
    'initial_chloride': Cl_in,
    'initial_atp': 4.4,
    'initial_ip3': 0.1,
    'ca_extrusion_rate': 100.0,  # Added calcium extrusion rate
    'resting_calcium': 0.0001  # Added resting calcium concentration
}

reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl = calculate_reversal_potentials(
    params['K_out'], params['K_in'], params['Ca_out'], params['Ca_in'],
    params['Na_out'], params['Na_in'], params['Cl_out'], params['Cl_in'])

params.update({
    'reversal_potential_K': reversal_potential_K,
    'reversal_potential_Ca': reversal_potential_Ca,
    'reversal_potential_Na': reversal_potential_Na,
    'reversal_potential_Cl': reversal_potential_Cl
})

sol = run_simulation(params)
plot_results(sol, params)