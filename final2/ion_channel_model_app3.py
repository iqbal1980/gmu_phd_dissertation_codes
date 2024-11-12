import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import io
import csv

# Physical constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)

# Cell parameters
membrane_capacitance = 0.94  # Membrane capacitance (pF)
cell_volume = 2e-12  # Cell volume (L)

# Ion valences
valence_K = 1  # Potassium ion valence
valence_Ca = 2  # Calcium ion valence
valence_Na = 1  # Sodium ion valence
valence_Cl = -1  # Chloride ion valence

# PMCA-related constants
z = 2  # Valence of calcium ions
Vcyto = cell_volume  # Cytoplasmic volume (L)

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in) * 1e3  # Convert to mV
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in) * 1e3
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in) * 1e3
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in) * 1e3
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_currents(V, Ca_i, atp, dpmca, params):
    # ATP effect on Kir6.1 channel
    ATP_effect = atp / (atp + params['ATP_half_max'])
    I_Kir61 = params['conductance_Kir61'] * ATP_effect * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))

    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])

    I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])

    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])

    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])  # Assuming potassium leak

    # PMCA calculations
    u4 = (params['vu'] * (Ca_i**params['aru']) / (Ca_i**params['aru'] + params['aku']**params['aru'])) / (6.6253e5)
    u5 = (params['vm'] * (Ca_i**params['arm']) / (Ca_i**params['arm'] + params['akmp']**params['arm'])) / (6.6253e5)

    cJpmca = (dpmca * u4 + (1 - dpmca) * u5)
    I_PMCA = cJpmca * z * faraday_constant * Vcyto * 1e6  # convert flux (μM/s) to current (pA)

    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, u4, u5

def model(t, y, params):
    V, Ca_i, atp, dpmca = y

    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, u4, u5 = calculate_currents(V, Ca_i, atp, dpmca, params)

    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA) / membrane_capacitance

    dCa_dt = -I_CaL / (2 * faraday_constant * cell_volume * 1e3) - params['calcium_extrusion_rate'] * (Ca_i - params['resting_calcium'])  # Convert pA to A

    # Dynamic ATP function
    dATP_dt = params['ATP_production_rate'] - params['ATP_consumption_rate'] * atp

    # PMCA state variable
    w1 = 0.1 * Ca_i
    w2 = 0.01
    taom = 1 / (w1 + w2)
    dpmcainf = w2 / (w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca) / taom

    return [dV_dt, dCa_dt, dATP_dt, ddpmca_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca']]

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='BDF', args=(params,), dense_output=True)
    return sol

def plot_results(sol, params):
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))
    axs[0].plot(sol.t, sol.y[0])
    axs[0].set_ylabel('Membrane Potential (mV)')
    axs[0].set_title('Membrane Potential Over Time')
    axs[0].grid(True)

    axs[1].plot(sol.t, sol.y[1])
    axs[1].set_ylabel('Intracellular Ca²⁺ (mM)')
    axs[1].set_title('Intracellular Calcium Concentration Over Time')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    axs[2].plot(sol.t, sol.y[2])
    axs[2].set_ylabel('ATP Concentration (mM)')
    axs[2].set_title('ATP Concentration Over Time')
    axs[2].grid(True)

    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, u4, u5 = calculate_currents(sol.y[0], sol.y[1], sol.y[2], sol.y[3], params)
    axs[3].plot(sol.t, I_Kir61, label='I_Kir6.1')
    axs[3].plot(sol.t, I_TRPC1, label='I_TRPC1')
    axs[3].plot(sol.t, I_CaCC, label='I_CaCC')
    axs[3].plot(sol.t, I_CaL, label='I_CaL')
    axs[3].plot(sol.t, I_leak, label='I_leak')
    axs[3].plot(sol.t, I_PMCA, label='I_PMCA')
    axs[3].set_ylabel('Current (pA)')
    axs[3].set_title('Currents Over Time')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(sol.t, d_inf, label='d_inf')
    axs[4].plot(sol.t, f_inf, label='f_inf')
    axs[4].set_ylabel('Gating Variables')
    axs[4].set_title('Gating Variables Over Time')
    axs[4].legend()
    axs[4].grid(True)

    axs[5].plot(sol.t, sol.y[3], label='dPMCA')
    axs[5].set_ylabel('dPMCA')
    axs[5].set_title('PMCA State Variable Over Time')
    axs[5].legend()
    axs[5].grid(True)

    for ax in axs:
        ax.set_xlabel('Time (ms)')
    plt.tight_layout()
    return fig

def save_to_csv(sol, params):
    t = sol.t
    V, Ca_i, atp, dpmca = sol.y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, u4, u5 = calculate_currents(V, Ca_i, atp, dpmca, params)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time (ms)', 'Membrane Potential (mV)', 'Intracellular Ca²⁺ (mM)', 'ATP (mM)', 'dPMCA',
                     'I_Kir6.1 (pA)', 'I_TRPC1 (pA)', 'I_CaCC (pA)', 'I_CaL (pA)', 'I_leak (pA)', 'I_PMCA (pA)',
                     'd_inf', 'f_inf', 'u4', 'u5'])
    for i in range(len(t)):
        writer.writerow([t[i], V[i], Ca_i[i], atp[i], dpmca[i],
                         I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i], I_PMCA[i],
                         d_inf[i], f_inf[i], u4[i], u5[i]])

    return output.getvalue()

def main():
    st.title("Pericyte Ion Channel Model Simulator")

    st.sidebar.header("Model Parameters")

    # Organize UI elements using expanders
    with st.sidebar.expander("Ion Concentrations", expanded=True):
        K_out = st.number_input("Extracellular K⁺ (mM)", value=4.5)
        K_in = st.number_input("Intracellular K⁺ (mM)", value=150.0)
        Ca_out = st.number_input("Extracellular Ca²⁺ (mM)", value=2.0)
        Ca_in = st.number_input("Initial Intracellular Ca²⁺ (mM)", value=0.0001, format="%.5f")
        Na_out = st.number_input("Extracellular Na⁺ (mM)", value=140.0)
        Na_in = st.number_input("Intracellular Na⁺ (mM)", value=10.0)
        Cl_out = st.number_input("Extracellular Cl⁻ (mM)", value=120.0)
        Cl_in = st.number_input("Intracellular Cl⁻ (mM)", value=30.0)

    with st.sidebar.expander("Channel Conductances", expanded=False):
        conductance_Kir61 = st.number_input("Kir6.1 Conductance (nS)", value=0.025)
        conductance_TRPC1 = st.number_input("TRPC1 Conductance (nS)", value=0.0)
        conductance_CaCC = st.number_input("CaCC Conductance (nS)", value=0.001)
        conductance_CaL = st.number_input("CaL Conductance (nS)", value=0.00001, format="%.5f")
        conductance_leak = st.number_input("Leak Conductance (nS)", value=0.0)

    with st.sidebar.expander("Other Parameters", expanded=False):
        calcium_extrusion_rate = st.number_input("Ca²⁺ Extrusion Rate (ms⁻¹)", value=0.1)
        resting_calcium = st.number_input("Resting Ca²⁺ (mM)", value=0.0001, format="%.5f")
        calcium_activation_threshold_CaCC = st.number_input("CaCC Activation Threshold (mM)", value=0.0005, format="%.5f")
        reference_K = st.number_input("Reference K⁺ (mM)", value=5.4)
        voltage_slope_Kir61 = st.number_input("Voltage Slope Kir6.1 (mV)", value=6.0)
        voltage_shift_Kir61 = st.number_input("Voltage Shift Kir6.1 (mV)", value=15.0)

    with st.sidebar.expander("L-type Calcium Channel Parameters", expanded=False):
        activation_midpoint_CaL = st.number_input("Activation Midpoint (mV)", value=-10.0)
        activation_slope_CaL = st.number_input("Activation Slope (mV)", value=6.24)
        inactivation_midpoint_CaL = st.number_input("Inactivation Midpoint (mV)", value=-35.06)
        inactivation_slope_CaL = st.number_input("Inactivation Slope (mV)", value=8.6)
        voltage_shift_CaL = st.number_input("Voltage Shift (mV)", value=50.0)
        slope_factor_CaL = st.number_input("Slope Factor (mV)", value=20.0)
        amplitude_factor_CaL = st.number_input("Amplitude Factor", value=0.6)

    with st.sidebar.expander("PMCA Parameters", expanded=False):
        vu = st.number_input("vu (ions/s)", value=1540000.0)
        vm = st.number_input("vm (ions/s)", value=2200000.0)
        aku = st.number_input("aku (μM)", value=0.303)
        akmp = st.number_input("akmp (μM)", value=0.14)
        aru = st.number_input("aru", value=1.8)
        arm = st.number_input("arm", value=2.1)
        initial_dpmca = st.number_input("Initial dPMCA", value=1.0)

    with st.sidebar.expander("ATP Parameters", expanded=False):
        ATP_production_rate = st.number_input("ATP Production Rate (mM/ms)", value=0.1)
        ATP_consumption_rate = st.number_input("ATP Consumption Rate (ms⁻¹)", value=0.01)
        ATP_half_max = st.number_input("Half-maximal ATP Concentration (mM)", value=1.0)
        initial_atp = st.number_input("Initial ATP Concentration (mM)", value=2.0)

    with st.sidebar.expander("Simulation Parameters", expanded=False):
        simulation_duration = st.number_input("Simulation Duration (ms)", value=1000.0)
        time_points = st.number_input("Number of Time Points", value=10000, step=1000)
        initial_voltage = st.number_input("Initial Voltage (mV)", value=-70.0)

    # Collect parameters into a dictionary
    params = {
        'K_out': K_out, 'K_in': K_in, 'Ca_out': Ca_out, 'Ca_in': Ca_in,
        'Na_out': Na_out, 'Na_in': Na_in, 'Cl_out': Cl_out, 'Cl_in': Cl_in,
        'conductance_Kir61': conductance_Kir61, 'conductance_TRPC1': conductance_TRPC1,
        'conductance_CaCC': conductance_CaCC, 'conductance_CaL': conductance_CaL,
        'conductance_leak': conductance_leak,
        'calcium_extrusion_rate': calcium_extrusion_rate,
        'resting_calcium': resting_calcium,
        'calcium_activation_threshold_CaCC': calcium_activation_threshold_CaCC,
        'reference_K': reference_K,
        'voltage_slope_Kir61': voltage_slope_Kir61,
        'voltage_shift_Kir61': voltage_shift_Kir61,
        'activation_midpoint_CaL': activation_midpoint_CaL,
        'activation_slope_CaL': activation_slope_CaL,
        'inactivation_midpoint_CaL': inactivation_midpoint_CaL,
        'inactivation_slope_CaL': inactivation_slope_CaL,
        'voltage_shift_CaL': voltage_shift_CaL,
        'slope_factor_CaL': slope_factor_CaL,
        'amplitude_factor_CaL': amplitude_factor_CaL,
        'vu': vu,
        'vm': vm,
        'aku': aku,
        'akmp': akmp,
        'aru': aru,
        'arm': arm,
        'initial_dpmca': initial_dpmca,
        'ATP_production_rate': ATP_production_rate,
        'ATP_consumption_rate': ATP_consumption_rate,
        'ATP_half_max': ATP_half_max,
        'initial_atp': initial_atp,
        'simulation_duration': simulation_duration,
        'time_points': int(time_points),
        'initial_voltage': initial_voltage,
        'initial_calcium': Ca_in
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
    fig = plot_results(sol, params)
    st.pyplot(fig)

    # Display final values
    st.subheader("Final Values")
    st.write(f"Membrane Potential: {sol.y[0][-1]:.2f} mV")
    st.write(f"Intracellular Ca²⁺: {sol.y[1][-1]:.6f} mM")
    st.write(f"ATP Concentration: {sol.y[2][-1]:.4f} mM")
    st.write(f"dPMCA: {sol.y[3][-1]:.4f}")

    # Calculate and display current balance
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _, I_PMCA, _, _ = calculate_currents(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], params)
    total_inward = I_CaCC + I_CaL + I_TRPC1
    total_outward = I_Kir61 + I_leak + I_PMCA
    balance = total_inward + total_outward

    st.subheader("Current Balance at Steady State")
    st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward:.4f} pA")
    st.write(f"Total outward current (I_Kir6.1 + I_leak + I_PMCA): {total_outward:.4f} pA")
    st.write(f"Balance (should be close to 0): {balance:.4f} pA")

    # Add download button for CSV
    csv_data = save_to_csv(sol, params)
    st.download_button(
        label="Download simulation results as CSV",
        data=csv_data,
        file_name="pericyte_simulation_results.csv",
        mime="text/csv"
    )

    # Analyze currents
    st.subheader("Analysis of Currents")
    I_Kir61_all, I_TRPC1_all, I_CaCC_all, I_CaL_all, I_leak_all, _, _, I_PMCA_all, _, _ = calculate_currents(sol.y[0], sol.y[1], sol.y[2], sol.y[3], params)
    st.write(f"Mean I_Kir6.1: {np.mean(I_Kir61_all):.4f} pA")
    st.write(f"Mean I_TRPC1: {np.mean(I_TRPC1_all):.4f} pA")
    st.write(f"Mean I_CaCC: {np.mean(I_CaCC_all):.4f} pA")
    st.write(f"Mean I_CaL: {np.mean(I_CaL_all):.4f} pA")
    st.write(f"Mean I_leak: {np.mean(I_leak_all):.4f} pA")
    st.write(f"Mean I_PMCA: {np.mean(I_PMCA_all):.4f} pA")

    # Check current balance at rest
    rest_index = 0  # Assuming the first point is at rest
    I_Kir61_rest, I_TRPC1_rest, I_CaCC_rest, I_CaL_rest, I_leak_rest, _, _, I_PMCA_rest, _, _ = calculate_currents(sol.y[0][rest_index], sol.y[1][rest_index], sol.y[2][rest_index], sol.y[3][rest_index], params)
    total_inward_rest = I_CaCC_rest + I_CaL_rest + I_TRPC1_rest
    total_outward_rest = I_Kir61_rest + I_leak_rest + I_PMCA_rest
    balance_rest = total_inward_rest + total_outward_rest

    st.subheader("Current Balance at Rest")
    st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward_rest:.4f} pA")
    st.write(f"Total outward current (I_Kir6.1 + I_leak + I_PMCA): {total_outward_rest:.4f} pA")
    st.write(f"Balance (should be close to 0): {balance_rest:.4f} pA")

if __name__ == "__main__":
    main()
