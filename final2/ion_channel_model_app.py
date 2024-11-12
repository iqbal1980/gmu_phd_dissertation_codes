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

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    print("reversal_potential_K="+str(reversal_potential_K))
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in)
    print("reversal_potential_Ca="+str(reversal_potential_Ca))
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    print("reversal_potential_Na="+str(reversal_potential_Na))
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    print("reversal_potential_Cl="+str(reversal_potential_Cl))
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_currents(V, Ca_i, atp, params):
    I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(params['K_out'] / params['reference_K']) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
    
    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
    
    I_CaCC = params['conductance_CaCC'] * (Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])) * (V - params['reversal_potential_Cl'])
    
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])  # Assuming potassium leak
    
    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf

def model(t, y, params):
    V, Ca_i, atp = y
    
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _ = calculate_currents(V, Ca_i, atp, params)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak) / membrane_capacitance
    
    dCa_dt = -I_CaL / (2 * faraday_constant * cell_volume) - params['calcium_extrusion_rate'] * (Ca_i - params['resting_calcium'])
    
    datp_dt = 0  # ATP concentration remains constant

    return [dV_dt, dCa_dt, datp_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp']]
    
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA', args=(params,), dense_output=True)
    return sol

def plot_results(sol, params):
    fig, axs = plt.subplots(5, 1, figsize=(10, 25))
    axs[0].plot(sol.t, sol.y[0])
    axs[0].set_ylabel('Membrane Potential (mV)')
    axs[1].plot(sol.t, sol.y[1])
    axs[1].set_ylabel('Intracellular Ca2+ (mM)')
    axs[1].set_yscale('log')
    axs[2].plot(sol.t, sol.y[2])
    axs[2].set_ylabel('ATP')

    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf = calculate_currents(sol.y[0], sol.y[1], sol.y[2], params)
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
    return fig

def save_to_csv(sol, params):
    t = sol.t
    V, Ca_i, atp = sol.y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf = calculate_currents(V, Ca_i, atp, params)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time (ms)', 'Membrane Potential (mV)', 'Intracellular Ca2+ (mM)', 'ATP',
                     'I_Kir61 (pA)', 'I_TRPC1 (pA)', 'I_CaCC (pA)', 'I_CaL (pA)', 'I_leak (pA)',
                     'd_inf', 'f_inf'])
    for i in range(len(t)):
        writer.writerow([t[i], V[i], Ca_i[i], atp[i],
                         I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i],
                         d_inf[i], f_inf[i]])
    
    return output.getvalue()

def main():
    st.title("Pericyte Ion Channel Model Simulator")

    st.sidebar.header("Model Parameters")

    # Ion concentrations
    st.sidebar.subheader("Ion Concentrations")
    K_out = st.sidebar.text_input("Extracellular K+ (mM)", value="4.5")
    K_in = st.sidebar.text_input("Intracellular K+ (mM)", value="150.0")
    Ca_out = st.sidebar.text_input("Extracellular Ca2+ (mM)", value="0.05")
    Ca_in = st.sidebar.text_input("Initial Intracellular Ca2+ (mM)", value="0.00001")
    Na_out = st.sidebar.text_input("Extracellular Na+ (mM)", value="140.0")
    Na_in = st.sidebar.text_input("Intracellular Na+ (mM)", value="10.0")
    Cl_out = st.sidebar.text_input("Extracellular Cl- (mM)", value="120.0")
    Cl_in = st.sidebar.text_input("Intracellular Cl- (mM)", value="30.0")

    # Channel conductances
    st.sidebar.subheader("Channel Conductances")
    conductance_Kir61 = st.sidebar.text_input("Kir6.1 (nS)", value="0.025")
    conductance_TRPC1 = st.sidebar.text_input("TRPC1 (nS)", value="0")
    conductance_CaCC = st.sidebar.text_input("CaCC (nS)", value="0.001")
    conductance_CaL = st.sidebar.text_input("CaL (nS)", value="0.00001")
    conductance_leak = st.sidebar.text_input("Leak (nS)", value="0")

    # Other parameters
    st.sidebar.subheader("Other Parameters")
    calcium_extrusion_rate = st.sidebar.text_input("Ca2+ Extrusion Rate (ms^-1)", value="0.1")
    resting_calcium = st.sidebar.text_input("Resting Ca2+ (mM)", value="0.0001")
    calcium_activation_threshold_CaCC = st.sidebar.text_input("CaCC Activation Threshold (mM)", value="0.5")
    reference_K = st.sidebar.text_input("Reference K+ (mM)", value="5.4")
    voltage_slope_Kir61 = st.sidebar.text_input("Voltage Slope Kir6.1 (mV)", value="6")
    voltage_shift_Kir61 = st.sidebar.text_input("Voltage Shift Kir6.1 (mV)", value="15")

    # L-type calcium channel parameters
    st.sidebar.subheader("L-type Calcium Channel Parameters")
    activation_midpoint_CaL = st.sidebar.text_input("Activation Midpoint (mV)", value="-10")
    activation_slope_CaL = st.sidebar.text_input("Activation Slope (mV)", value="6.24")
    inactivation_midpoint_CaL = st.sidebar.text_input("Inactivation Midpoint (mV)", value="-35.06")
    inactivation_slope_CaL = st.sidebar.text_input("Inactivation Slope (mV)", value="8.6")
    voltage_shift_CaL = st.sidebar.text_input("Voltage Shift (mV)", value="50")
    slope_factor_CaL = st.sidebar.text_input("Slope Factor (mV)", value="20")
    amplitude_factor_CaL = st.sidebar.text_input("Amplitude Factor", value="0.6")

    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    simulation_duration = st.sidebar.text_input("Simulation Duration (ms)", value="1000")
    time_points = st.sidebar.text_input("Number of Time Points", value="10000")
    initial_voltage = st.sidebar.text_input("Initial Voltage (mV)", value="-70")
    initial_atp = st.sidebar.text_input("Initial ATP (unitless)", value="4.4")

    # Convert inputs to float
    try:
        params = {
            'K_out': float(K_out), 'K_in': float(K_in), 'Ca_out': float(Ca_out), 'Ca_in': float(Ca_in),
            'Na_out': float(Na_out), 'Na_in': float(Na_in), 'Cl_out': float(Cl_out), 'Cl_in': float(Cl_in),
            'conductance_Kir61': float(conductance_Kir61), 'conductance_TRPC1': float(conductance_TRPC1),
            'conductance_CaCC': float(conductance_CaCC), 'conductance_CaL': float(conductance_CaL),
            'conductance_leak': float(conductance_leak),
            'calcium_extrusion_rate': float(calcium_extrusion_rate),
            'resting_calcium': float(resting_calcium),
            'calcium_activation_threshold_CaCC': float(calcium_activation_threshold_CaCC),
            'reference_K': float(reference_K),
            'voltage_slope_Kir61': float(voltage_slope_Kir61),
            'voltage_shift_Kir61': float(voltage_shift_Kir61),
            'activation_midpoint_CaL': float(activation_midpoint_CaL),
            'activation_slope_CaL': float(activation_slope_CaL),
            'inactivation_midpoint_CaL': float(inactivation_midpoint_CaL),
            'inactivation_slope_CaL': float(inactivation_slope_CaL),
            'voltage_shift_CaL': float(voltage_shift_CaL),
            'slope_factor_CaL': float(slope_factor_CaL),
            'amplitude_factor_CaL': float(amplitude_factor_CaL),
            'simulation_duration': float(simulation_duration),
            'time_points': int(float(time_points)),
            'initial_voltage': float(initial_voltage),
            'initial_calcium': float(Ca_in),
            'initial_atp': float(initial_atp)
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
# Display final values
        st.subheader("Final Values")
        st.write(f"Membrane Potential: {sol.y[0][-1]:.2f} mV")
        st.write(f"Intracellular Ca2+: {sol.y[1][-1]:.6f} mM")
        st.write(f"ATP: {sol.y[2][-1]:.2f} (unitless)")

        # Calculate and display current balance
        I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, _, _ = calculate_currents(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], params)
        total_inward = I_CaCC + I_CaL + I_TRPC1
        total_outward = I_Kir61 + I_leak
        balance = total_inward + total_outward

        st.subheader("Current Balance at Steady State")
        st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward:.4f} pA")
        st.write(f"Total outward current (I_Kir61 + I_leak): {total_outward:.4f} pA")
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
        st.write(f"Mean I_Kir61: {np.mean(I_Kir61):.4f} pA")
        st.write(f"Mean I_TRPC1: {np.mean(I_TRPC1):.4f} pA")
        st.write(f"Mean I_CaCC: {np.mean(I_CaCC):.4f} pA")
        st.write(f"Mean I_CaL: {np.mean(I_CaL):.4f} pA")
        st.write(f"Mean I_leak: {np.mean(I_leak):.4f} pA")

        # Check current balance at rest
        rest_index = 0  # Assuming the first point is at rest
        total_inward_rest = I_CaCC[rest_index] + I_CaL[rest_index] + I_TRPC1[rest_index]
        total_outward_rest = I_Kir61[rest_index] + I_leak[rest_index]
        balance_rest = total_inward_rest + total_outward_rest

        st.subheader("Current Balance at Rest")
        st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward_rest:.4f} pA")
        st.write(f"Total outward current (I_Kir61 + I_leak): {total_outward_rest:.4f} pA")
        st.write(f"Balance (should be close to 0): {balance_rest:.4f} pA")

    except ValueError as e:
        st.error(f"Invalid input: {str(e)}. Please ensure all inputs are valid numbers.")

if __name__ == "__main__":
    main()