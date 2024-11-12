import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import io
import csv
import base64

# Physical constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8.314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)

# Cell parameters
membrane_capacitance = 25e-12  # Membrane capacitance (F)
cell_volume = 1e-15  # Cell volume (L)

# Ion valences
valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_currents(V, Ca_i, atp, dpmca, params):
    # Kir6.1 current (ATP-sensitive K+ channel)
    I_Kir61 = params['conductance_Kir61'] * (atp / (atp + params['K_m_ATP'])) * np.sqrt(params['K_out'] / 5.4) * (V - params['reversal_potential_K']) / (1 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61']) / params['voltage_slope_Kir61']))
    
    # TRPC1 current (non-selective cation channel)
    V_trpc = -20  # Reversal potential for TRPC1 (mV)
    I_TRPC1 = params['conductance_TRPC1'] * (1 / (1 + np.exp(-(V - params['activation_midpoint_TRPC1']) / params['activation_slope_TRPC1']))) * (V - V_trpc)
    
    # CaCC current (Ca2+-activated Cl- channel)
    I_CaCC = params['conductance_CaCC'] * (Ca_i**2 / (Ca_i**2 + params['calcium_activation_threshold_CaCC']**2)) * (V - params['reversal_potential_Cl'])
    
    # CaL current (L-type Ca2+ channel)
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    
    # Leak current (assumed to be mainly K+)
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
    
    # PMCA current (Plasma Membrane Ca2+ ATPase)
    I_PMCA = params['conductance_PMCA'] * (Ca_i / (Ca_i + params['K_m_PMCA'])) * (V - params['reversal_potential_Ca'])
    
    # Na/K-ATPase current
    I_NaK = params['conductance_NaK'] * atp / (atp + params['K_m_ATP']) * (params['Na_in']**1.5 / (params['Na_in']**1.5 + params['K_NaK_Na']**1.5)) * (params['K_out'] / (params['K_out'] + params['K_NaK_K']))
    
    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, I_PMCA, I_NaK, d_inf, f_inf

def model(t, y, params):
    V, Ca_i, atp, dpmca = y
    
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, I_PMCA, I_NaK, _, _ = calculate_currents(V, Ca_i, atp, dpmca, params)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NaK) / membrane_capacitance
    
    # Ca2+ dynamics
    J_in = -I_CaL / (2 * faraday_constant * cell_volume)  # Ca2+ influx
    J_pmca = params['conductance_PMCA'] * (Ca_i / (Ca_i + params['K_m_PMCA']))  # PMCA flux
    J_leak = params['Ca_leak_rate'] * (params['Ca_out'] - Ca_i)  # Ca2+ leak
    dCa_dt = J_in - J_pmca + J_leak
    
    # ATP dynamics
    atp_production_rate = 0.1  # ATP production rate (mM/ms)
    atp_consumption_rate = 0.01  # ATP consumption rate (1/ms)
    datp_dt = atp_production_rate - atp_consumption_rate * atp - 0.1 * np.abs(I_NaK) / (faraday_constant * cell_volume)
    
    # PMCA state variable
    ddpmca_dt = params['k_on_PMCA'] * Ca_i * (1 - dpmca) - params['k_off_PMCA'] * dpmca

    return [dV_dt, dCa_dt, datp_dt, ddpmca_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca']]
    
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA', args=(params,), dense_output=True)
    return sol

def plot_results(sol, params):
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))
    axs[0].plot(sol.t, sol.y[0])
    axs[0].set_ylabel('Membrane Potential (mV)')
    axs[1].plot(sol.t, sol.y[1] * 1e3)  # Convert to µM
    axs[1].set_ylabel('Intracellular Ca2+ (µM)')
    axs[1].set_yscale('log')
    axs[2].plot(sol.t, sol.y[2])
    axs[2].set_ylabel('ATP (mM)')

    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, I_PMCA, I_NaK, d_inf, f_inf = calculate_currents(sol.y[0], sol.y[1], sol.y[2], sol.y[3], params)
    axs[3].plot(sol.t, I_Kir61, label='I_Kir61')
    axs[3].plot(sol.t, I_TRPC1, label='I_TRPC1')
    axs[3].plot(sol.t, I_CaCC, label='I_CaCC')
    axs[3].plot(sol.t, I_CaL, label='I_CaL')
    axs[3].plot(sol.t, I_leak, label='I_leak')
    axs[3].plot(sol.t, I_PMCA, label='I_PMCA')
    axs[3].plot(sol.t, I_NaK, label='I_NaK')
    axs[3].set_ylabel('Current (pA)')
    axs[3].legend()

    axs[4].plot(sol.t, d_inf, label='d_inf')
    axs[4].plot(sol.t, f_inf, label='f_inf')
    axs[4].set_ylabel('Gating variables')
    axs[4].legend()

    axs[5].plot(sol.t, sol.y[3], label='dPMCA')
    axs[5].set_ylabel('dPMCA')
    axs[5].legend()

    for ax in axs:
        ax.set_xlabel('Time (ms)')
    plt.tight_layout()
    return fig

def save_to_csv(sol, params):
    t = sol.t
    V, Ca_i, atp, dpmca = sol.y
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, I_PMCA, I_NaK, d_inf, f_inf = calculate_currents(V, Ca_i, atp, dpmca, params)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time (ms)', 'Membrane Potential (mV)', 'Intracellular Ca2+ (mM)', 'ATP (mM)', 'dPMCA',
                     'I_Kir61 (pA)', 'I_TRPC1 (pA)', 'I_CaCC (pA)', 'I_CaL (pA)', 'I_leak (pA)', 'I_PMCA (pA)', 'I_NaK (pA)',
                     'd_inf', 'f_inf'])
    for i in range(len(t)):
        writer.writerow([t[i], V[i], Ca_i[i], atp[i], dpmca[i],
                         I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i], I_PMCA[i], I_NaK[i],
                         d_inf[i], f_inf[i]])
    
    return output.getvalue()

def export_plots_as_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def main():
    st.title("Improved Pericyte Ion Channel Model Simulator")

    st.sidebar.header("Model Parameters")

    # Ion concentrations
    st.sidebar.subheader("Ion Concentrations")
    K_out = st.sidebar.text_input("Extracellular K+ (mM)", value="5.4")
    K_in = st.sidebar.text_input("Intracellular K+ (mM)", value="140.0")
    Ca_out = st.sidebar.text_input("Extracellular Ca2+ (mM)", value="2.0")
    Ca_in = st.sidebar.text_input("Initial Intracellular Ca2+ (mM)", value="0.0001")
    Na_out = st.sidebar.text_input("Extracellular Na+ (mM)", value="145.0")
    Na_in = st.sidebar.text_input("Intracellular Na+ (mM)", value="10.0")
    Cl_out = st.sidebar.text_input("Extracellular Cl- (mM)", value="125.0")
    Cl_in = st.sidebar.text_input("Intracellular Cl- (mM)", value="30.0")

    # Channel conductances
    st.sidebar.subheader("Channel Conductances")
    conductance_Kir61 = st.sidebar.text_input("Kir6.1 (nS)", value="0.5")
    conductance_TRPC1 = st.sidebar.text_input("TRPC1 (nS)", value="0.1")
    conductance_CaCC = st.sidebar.text_input("CaCC (nS)", value="0.2")
    conductance_CaL = st.sidebar.text_input("CaL (nS)", value="0.3")
    conductance_leak = st.sidebar.text_input("Leak (nS)", value="0.05")
    conductance_PMCA = st.sidebar.text_input("PMCA (nS)", value="0.1")
    conductance_NaK = st.sidebar.text_input("Na/K-ATPase (nS)", value="0.5")

    # Other parameters
    st.sidebar.subheader("Other Parameters")
    Ca_leak_rate = st.sidebar.text_input("Ca2+ Leak Rate (1/ms)", value="0.00001")
    calcium_activation_threshold_CaCC = st.sidebar.text_input("CaCC Activation Threshold (mM)", value="0.0005")
    reference_K = st.sidebar.text_input("Reference K+ (mM)", value="5.4")
    voltage_slope_Kir61 = st.sidebar.text_input("Voltage Slope Kir6.1 (mV)", value="10")
    voltage_shift_Kir61 = st.sidebar.text_input("Voltage Shift Kir6.1 (mV)", value="10")

    # L-type calcium channel parameters
    st.sidebar.subheader("L-type Calcium Channel Parameters")
    activation_midpoint_CaL = st.sidebar.text_input("Activation Midpoint (mV)", value="-20")
    activation_slope_CaL = st.sidebar.text_input("Activation Slope (mV)", value="6")
    inactivation_midpoint_CaL = st.sidebar.text_input("Inactivation Midpoint (mV)", value="-30")
    inactivation_slope_CaL = st.sidebar.text_input("Inactivation Slope (mV)", value="6")

    # TRPC1 parameters
    st.sidebar.subheader("TRPC1 Parameters")
    activation_midpoint_TRPC1 = st.sidebar.text_input("Activation Midpoint (mV)", value="-70")
    activation_slope_TRPC1 = st.sidebar.text_input("Activation Slope (mV)", value="10")

    # PMCA parameters
    st.sidebar.subheader("PMCA Parameters")
    K_m_PMCA = st.sidebar.text_input("K_m PMCA (mM)", value="0.0001")
    k_on_PMCA = st.sidebar.text_input("k_on PMCA (1/(mM*ms))", value="100")
    k_off_PMCA = st.sidebar.text_input("k_off PMCA (1/ms)", value="0.01")

    # Na/K-ATPase parameters
    st.sidebar.subheader("Na/K-ATPase Parameters")
    K_m_ATP = st.sidebar.text_input("K_m ATP (mM)", value="0.5")
    K_NaK_Na = st.sidebar.text_input("K_NaK_Na (mM)", value="10")
    K_NaK_K = st.sidebar.text_input("K_NaK_K (mM)", value="1.5")

    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    simulation_duration = st.sidebar.text_input("Simulation Duration (ms)", value="1000")
    time_points = st.sidebar.text_input("Number of Time Points", value="10000")
    initial_voltage = st.sidebar.text_input("Initial Voltage (mV)", value="-60")
    initial_atp = st.sidebar.text_input("Initial ATP (mM)", value="2")

    # Convert inputs to float
    try:
        params = {
            'K_out': float(K_out), 'K_in': float(K_in), 'Ca_out': float(Ca_out), 'Ca_in': float(Ca_in),
            'Na_out': float(Na_out), 'Na_in': float(Na_in), 'Cl_out': float(Cl_out), 'Cl_in': float(Cl_in),
            'conductance_Kir61': float(conductance_Kir61) * 1e-9, 'conductance_TRPC1': float(conductance_TRPC1) * 1e-9,
            'conductance_CaCC': float(conductance_CaCC) * 1e-9, 'conductance_CaL': float(conductance_CaL) * 1e-9,
            'conductance_leak': float(conductance_leak) * 1e-9, 'conductance_PMCA': float(conductance_PMCA) * 1e-9,
            'conductance_NaK': float(conductance_NaK) * 1e-9,
            'Ca_leak_rate': float(Ca_leak_rate),
            'calcium_activation_threshold_CaCC': float(calcium_activation_threshold_CaCC),
            'reference_K': float(reference_K),
            'voltage_slope_Kir61': float(voltage_slope_Kir61),
            'voltage_shift_Kir61': float(voltage_shift_Kir61),
            'activation_midpoint_CaL': float(activation_midpoint_CaL),
            'activation_slope_CaL': float(activation_slope_CaL),
            'inactivation_midpoint_CaL': float(inactivation_midpoint_CaL),
            'inactivation_slope_CaL': float(inactivation_slope_CaL),
            'activation_midpoint_TRPC1': float(activation_midpoint_TRPC1),
            'activation_slope_TRPC1': float(activation_slope_TRPC1),
            'K_m_PMCA': float(K_m_PMCA),
            'k_on_PMCA': float(k_on_PMCA),
            'k_off_PMCA': float(k_off_PMCA),
            'K_m_ATP': float(K_m_ATP),
            'K_NaK_Na': float(K_NaK_Na),
            'K_NaK_K': float(K_NaK_K),
            'simulation_duration': float(simulation_duration),
            'time_points': int(float(time_points)),
            'initial_voltage': float(initial_voltage),
            'initial_calcium': float(Ca_in),
            'initial_atp': float(initial_atp),
            'initial_dpmca': 0.1  # Initial fraction of active PMCA
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

        # Add export button
        if st.button("Export All Plots as PNG"):
            buf = export_plots_as_png(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="pericyte_simulation_plots.png">Download PNG</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Display final values
        st.subheader("Final Values")
        st.write(f"Membrane Potential: {sol.y[0][-1]:.2f} mV")
        st.write(f"Intracellular Ca2+: {sol.y[1][-1]*1e3:.2f} µM")
        st.write(f"ATP: {sol.y[2][-1]:.2f} mM")
        st.write(f"dPMCA: {sol.y[3][-1]:.4f}")

        # Calculate and display current balance
        I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, I_PMCA, I_NaK, _, _ = calculate_currents(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], params)
        total_inward = I_CaCC + I_CaL + I_TRPC1
        total_outward = I_Kir61 + I_leak + I_PMCA + I_NaK
        balance = total_inward + total_outward

        st.subheader("Current Balance at Steady State")
        st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward*1e12:.4f} fA")
        st.write(f"Total outward current (I_Kir61 + I_leak + I_PMCA + I_NaK): {total_outward*1e12:.4f} fA")
        st.write(f"Balance (should be close to 0): {balance*1e12:.4f} fA")

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
        st.write(f"Mean I_Kir61: {np.mean(I_Kir61)*1e12:.4f} fA")
        st.write(f"Mean I_TRPC1: {np.mean(I_TRPC1)*1e12:.4f} fA")
        st.write(f"Mean I_CaCC: {np.mean(I_CaCC)*1e12:.4f} fA")
        st.write(f"Mean I_CaL: {np.mean(I_CaL)*1e12:.4f} fA")
        st.write(f"Mean I_leak: {np.mean(I_leak)*1e12:.4f} fA")
        st.write(f"Mean I_PMCA: {np.mean(I_PMCA)*1e12:.4f} fA")
        st.write(f"Mean I_NaK: {np.mean(I_NaK)*1e12:.4f} fA")

        # Check current balance at rest
        rest_index = 0  # Assuming the first point is at rest
        I_Kir61_rest, I_TRPC1_rest, I_CaCC_rest, I_CaL_rest, I_leak_rest, I_PMCA_rest, I_NaK_rest, _, _ = calculate_currents(
            sol.y[0][rest_index], sol.y[1][rest_index], sol.y[2][rest_index], sol.y[3][rest_index], params)
        total_inward_rest = I_CaCC_rest + I_CaL_rest + I_TRPC1_rest
        total_outward_rest = I_Kir61_rest + I_leak_rest + I_PMCA_rest + I_NaK_rest
        balance_rest = total_inward_rest + total_outward_rest

        st.subheader("Current Balance at Rest")
        st.write(f"Total inward current (I_CaCC + I_CaL + I_TRPC1): {total_inward_rest*1e12:.4f} fA")
        st.write(f"Total outward current (I_Kir61 + I_leak + I_PMCA + I_NaK): {total_outward_rest*1e12:.4f} fA")
        st.write(f"Balance (should be close to 0): {balance_rest*1e12:.4f} fA")

    except ValueError as e:
        st.error(f"Invalid input: {str(e)}. Please ensure all inputs are valid numbers.")

if __name__ == "__main__":
    main()