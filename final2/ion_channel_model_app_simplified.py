import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import io
import csv
import base64

# Physical constants
faraday_constant = 96485  # Faraday constant (C/mol)
gas_constant = 8314  # Gas constant (J/(mol*K))
temperature = 310  # Temperature (K)
membrane_capacitance = 0.94  # Membrane capacitance (pF)
cell_volume = 2e-12  # Cell volume (L)

# Function to calculate reversal potentials
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant * temperature / (1 * faraday_constant)) * np.log(K_out / K_in)
    reversal_potential_Ca = (gas_constant * temperature / (2 * faraday_constant)) * np.log(Ca_out / Ca_in)
    reversal_potential_Na = (gas_constant * temperature / (1 * faraday_constant)) * np.log(Na_out / Na_in)
    reversal_potential_Cl = (gas_constant * temperature / (-1 * faraday_constant)) * np.log(Cl_out / Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

# Function to calculate ion currents
def calculate_currents(V, params):
    I_TRPC1 = params['conductance_TRPC1'] * (V - params['reversal_potential_Ca'])
    I_leak = params['conductance_leak'] * (V - params['reversal_potential_K'])
    return I_TRPC1, I_leak

# ODE model
def model(t, y, params):
    V = y[0]
    I_TRPC1, I_leak = calculate_currents(V, params)
    dV_dt = (-I_TRPC1 - I_leak) / membrane_capacitance
    return [dV_dt]

# Function to run simulation
def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage']]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, args=(params,), method='RK45')
    return sol

# Function to plot results
def plot_results(sol):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sol.t, sol.y[0], label='Membrane Potential (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.legend()
    ax.grid()
    return fig

# Function to save simulation results as CSV
def save_to_csv(sol):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time (ms)', 'Membrane Potential (mV)'])
    for i in range(len(sol.t)):
        writer.writerow([sol.t[i], sol.y[0][i]])
    return output.getvalue()

# Streamlit app main function
def main():
    st.title("Simplified Pericyte Ion Channel Model Simulator")

    st.sidebar.header("Model Parameters")

    # Allow user to set only the two influential parameters
    st.sidebar.subheader("Key Channel Conductances")
    conductance_TRPC1 = st.sidebar.slider("TRPC1 (nS)", 0.0, 1.0, 0.5)
    conductance_leak = st.sidebar.slider("Leak (nS)", 0.0, 1.0, 0.1)

    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    simulation_duration = st.sidebar.slider("Simulation Duration (ms)", 10, 200, 100)
    time_points = st.sidebar.slider("Number of Time Points", 100, 2000, 1000)
    initial_voltage = st.sidebar.slider("Initial Voltage (mV)", -100, 50, -70)

    # Default constant parameters
    params = {
        'K_out': 4.5, 'K_in': 150.0, 'Ca_out': 2.0, 'Ca_in': 0.0001,
        'Na_out': 140.0, 'Na_in': 10.0, 'Cl_out': 120.0, 'Cl_in': 30.0,
        'conductance_TRPC1': conductance_TRPC1,
        'conductance_leak': conductance_leak,
        'simulation_duration': simulation_duration,
        'time_points': time_points,
        'initial_voltage': initial_voltage,
    }

    # Calculate reversal potentials with fixed concentrations
    reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl = calculate_reversal_potentials(
        params['K_out'], params['K_in'], params['Ca_out'], params['Ca_in'],
        params['Na_out'], params['Na_in'], params['Cl_out'], params['Cl_in'])

    params.update({
        'reversal_potential_K': reversal_potential_K,
        'reversal_potential_Ca': reversal_potential_Ca,
        'reversal_potential_Na': reversal_potential_Na,
        'reversal_potential_Cl': reversal_potential_Cl
    })

    # Run the simulation and plot results
    sol = run_simulation(params)
    fig = plot_results(sol)
    st.pyplot(fig)

    # Export simulation results as CSV
    csv_data = save_to_csv(sol)
    st.download_button(label="Download CSV", data=csv_data, file_name="simulation_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
