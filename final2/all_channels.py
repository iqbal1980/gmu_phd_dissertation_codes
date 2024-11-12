import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import streamlit as st

MIN_VALUE = -80
MAX_VALUE = 40


def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)
 

def exponential_function(x, a):
    return np.exp(a * x) 

# Define all ion channels
channels = {
    'KIR6_1': {'E': 1.0, 'G': 1.0, 'reversal': -80.0},
    'KIR2_2': {'E': 1.0, 'G': 1.0, 'reversal': -80.0},
    'KIR1_2': {'E': 0.8, 'G': 1.0, 'reversal': -80.0},
    'KIR2_1': {'E': 0.9, 'G': 1.0, 'reversal': -80.0},
    'KIR6_1': {'E': 0.7, 'G': 1.0, 'reversal': -80.0},
    'KV7_4': {'E': 0.6, 'G': 1.0, 'reversal': -80.0},
    'KV7_5': {'E': 0.5, 'G': 1.0, 'reversal': -80.0},
    'KV9_1': {'E': 0.4, 'G': 1.0, 'reversal': -80.0},
    'KV9_3': {'E': 0.3, 'G': 1.0, 'reversal': -80.0},
    'KCA3_1': {'E': 0.7, 'G': 1.0, 'reversal': -80.0},
    'KCNK1_2': {'E': 0.5, 'G': 1.0, 'reversal': -80.0},
    'KCNK2_3': {'E': 0.4, 'G': 1.0, 'reversal': -80.0},
    'TRPC1': {'E': 0.6, 'G': 1.0, 'reversal': 0.0},
    'TRPC3': {'E': 0.5, 'G': 1.0, 'reversal': 0.0},
    'TRPC4': {'E': 0.4, 'G': 1.0, 'reversal': 0.0},
    'TRPC6': {'E': 0.6, 'G': 1.0, 'reversal': 0.0},
    'TRPM3': {'E': 0.5, 'G': 1.0, 'reversal': 0.0},
    'TRPM4': {'E': 0.7, 'G': 1.0, 'reversal': 0.0},
    'TRPM7': {'E': 0.4, 'G': 1.0, 'reversal': 0.0},
    'TRPML1': {'E': 0.3, 'G': 1.0, 'reversal': 0.0},
    'TRPP1': {'E': 0.5, 'G': 1.0, 'reversal': 0.0},
    'TRPP3': {'E': 0.4, 'G': 1.0, 'reversal': 0.0},
    'TRPV2': {'E': 0.6, 'G': 1.0, 'reversal': 0.0},
    'IP3R1': {'E': 0.7, 'G': 1.0, 'reversal': 120.0},
    'IP3R2': {'E': 0.6, 'G': 1.0, 'reversal': 120.0},
    'IP3R3': {'E': 0.5, 'G': 1.0, 'reversal': 120.0},
    'CAV1_2': {'E': 0.8, 'G': 1.0, 'reversal': 120.0},
    'CAV1_3': {'E': 0.7, 'G': 1.0, 'reversal': 120.0},
    'CAV2_1': {'E': 0.6, 'G': 1.0, 'reversal': 120.0},
    'CAV3_1': {'E': 0.5, 'G': 1.0, 'reversal': 120.0},
    'CAV3_2': {'E': 0.4, 'G': 1.0, 'reversal': 120.0},
    'ORAI1': {'E': 0.7, 'G': 1.0, 'reversal': 120.0},
    'ORAI3': {'E': 0.6, 'G': 1.0, 'reversal': 120.0},
    'CACC': {'E': 0.5, 'G': 1.0, 'reversal': -60.0},
    'CIC_2': {'E': 0.4, 'G': 1.0, 'reversal': -60.0},
    'ASIC2': {'E': 0.3, 'G': 1.0, 'reversal': 50.0},
    'NAV1_2': {'E': 0.6, 'G': 1.0, 'reversal': 50.0},
    'NAV1_3': {'E': 0.5, 'G': 1.0, 'reversal': 50.0},
    'P2X1': {'E': 0.4, 'G': 1.0, 'reversal': 0.0},
    'P2X4': {'E': 0.5, 'G': 1.0, 'reversal': 0.0},
    'PIEZO1': {'E': 0.6, 'G': 1.0, 'reversal': 0.0},
    'TPC1': {'E': 0.4, 'G': 1.0, 'reversal': 0.0},
    'TPC2': {'E': 0.3, 'G': 1.0, 'reversal': 0.0}
}


def ode_system(t, y, g_gap, I_app_val, cm, dx, channel_params):
    Ng = len(y)
    Vm = y

    I_channels = np.zeros_like(Vm)
    for channel, params in channel_params.items():
        I_channels += params['E'] * params['G'] * (Vm - params['reversal'])

    I_app = np.zeros_like(Vm)
    I_app[99] = I_app_val if 100 <= t <= 400 else 0.0

    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_channels[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_channels[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_channels[kk] + I_app[kk])

    return dVm_dt

def simulate_and_plot(ggap, cm, dx, I_app_val, t_span, Ng, channel_params):
    y0 = np.ones(Ng) * (-33)  # Initial condition

    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, I_app_val, cm, dx, channel_params), method='RK45')

    x = sol.t
    y = sol.y[100:134]

    fig, ax = plt.subplots()
    ax.plot(x, y.T, linewidth=3)
    ax.set_title(f"G gap = {ggap}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Vm")
    
    return fig

def main():
    st.title("Pericyte Ion Channel Simulation")

    st.sidebar.header("Simulation Parameters")
    ggap = st.sidebar.slider("Gap Junction Conductance", 0.0, 2.0, 1.008702984240095)
    cm = st.sidebar.slider("Membrane Capacitance", 1.0, 20.0, 11.0)
    dx = st.sidebar.slider("Spatial Step", 0.1, 2.0, 1.0)
    I_app_val = st.sidebar.slider("Applied Current", -100.0, 100.0, -70.0)
    t_end = st.sidebar.slider("Simulation Duration", 100, 1000, 600)
    Ng = st.sidebar.slider("Number of Grid Points", 50, 500, 200)

    st.sidebar.header("Channel Parameters")
    channel_params = {}
    for channel, params in channels.items():
        st.sidebar.subheader(f"{channel} Channel")
        E = st.sidebar.slider(f"{channel} Expression", 0.0, 1.0, params['E'])
        G = st.sidebar.slider(f"{channel} Conductance", 0.0, 2.0, params['G'])
        channel_params[channel] = {'E': E, 'G': G, 'reversal': params['reversal']}

    t_span = (0, t_end)

    if st.button("Run Simulation"):
        fig = simulate_and_plot(ggap, cm, dx, I_app_val, t_span, Ng, channel_params)
        st.pyplot(fig)

if __name__ == "__main__":
    main()