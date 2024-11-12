import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import streamlit as st

MIN_VALUE = -80
MAX_VALUE = 40

@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)
 
@njit
def exponential_function(x, a):
    return np.exp(a * x) 

@njit
def ode_system(t, y, g_gap, I_app_val, cm, dx, channel_params, channel_active):
    Ng = len(y)
    Vm = y

    I_channels = np.zeros_like(Vm)
    for i in range(channel_params.shape[0]):
        if channel_active[i]:
            I_channels += channel_params[i, 0] * channel_params[i, 1] * (Vm - channel_params[i, 2])

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

def simulate_and_plot(ggap, cm, dx, I_app_val, t_span, Ng, channel_params, channel_active):
    y0 = np.ones(Ng) * (-33)  # Initial condition

    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, I_app_val, cm, dx, channel_params, channel_active), method='RK45')

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
    ggap = st.sidebar.slider("Gap Junction Conductance", 0.0, 2.0, 1.008702984240095, key="ggap")
    cm = st.sidebar.slider("Membrane Capacitance", 1.0, 20.0, 11.0, key="cm")
    dx = st.sidebar.slider("Spatial Step", 0.1, 2.0, 1.0, key="dx")
    I_app_val = st.sidebar.slider("Applied Current", -100.0, 100.0, -70.0, key="I_app_val")
    t_end = st.sidebar.slider("Simulation Duration", 100, 1000, 600, key="t_end")
    Ng = st.sidebar.slider("Number of Grid Points", 50, 500, 200, key="Ng")

    channel_names = [
        'KIR6_1', 'KIR2_2', 'KIR1_2', 'KIR2_1', 'KIR6_2', 'KV7_4', 'KV7_5', 'KV9_1', 'KV9_3', 'KCA3_1',
        'KCNK1_2', 'KCNK2_3', 'TRPC1', 'TRPC3', 'TRPC4', 'TRPC6', 'TRPM3', 'TRPM4', 'TRPM7', 'TRPML1',
        'TRPP1', 'TRPP3', 'TRPV2', 'IP3R1', 'IP3R2', 'IP3R3', 'CAV1_2', 'CAV1_3', 'CAV2_1', 'CAV3_1',
        'CAV3_2', 'ORAI1', 'ORAI3', 'CACC', 'CIC_2', 'ASIC2', 'NAV1_2', 'NAV1_3', 'P2X1', 'P2X4',
        'PIEZO1', 'TPC1', 'TPC2'
    ]

    st.sidebar.header("Channel Parameters")
    channel_params = np.zeros((len(channel_names), 3))
    channel_active = np.ones(len(channel_names), dtype=bool)
    for i, channel in enumerate(channel_names):
        st.sidebar.subheader(f"{channel} Channel")
        channel_active[i] = st.sidebar.checkbox(f"Activate {channel}", value=False, key=f"{i}_{channel}_active")
        if channel_active[i]:
            channel_params[i, 0] = st.sidebar.slider(f"{channel} Expression", 0.0, 1.0, 0.5, key=f"{i}_{channel}_expression")
            channel_params[i, 1] = st.sidebar.slider(f"{channel} Conductance", 0.0, 2.0, 1.0, key=f"{i}_{channel}_conductance")
            channel_params[i, 2] = st.sidebar.slider(f"{channel} Reversal Potential", -100.0, 120.0, 0.0, key=f"{i}_{channel}_reversal")

    t_span = (0, t_end)

    if st.button("Run Simulation", key="run_simulation"):
        fig = simulate_and_plot(ggap, cm, dx, I_app_val, t_span, Ng, channel_params, channel_active)
        st.pyplot(fig)

if __name__ == "__main__":
    main()