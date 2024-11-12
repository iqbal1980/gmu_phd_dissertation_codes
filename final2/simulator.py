import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.extending import overload
from scipy.optimize import curve_fit
#from numba import generated_jit
from numba import types
import streamlit as st

# Random number not needed really
random_number = 1

global_dx = 1  # No spatial discretisation in network model

# Constants for safe_log and safe_exponential

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
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)



MIN_VM = -80  # Minimum physiologically reasonable value for Vm
MAX_VM = 40  # Maximum physiologically reasonable value for Vm



@njit(parallel=False)
def simulate_process(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_value):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(eki1))
    
    eki2 = dt / cm
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(eki1))

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)
    distance_m = np.zeros(Ng)
    vstims = np.zeros(Ng)
    vresps = np.zeros(Ng)

    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            #I_app[99] = 50.0
            I_app[99] = I_app_value
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o/150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm += random_number * 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                new_Vm += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VM), MIN_VM)

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

        # Debugging: Check for NaNs in Vm
        #if np.any(np.isnan(Vm)):
        #    print(f"NaN detected in Vm at iteration {j}")
        #    print(f"Vm: {Vm}")

    return A
    
    
def plot_data2(A, ggap):
    cellLength = 60  # in Microns
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / np.abs(A[99000, 101:135])[0]
    distance_m = cellLength * np.arange(34)  # Adjusted to match your subset of data

    A_initial = D[0]
    B_initial = np.log(D[1] / D[0]) / (distance_m[1] - distance_m[0])

    try:
        popt, pcov = curve_fit(exponential_decay_function, distance_m, D, p0=[A_initial, B_initial])
    except RuntimeError as e:
        st.error("Error in curve fitting: " + str(e))
        return

    A, B = popt
    x_fit = np.linspace(min(distance_m), max(distance_m), 100)
    y_fit = exponential_decay_function(x_fit, A, B)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(distance_m, D, 'o', label='Data')
    ax.plot(x_fit, y_fit, '-', label=f'Fit: A={A:.2f}, B={B:.2f}')
    ax.set_title(f"Exponential Decay Fit for G_gap = {ggap}")
    ax.set_xlabel("Distance (Microns)")
    ax.set_ylabel("Delta Vm")
    ax.legend()

    return fig    
    
    
    
    
    
    
    
    
    
    
def plot_data2_modified(A, ggap):
    cellLength = 60  # in Microns
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / np.abs(A[99000, 101:135])[0]
    distance_m = cellLength * np.arange(34)  # Adjusted for simplicity

    A_initial = D[0]
    B_initial = np.log(D[1] / D[0]) / (distance_m[1] - distance_m[0])

    # Curve fitting
    popt, _ = curve_fit(exponential_decay_function, distance_m, D, p0=[A_initial, B_initial])
    A, B = popt

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(distance_m, D, '.', markersize=8, label='Data Points')
    x_fit = np.linspace(min(distance_m), max(distance_m), 100)
    y_fit = exponential_decay_function(x_fit, A, B)
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label='Fit: Exponential Decay')
    
    # Reference line
    y_ref = np.exp(-0.003 * distance_m)
    ax.plot(distance_m, y_ref, 'b-', linewidth=2, label='Reference: Experimental Decay')

    ax.set_ylim(-0.5, 1)
    ax.legend()
    ax.set_title(f'Chart for Ggap = {ggap}')
    
    return fig



    
    
    
    
    
    
    

# Streamlit app starts here
st.title("Simulation and Analysis")

# Sliders for parameters
g_gap_value = st.slider("G_gap value", min_value=0.0, max_value=1000.0, value=20.01, step=0.01)
Ibg_init = st.slider("Ibg_init", min_value=0.0, max_value=1.0, value=0.658, step=0.001)
Ikir_coef = st.slider("Ikir_coef", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
cm = st.slider("cm", min_value=0.0, max_value=20.0, value=11.0, step=0.1)
dx = 1  # Assuming dx is fixed as per your setup
K_o = st.slider("K_o", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
I_app_value = st.slider("I_app at cell 99", min_value=-100.0, max_value=100.0, value=-70.0, step=1.0)






# Button to run additional analysis and plot generation
if st.button("Run Full Analysis"):
    A = simulate_process(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_value)

    # Membrane Potential Plot
    fig_mp, ax_mp = plt.subplots()
    ax_mp.plot(A[:,0], A[:,101:135])  # Modify as needed to select which cells to plot
    ax_mp.set_title("Membrane Potential Over Time")
    ax_mp.set_xlabel("Time (s)")
    ax_mp.set_ylabel("Membrane Potential (mV)")
    st.pyplot(fig_mp)

    # Exponential Decay Fit Plot
    fig_decay = plot_data2(A, g_gap_value)
    st.pyplot(fig_decay)
    
    
    
# Streamlit app section for generating plots
if st.button("Generate Plots"):
    A = simulate_process(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_value)
    # Generate the plot with reference included
    plot_with_ref = plot_data2_modified(A, g_gap_value)
    st.pyplot(plot_with_ref)