import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants (from paste.txt)
F = 96485  # Faraday constant (C/mol)
R = 8314  # Gas constant (J/(mol*K))
T = 310  # Temperature (K)
z_Ca = 2  # Valence of calcium ion

# Cell geometry (from paste.txt)
V_cell = 2e-12  # Cell volume (L)
V_cyt = 0.55 * V_cell  # Cytosolic volume (L)
V_er = 0.1 * V_cyt  # ER volume (L)
V_ss = 0.1 * V_cyt  # Subspace volume (L)

# Channel parameters (from paste.txt)
g_KIR61 = 0.5  # Conductance of KIR6.1 (nS)
g_KIR22 = 0.3  # Conductance of KIR2.2 (nS)
g_TRPC1 = 0.2  # Conductance of TRPC1 (nS)
g_TRPC3 = 0.15  # Conductance of TRPC3 (nS)
g_CaCC = 0.1  # Conductance of CaCC (nS)
g_ClC2 = 0.05  # Conductance of ClC-2 (nS)

# IP3R parameters (from paste.txt)
k_IP3R = 0.5  # IP3R rate constant (1/s)
K_IP3 = 0.5  # IP3 dissociation constant (μM)
K_act = 0.17  # Ca2+ activation constant (μM)
K_inh = 0.5  # Ca2+ inhibition constant (μM)

# SERCA parameters (from paste.txt)
V_SERCA = 5  # Maximum SERCA pump rate (μM/s)
K_SERCA = 0.1  # SERCA Ca2+ affinity (μM)

# L-type Ca2+ channel parameters (from paste.txt)
P_CaL = 0.3e-7  # Permeability of L-type Ca2+ channel (cm^3/s)
v_half_d = -10  # Half-activation voltage for d gate (mV)
k_d = 6.24  # Slope factor for d gate
v_half_f = -35  # Half-inactivation voltage for f gate (mV)
k_f = 8.6  # Slope factor for f gate
tau_d = 0.1  # Time constant for d gate (s)
tau_f = 0.1  # Time constant for f gate (s)

# Initial concentrations (μM) (from paste.txt)
Ca_cyt_0 = 0.1
Ca_er_0 = 400
Ca_ss_0 = 0.1
IP3_0 = 0.1

# New function for voltage protocol (not in original codes)
def voltage_protocol(t):
    V_hold = -70
    V_step = 0
    t_start = 2
    t_end = 7
    
    if t_start <= t < t_end:
        return V_step
    else:
        return V_hold

# L-type channel gating function (from paste.txt)
def l_type_channel_gates(V):
    d_inf = 1 / (1 + np.exp(-(V - v_half_d) / k_d))
    f_inf = 1 / (1 + np.exp((V - v_half_f) / k_f))
    return d_inf, f_inf

# Main model function (from paste.txt, with modifications for voltage protocol)
def model(y, t):
    Ca_cyt, Ca_er, Ca_ss, IP3, d, f = y
    
    # Get voltage from protocol (new addition)
    V_m = voltage_protocol(t)
    
    # IP3R flux (from paste.txt)
    J_IP3R = k_IP3R * (IP3 / (IP3 + K_IP3)) * (Ca_cyt / (Ca_cyt + K_act)) * \
             (1 - Ca_cyt / (Ca_cyt + K_inh)) * (Ca_er - Ca_cyt)
    
    # SERCA pump (from paste.txt)
    J_SERCA = V_SERCA * (Ca_cyt**2 / (Ca_cyt**2 + K_SERCA**2))
    
    # L-type Ca2+ current (from paste.txt)
    E_Ca = (R * T / (z_Ca * F)) * np.log(2000 / Ca_cyt)  # Reversal potential
    d_inf, f_inf = l_type_channel_gates(V_m)
    I_CaL = P_CaL * d * f * z_Ca**2 * F**2 * V_m / (R * T) * \
            (Ca_cyt * np.exp(z_Ca * F * V_m / (R * T)) - 2000) / \
            (np.exp(z_Ca * F * V_m / (R * T)) - 1)
    
    # Other currents (from paste.txt)
    I_KIR = (g_KIR61 + g_KIR22) * (V_m + 80)  # Assuming reversal potential of -80 mV
    I_TRPC = (g_TRPC1 + g_TRPC3) * V_m
    I_Cl = (g_CaCC + g_ClC2) * (V_m + 30)  # Assuming reversal potential of -30 mV
    
    # Calcium dynamics (from paste.txt)
    dCa_cyt_dt = (J_IP3R - J_SERCA) * (V_er / V_cyt) - I_CaL / (2 * F * V_cyt)
    dCa_er_dt = (-J_IP3R + J_SERCA) * (V_cyt / V_er)
    dCa_ss_dt = I_CaL / (2 * F * V_ss) - (Ca_ss - Ca_cyt) / 0.1  # Simple diffusion from subspace to cytosol
    
    # IP3 dynamics (from paste.txt)
    dIP3_dt = 0.1 - 0.05 * IP3  # Production - degradation
    
    # L-type channel gate dynamics (from paste.txt)
    dd_dt = (d_inf - d) / tau_d
    df_dt = (f_inf - f) / tau_f
    
    return [dCa_cyt_dt, dCa_er_dt, dCa_ss_dt, dIP3_dt, dd_dt, df_dt]

# Simulation parameters (from paste.txt, with modifications)
t = np.linspace(0, 10, 1000)

# Initial conditions (from paste.txt, with modifications for voltage protocol)
V_init = voltage_protocol(0)
d_0, f_0 = l_type_channel_gates(V_init)
y0 = [Ca_cyt_0, Ca_er_0, Ca_ss_0, IP3_0, d_0, f_0]

# Solve ODE (from paste.txt)
sol = odeint(model, y0, t)

# Calculate voltage for plotting (new addition)
V_m = [voltage_protocol(ti) for ti in t]

# Plotting code (modified from paste.txt)
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(t, V_m)
plt.ylabel('Membrane Potential (mV)')
plt.title('Voltage Clamp Protocol')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, sol[:, 0], label='Ca_cyt')
plt.plot(t, sol[:, 1], label='Ca_er')
plt.plot(t, sol[:, 2], label='Ca_ss')
plt.plot(t, sol[:, 3], label='IP3')
plt.ylabel('Concentration (μM)')
plt.title('Pericyte Single-Cell Model')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, sol[:, 4], label='d (activation)')
plt.plot(t, sol[:, 5], label='f (inactivation)')
plt.xlabel('Time (s)')
plt.ylabel('Gate probability')
plt.title('L-type Ca2+ Channel Gates')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()