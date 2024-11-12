import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
F = 96485  # Faraday constant (C/mol)
R = 8314  # Gas constant (J/(mol*K))
T = 310  # Temperature (K)

# Cell parameters
Cm = 11.0  # Membrane capacitance (pF) - from ode_system2.py
V_cell = 2e-12  # Cell volume (L) - from PIIS2405844020303716.pdf

# Ion concentrations (mM) - from test.f
K_o = 4.5  # Extracellular potassium
K_i = 150  # Intracellular potassium
Ca_o = 2.0  # Extracellular calcium
Na_o = 140  # Extracellular sodium
Na_i = 10  # Intracellular sodium

# Channel conductances (nS) - qualitatively estimated based on relative expression levels in Fig 9
g_Kir61 = 4.0  # Kir6.1 conductance (highest expressed)
g_Kir22 = 3.5  # Kir2.2 conductance (second highest expressed)
g_CaL = 0.3  # L-type calcium conductance (Ca_v1.2, moderately expressed)
g_TRPC = 0.5  # TRPC conductance (TRPC1 and TRPC3, moderately expressed)

# Reversal potentials (mV) - calculated using Nernst equation
E_K = (R * T / F) * np.log(K_o / K_i)
E_Ca = (R * T / (2 * F)) * np.log(Ca_o / 0.0001)  # Assuming intracellular Ca2+ of 100 nM
E_Na = (R * T / F) * np.log(Na_o / Na_i)

# IP3R parameters - from test.f
v1 = 90.0
a1 = 400.0
a2 = 0.2
a3 = 400.0
a4 = 0.2
a5 = 20.0
d1 = 0.13
d2 = 1.049
d3 = 0.9434
d4 = 0.1445
d5 = 82.34e-3

def I_stim(t):
    return 10 if 200 <= t <= 700 else 0

def model(y, t):
    V, Ca_i, IP3, h, atp = y
    
    # Kir6.1 current (KATP) - adapted based on ode_system2.py Kir current
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    
    # Kir2.2 current - adapted based on ode_system2.py Kir current
    I_Kir22 = g_Kir22 * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    
    # L-type Ca channel (Ca_v1.2) - adapted from test.f
    d_inf = 1 / (1 + np.exp(-(V + 10) / 6.24))
    f_inf = 1 / (1 + np.exp((V + 35.06) / 8.6)) + 0.6 / (1 + np.exp((50 - V) / 20))
    I_CaL = g_CaL * d_inf * f_inf * (V - E_Ca)
    
    # TRPC channel - simplified linear model based on ode_system2.py
    I_TRPC = g_TRPC * (V - E_Ca)
    
    # IP3R flux - adapted from test.f
    J_IP3R = 1
    
    # h gate for IP3R
    h_inf = d2 / (Ca_i + d2)
    tau_h = 1 / (a2 * (Ca_i + d2))
    dh_dt = (h_inf - h) / tau_h
    
    # Calcium dynamics
    dCa_dt = -I_CaL / (2 * F * V_cell) + J_IP3R / V_cell
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_Kir22 - I_CaL - I_TRPC + I_stim(t)) / Cm
    
    # IP3 dynamics - simplified model
    dIP3_dt = 0.1 * (1 - IP3) - 0.1 * IP3
    
    # ATP dynamics - simplified model
    datp_dt = 0.01 * (1 - atp) - 0.1 * atp

    return [dV_dt, dCa_dt, dIP3_dt, dh_dt, datp_dt]

# Simulation parameters
t = np.linspace(0, 1000, 10000)

# Initial conditions
y0 = [-70, 0.1, 0.1, 0.5, 0.5]  # V (mV), Ca_i (μM), IP3 (μM), h, atp

# Solve ODE
sol = odeint(model, y0, t)

# Plot results
fig, axs = plt.subplots(3, 2, figsize=(12, 15))
axs[0, 0].plot(t, sol[:, 0])
axs[0, 0].set_ylabel('Membrane Potential (mV)')
axs[0, 1].plot(t, sol[:, 1])
axs[0, 1].set_ylabel('Intracellular Ca2+ (μM)')
axs[1, 0].plot(t, sol[:, 2])
axs[1, 0].set_ylabel('IP3 (μM)')
axs[1, 1].plot(t, sol[:, 3])
axs[1, 1].set_ylabel('h (IP3R gate)')
axs[2, 0].plot(t, sol[:, 4])
axs[2, 0].set_ylabel('ATP')
axs[2, 1].plot(t, [I_stim(ti) for ti in t])
axs[2, 1].set_ylabel('Stimulation Current (pA)')
for ax in axs.flat:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()