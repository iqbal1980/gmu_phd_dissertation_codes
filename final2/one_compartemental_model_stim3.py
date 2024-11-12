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
V_er = V_cell * 0.1  # ER volume (10% of cell volume)

# Ion concentrations (mM) - from Fortran code
K_o = 5.4  # Extracellular potassium
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

# IP3R parameters - from Fortran code
v1 = 90.0
c1 = 0.185  # Ratio of ER to cytosol volume
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
    V, Ca_i, IP3, h, atp, x000, x100, x010, x001, x110, x101, x011, x111, Ca_er = y
    
    # Kir6.1 current (KATP) - adapted based on ode_system2.py Kir current
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    
    # Kir2.2 current - adapted based on ode_system2.py Kir current
    I_Kir22 = 0*( g_Kir22 * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6)) )
    
    # L-type Ca channel (Ca_v1.2) - adapted from Fortran code
    d_inf = 1 / (1 + np.exp(-(V + 10) / 6.24))
    f_inf = 1 / (1 + np.exp((V + 35.06) / 8.6)) + 0.6 / (1 + np.exp((50 - V) / 20))
    I_CaL = g_CaL * d_inf * f_inf * (V - E_Ca)
    

 
    
    # TRPC channel - simplified linear model based on ode_system2.py
    I_TRPC = g_TRPC * (V - E_Ca)
    
    # IP3R flux - adapted from Fortran code
    J_IP3R = 0*( c1 * v1 * (x110**3) * (Ca_er - Ca_i))
    
    print("I_Kir61="+str(I_Kir61))
    print("I_Kir22="+str(I_Kir22))
    print("I_CaL="+str(I_CaL))
    print("I_TRPC="+str(I_TRPC))

    
    # IP3R state transitions
    f1 = a5 * Ca_i * x010 - d5 * x000
    f2 = a1 * IP3 * x000 - d1 * x100
    f3 = a4 * Ca_i * x000 - d4 * x001
    f4 = a5 * Ca_i * x100 - d5 * x110
    f5 = a2 * Ca_i * x100 - d2 * x101
    f6 = a1 * IP3 * x010 - d1 * x110
    f7 = a4 * Ca_i * x010 - d4 * x011
    f8 = a5 * Ca_i * x001 - d5 * x011
    f9 = a3 * IP3 * x001 - d3 * x101
    f10 = a2 * Ca_i * x110 - d2 * x111
    f11 = a5 * Ca_i * x101 - d5 * x111
    f12 = a3 * IP3 * x011 - d3 * x111

    dx000_dt = -(f1 + f2 + f3)
    dx100_dt = f4 + f5 - f2
    dx010_dt = -f1 + f6 + f7
    dx001_dt = f8 - f3 + f9
    dx110_dt = -f4 - f6 + f10
    dx101_dt = f11 - f9 - f5
    dx011_dt = -f8 - f7 + f12
    dx111_dt = -f11 - f12 - f10
    
    # Calcium dynamics
    dCa_dt = -I_CaL / (2 * F * V_cell) + J_IP3R / V_cell
    dCa_er_dt = -J_IP3R / V_er
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_Kir22 - I_CaL - I_TRPC + I_stim(t)) / Cm
    
    # IP3 dynamics - simplified model
    dIP3_dt = 0.1 * (1 - IP3) - 0.1 * IP3
    
    # ATP dynamics - simplified model
    datp_dt = 0.01 * (1 - atp) - 0.1 * atp

    # h gate for IP3R (kept for compatibility, not used in new IP3R model)
    dh_dt = 0

    return [dV_dt, dCa_dt, dIP3_dt, dh_dt, datp_dt, dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt, dCa_er_dt]

# Simulation parameters
t = np.linspace(0, 1000, 10000)

# Initial conditions
y0 = [-70, 0.1, 0.1, 0.5, 0.5, 0.27, 0.039, 0.29, 0.17, 0.042, 0.0033, 0.18, 0.0035, 335.5]

# Solve ODE
sol = odeint(model, y0, t)

# Plot results
fig, axs = plt.subplots(4, 2, figsize=(12, 20))
axs[0, 0].plot(t, sol[:, 0])
axs[0, 0].set_ylabel('Membrane Potential (mV)')
axs[0, 1].plot(t, sol[:, 1])
axs[0, 1].set_ylabel('Intracellular Ca2+ (μM)')
axs[1, 0].plot(t, sol[:, 2])
axs[1, 0].set_ylabel('IP3 (μM)')
axs[1, 1].plot(t, sol[:, 4])
axs[1, 1].set_ylabel('ATP')
axs[2, 0].plot(t, sol[:, -1])
axs[2, 0].set_ylabel('ER Ca2+ (μM)')
axs[2, 1].plot(t, sol[:, 5:13])
axs[2, 1].set_ylabel('IP3R States')
axs[2, 1].legend(['x000', 'x100', 'x010', 'x001', 'x110', 'x101', 'x011', 'x111'], loc='best')
axs[3, 0].plot(t, [I_stim(ti) for ti in t])
axs[3, 0].set_ylabel('Stimulation Current (pA)')
axs[3, 1].plot(t, sol[:, 3])
axs[3, 1].set_ylabel('h (IP3R gate, unused)')
for ax in axs.flat:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()