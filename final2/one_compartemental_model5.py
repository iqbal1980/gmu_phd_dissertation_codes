import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
F = 96485  # Faraday constant (C/mol)
R = 8314  # Gas constant (J/(mol*K))
T = 310  # Temperature (K)

# Cell parameters
Cm = 11.0  # Membrane capacitance (pF)
V_cell = 2e-12  # Cell volume (L)

# Ion concentrations (mM)
K_o = 4.5  # Extracellular potassium
K_i = 150  # Intracellular potassium
Ca_o = 2.0  # Extracellular calcium
Ca_i = 0.0001  # Intracellular calcium (100 nM)
Na_o = 140  # Extracellular sodium
Na_i = 10  # Intracellular sodium
Cl_o = 120  # Extracellular chloride
Cl_i = 30  # Intracellular chloride

# Channel conductances (nS)
g_Kir61 = 4.0  # Kir6.1 conductance (KATP)
g_TRPC1 = 0.5  # TRPC1 conductance
g_CaCC = 0.3  # CaCC conductance
g_CaL = 0.3  # L-type calcium conductance

# Reversal potentials (mV)
E_K = (R * T / F) * np.log(K_o / K_i)
E_Ca = (R * T / (2 * F)) * np.log(Ca_o / Ca_i)
E_Na = (R * T / F) * np.log(Na_o / Na_i)
E_Cl = -(R * T / F) * np.log(Cl_o / Cl_i)

def model(t, y):
    V, Ca_i, atp = y
    
    # Kir6.1 current (KATP)
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    
    # TRPC1 channel
    I_TRPC1 = g_TRPC1 * (V - E_Ca)
    
    # CaCC channel - chloride-dependent current
    I_CaCC = g_CaCC * (Ca_i / (Ca_i + 0.5)) * (V - E_Cl)
    
    # L-type calcium channel
    d_inf = 1 / (1 + np.exp(-(V + 10) / 6.24))
    f_inf = 1 / (1 + np.exp((V + 35.06) / 8.6)) + 0.6 / (1 + np.exp((50 - V) / 20))
    I_CaL = g_CaL * d_inf * f_inf * (V - E_Ca)
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL) / Cm
    
    # Calcium dynamics
    dCa_dt = -I_CaL / (2 * F * V_cell) - 0.01 * (Ca_i - 0.0001)  # Simple calcium extrusion
    
    # ATP dynamics - simplified model
    datp_dt = 0.01 * (1 - atp) - 0.1 * atp

    return [dV_dt, dCa_dt, datp_dt]

# Simulation parameters
t_span = (0, 1000)
t_eval = np.linspace(0, 1000, 10000)

# Initial conditions
y0 = [-70, 0.0001, 0.5]  # V (mV), Ca_i (mM), atp

# Solve ODE
sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(sol.t, sol.y[0])
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].plot(sol.t, sol.y[1])
axs[1].set_ylabel('Intracellular Ca2+ (mM)')
axs[2].plot(sol.t, sol.y[2])
axs[2].set_ylabel('ATP')
for ax in axs:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()