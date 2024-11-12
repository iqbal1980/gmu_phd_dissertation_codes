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
g_Kir61 = 4.0  # Kir6.1 conductance
g_TRPC1 = 0.5  # TRPC1 conductance
g_CaCC = 0.3  # CaCC conductance
g_ASIC2 = 0.2  # ASIC2 conductance

# Reversal potentials (mV)
E_K = (R * T / F) * np.log(K_o / K_i)
E_Ca = (R * T / (2 * F)) * np.log(Ca_o / Ca_i)
E_Na = (R * T / F) * np.log(Na_o / Na_i)
E_Cl = -(R * T / F) * np.log(Cl_o / Cl_i)

# def I_stim(t):
#     return 10 if 200 <= t <= 700 else 0

def model(t, y):
    V, atp = y
    
    # Kir6.1 current (KATP)
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    
    # TRPC1 channel - simplified linear model
    I_TRPC1 = g_TRPC1 * (V - E_Ca)
    
    # CaCC channel - simplified model
    I_CaCC = g_CaCC * (V - E_Cl)
    
    # ASIC2 channel - simplified model, assuming pH-dependent activation
    pH_activation = 0.5  # This should be a function of extracellular pH
    I_ASIC2 = g_ASIC2 * pH_activation * (V - E_Na)
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_ASIC2) / Cm  # Removed I_stim(t)
    
    # ATP dynamics - simplified model
    datp_dt = 0.01 * (1 - atp) - 0.1 * atp

    return [dV_dt, datp_dt]

# Simulation parameters
t_span = (0, 1000)
t_eval = np.linspace(0, 1000, 10000)

# Initial conditions
y0 = [-70, 0.5]  # V (mV), atp

# Solve ODE
sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(sol.t, sol.y[0])
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].plot(sol.t, sol.y[1])
axs[1].set_ylabel('ATP')
for ax in axs:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()