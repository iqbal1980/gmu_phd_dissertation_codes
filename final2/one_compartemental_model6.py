import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
F = 96485  # Faraday constant (C/mol) - Standard value
R = 8314  # Gas constant (J/(mol*K)) - Standard value
T = 310  # Temperature (K) - Normal body temperature

# Cell parameters
Cm = 0.94  # Membrane capacitance (pF) - Found value: 0.94 pF from Simulation Params.docx, page 1 &#8203;:citation[oaicite:28]{index=28}&#8203;
V_cell = 2e-12  # Cell volume (L) - Typical small cell volume, estimated

# Ion concentrations (mM)
K_o = 4.5  # Extracellular potassium - Normal physiological level, merged_output.pdf, page 10 &#8203;:citation[oaicite:27]{index=27}&#8203;
K_i = 150  # Intracellular potassium - Typical intracellular concentration, merged_output.pdf, page 10 &#8203;:citation[oaicite:26]{index=26}&#8203;
Ca_o = 2.0  # Extracellular calcium - Normal physiological level, merged_output.pdf, page 10 &#8203;:citation[oaicite:25]{index=25}&#8203;
Ca_i = 0.0001  # Intracellular calcium (100 nM) - Resting intracellular concentration, merged_output.pdf, page 10 &#8203;:citation[oaicite:24]{index=24}&#8203;
Na_o = 140  # Extracellular sodium - Normal physiological level, merged_output.pdf, page 10 &#8203;:citation[oaicite:23]{index=23}&#8203;
Na_i = 10  # Intracellular sodium - Typical intracellular concentration, merged_output.pdf, page 10 &#8203;:citation[oaicite:22]{index=22}&#8203;
Cl_o = 120  # Extracellular chloride - Normal physiological level, merged_output.pdf, page 10 &#8203;:citation[oaicite:21]{index=21}&#8203;
Cl_i = 30  # Intracellular chloride - Typical intracellular concentration, merged_output.pdf, page 10 &#8203;:citation[oaicite:20]{index=20}&#8203;

# Channel conductances (nS)
g_Kir61 = 0.025  # Kir6.1 conductance (KATP) - Adjusted to 0.025 nS based on Simulation Params.docx, page 3 &#8203;:citation[oaicite:19]{index=19}&#8203;
g_TRPC1 = 0.017  # TRPC1 conductance - Found value from Simulation Params.docx, page 3 &#8203;:citation[oaicite:18]{index=18}&#8203;
g_CaCC = 0.3  # CaCC conductance - Estimated value
g_CaL = 0.025  # L-type calcium conductance - Adjusted to 0.025 nS based on Cav1.2 value from Simulation Params.docx, page 3 &#8203;:citation[oaicite:17]{index=17}&#8203;

# Reversal potentials (mV)
E_K = (R * T / F) * np.log(K_o / K_i)  # Nernst equation for potassium, standard formula &#8203;:citation[oaicite:16]{index=16}&#8203;
E_Ca = (R * T / (2 * F)) * np.log(Ca_o / Ca_i)  # Nernst equation for calcium, divalent ion, standard formula &#8203;:citation[oaicite:15]{index=15}&#8203;
E_Na = (R * T / F) * np.log(Na_o / Na_i)  # Nernst equation for sodium, standard formula &#8203;:citation[oaicite:14]{index=14}&#8203;
E_Cl = -(R * T / F) * np.log(Cl_o / Cl_i)  # Nernst equation for chloride, standard formula &#8203;:citation[oaicite:13]{index=13}&#8203;

# Gating variable parameters for L-type calcium channels
V_half_d = -10  # Half-activation voltage for d_inf (mV)
k_d = 6.24  # Slope factor for d_inf (mV)
V_half_f = -35.06  # Half-inactivation voltage for f_inf (mV)
k_f = 8.6  # Slope factor for f_inf (mV)
V_s = 50  # Voltage shift for second component of f_inf (mV)
k_s = 20  # Slope factor for second component of f_inf (mV)
a_f = 0.6  # Amplitude factor for second component of f_inf

# Calcium extrusion rate
k_Ca = 0.01  # Rate constant for calcium extrusion (1/ms)
Ca_rest = 0.0001  # Resting intracellular calcium concentration (mM)

def model(t, y):
    V, Ca_i, atp = y
    
    # Kir6.1 current (KATP)
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    # Justification: Kir6.1 is ATP-sensitive and typically contributes to K+ current regulation in cells, Simulation Params.docx, page 3 &#8203;:citation[oaicite:12]{index=12}&#8203;
    
    # TRPC1 channel
    I_TRPC1 = g_TRPC1 * (V - E_Ca)
    # Justification: TRPC1 is a non-selective cation channel contributing to Ca2+ entry, Simulation Params.docx, page 3 &#8203;:citation[oaicite:11]{index=11}&#8203;
    
    # CaCC channel - chloride-dependent current
    I_CaCC = g_CaCC * (Ca_i / (Ca_i + 0.5)) * (V - E_Cl)
    # Justification: CaCC channels are activated by intracellular Ca2+ levels, estimated value
    
    # L-type calcium channel
    d_inf = 1 / (1 + np.exp(-(V - V_half_d) / k_d))  # Gating variable, merged_output.pdf, page 15 &#8203;:citation[oaicite:10]{index=10}&#8203;
    f_inf = 1 / (1 + np.exp((V - V_half_f) / k_f)) + a_f / (1 + np.exp((V_s - V) / k_s))  # Gating variable, merged_output.pdf, page 15 &#8203;:citation[oaicite:9]{index=9}&#8203;
    I_CaL = g_CaL * d_inf * f_inf * (V - E_Ca)
    # Justification: Standard gating variables for L-type calcium channels, merged_output.pdf, page 15 &#8203;:citation[oaicite:8]{index=8}&#8203;
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL) / Cm
    # Justification: Sum of ionic currents through the membrane determines the rate of change of membrane potential, Simulation Params.docx, page 1 &#8203;:citation[oaicite:7]{index=7}&#8203;
    
    # Calcium dynamics
    dCa_dt = -I_CaL / (2 * F * V_cell) - k_Ca * (Ca_i - Ca_rest)  # Simple calcium extrusion
    # Justification: Ca2+ influx through L-type channels and a simple extrusion model
    
    # ATP dynamics - simplified model
    datp_dt = 0.01 * (1 - atp) - 0.1 * atp
    # Justification: Simplified model for ATP consumption and regeneration

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
