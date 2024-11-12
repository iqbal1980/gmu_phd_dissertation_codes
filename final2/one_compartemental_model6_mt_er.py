import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
F = 96485  # Faraday constant (C/mol) - Standard value
R = 8314  # Gas constant (J/(mol*K)) - Standard value
T = 310  # Temperature (K) - Normal body temperature

# Cell parameters
Cm = 0.94  # Membrane capacitance (pF) - Found value: 0.94 pF from Simulation Params.docx, page 1 【18†Simulation Params.docx】
V_cell = 2e-12  # Cell volume (L) - Typical small cell volume, estimated

# Ion concentrations (mM)
K_o = 4.5  # Extracellular potassium - Normal physiological level, merged_output.pdf, page 10 【32:10†merged_output.pdf】
K_i = 150  # Intracellular potassium - Typical intracellular concentration, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Ca_o = 2.0  # Extracellular calcium - Normal physiological level, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Ca_i = 0.0001  # Intracellular calcium (100 nM) - Resting intracellular concentration, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Na_o = 140  # Extracellular sodium - Normal physiological level, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Na_i = 10  # Intracellular sodium - Typical intracellular concentration, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Cl_o = 120  # Extracellular chloride - Normal physiological level, merged_output.pdf, page 10 【32:10†merged_output.pdf】
Cl_i = 30  # Intracellular chloride - Typical intracellular concentration, merged_output.pdf, page 10 【32:10†merged_output.pdf】

# Channel conductances (nS)
g_Kir61 = 0.025  # Kir6.1 conductance (KATP) - Adjusted to 0.025 nS based on Simulation Params.docx, page 3 【18†Simulation Params.docx】
g_TRPC1 = 0.017  # TRPC1 conductance - Found value from Simulation Params.docx, page 3 【18†Simulation Params.docx】
g_CaCC = 0.3  # CaCC conductance - Estimated value
g_CaL = 0.025  # L-type calcium conductance - Adjusted to 0.025 nS based on Cav1.2 value from Simulation Params.docx, page 3 【18†Simulation Params.docx】

# Reversal potentials (mV)
E_K = (R * T / F) * np.log(K_o / K_i)  # Nernst equation for potassium, standard formula 【32:0†merged_output.pdf】
E_Ca = (R * T / (2 * F)) * np.log(Ca_o / Ca_i)  # Nernst equation for calcium, divalent ion, standard formula 【32:0†merged_output.pdf】
E_Na = (R * T / F) * np.log(Na_o / Na_i)  # Nernst equation for sodium, standard formula 【32:0†merged_output.pdf】
E_Cl = -(R * T / F) * np.log(Cl_o / Cl_i)  # Nernst equation for chloride, standard formula 【32:0†merged_output.pdf】

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

# Mitochondrial parameters
V_mito = 1e-15  # Mitochondrial volume (L)
Ca_mito_init = 0.0001  # Initial mitochondrial calcium concentration (mM)
k_mito_in = 0.01  # Rate of mitochondrial calcium uptake (1/ms)
k_mito_out = 0.01  # Rate of mitochondrial calcium release (1/ms)
ATP_prod_rate = 0.05  # ATP production rate by mitochondria (mM/ms)

# ER parameters
V_ER = 2e-12  # ER volume (L)
Ca_ER_init = 0.1  # Initial ER calcium concentration (mM)
k_SERCA = 0.01  # Rate of SERCA pump calcium uptake (1/ms)
k_IP3R = 0.01  # Rate of IP3 receptor calcium release (1/ms)
k_RYR = 0.01  # Rate of ryanodine receptor calcium release (1/ms)

def mitochondrial_dynamics(Ca_i, Ca_mito):
    Ca_mito_influx = k_mito_in * (Ca_i - Ca_mito)
    Ca_mito_efflux = k_mito_out * (Ca_mito - Ca_i)
    return Ca_mito_influx - Ca_mito_efflux

def atp_production():
    return ATP_prod_rate

def er_dynamics(Ca_i, Ca_ER):
    Ca_ER_influx = k_SERCA * (Ca_i - Ca_ER)
    Ca_ER_efflux = k_IP3R * (Ca_ER - Ca_i) + k_RYR * (Ca_ER - Ca_i)
    return Ca_ER_influx - Ca_ER_efflux

def model(t, y):
    V, Ca_i, atp, Ca_mito, Ca_ER = y
    
    # Kir6.1 current (KATP)
    I_Kir61 = g_Kir61 * atp * np.sqrt(K_o / 5.4) * (V - E_K) / (1 + np.exp((V - E_K - 15) / 6))
    # Justification: Kir6.1 is ATP-sensitive and typically contributes to K+ current regulation in cells, Simulation Params.docx, page 3 【18†Simulation Params.docx】
    
    # TRPC1 channel
    I_TRPC1 = g_TRPC1 * (V - E_Ca)
    # Justification: TRPC1 is a non-selective cation channel contributing to Ca2+ entry, Simulation Params.docx, page 3 【18†Simulation Params.docx】
    
    # CaCC channel - chloride-dependent current
    I_CaCC = g_CaCC * (Ca_i / (Ca_i + 0.5)) * (V - E_Cl)
    # Justification: CaCC channels are activated by intracellular Ca2+ levels, estimated value
    
    # L-type calcium channel
    d_inf = 1 / (1 + np.exp(-(V - V_half_d) / k_d))  # Gating variable, merged_output.pdf, page 15 【32:15†merged_output.pdf】
    f_inf = 1 / (1 + np.exp((V - V_half_f) / k_f)) + a_f / (1 + np.exp((V_s - V) / k_s))  # Gating variable, merged_output.pdf, page 15 【32:15†merged_output.pdf】
    I_CaL = g_CaL * d_inf * f_inf * (V - E_Ca)
    # Justification: Standard gating variables for L-type calcium channels, merged_output.pdf, page 15 【32:15†merged_output.pdf】
    
    # Membrane potential
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL) / Cm
    # Justification: Sum of ionic currents through the membrane determines the rate of change of membrane potential, Simulation Params.docx, page 1 【18†Simulation Params.docx】
    
    # Calcium dynamics
    dCa_dt = -I_CaL / (2 * F * V_cell) - k_Ca * (Ca_i - Ca_rest) + mitochondrial_dynamics(Ca_i, Ca_mito) + er_dynamics(Ca_i, Ca_ER)
    # Justification: Ca2+ influx through L-type channels and a simple extrusion model
    
    # Mitochondrial calcium dynamics
    dCa_mito_dt = mitochondrial_dynamics(Ca_i, Ca_mito)
    
    # ER calcium dynamics
    dCa_ER_dt = er_dynamics(Ca_i, Ca_ER)
    
    # ATP dynamics - simplified model
    datp_dt = atp_production() - 0.1 * atp
    # Justification: Simplified model for ATP consumption and regeneration

    return [dV_dt, dCa_dt, datp_dt, dCa_mito_dt, dCa_ER_dt]

# Simulation parameters
t_span = (0, 1000)
t_eval = np.linspace(0, 1000, 10000)

# Initial conditions
y0 = [-70, 0.0001, 0.5, 0.0001, 0.1]  # V (mV), Ca_i (mM), atp, Ca_mito, Ca_ER

# Solve ODE
sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')

# Plot results
fig, axs = plt.subplots(5, 1, figsize=(10, 25))
axs[0].plot(sol.t, sol.y[0])
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].plot(sol.t, sol.y[1])
axs[1].set_ylabel('Intracellular Ca2+ (mM)')
axs[2].plot(sol.t, sol.y[2])
axs[2].set_ylabel('ATP')
axs[3].plot(sol.t, sol.y[3])
axs[3].set_ylabel('Mitochondrial Ca2+ (mM)')
axs[4].plot(sol.t, sol.y[4])
axs[4].set_ylabel('ER Ca2+ (mM)')
for ax in axs:
    ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
