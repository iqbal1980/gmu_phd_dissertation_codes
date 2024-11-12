import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Physical constants
F = 96485  # Faraday constant (C/mol)
R = 8314  # Gas constant (J/(mol*K))
T = 310  # Temperature (K)

# Cell parameters
Cm = 20e-12  # Membrane capacitance (F)
Vcyto = 1e-15  # Cytoplasmic volume (L)
Ver = 0.1 * Vcyto  # ER volume (L)
Vmito = 0.08 * Vcyto  # Mitochondrial volume (L)

# Ion concentrations (mM)
K_out, K_in = 5.4, 140
Ca_out, Ca_in = 2.0, 0.0001
Na_out, Na_in = 140, 10

# Conductances (nS)
g_Kir61 = 0.5
g_Kir22 = 0.3  # Added Kir2.2
g_TRPC1 = 0.005
g_TRPC3 = 0.01  # Added TRPC3
g_CaL = 0.05
g_CaCC = 0.05
g_leak = 0.01

# Other parameters
k_SERCA = 0.05
Km_SERCA = 0.0003
k_leak_er = 0.0002
IP3_conc = 0.0001
k_IP3R = 0.05
k_RyR = 0.01
k_NCX = 0.001
k_PMCA = 0.02  # Added PMCA
k_buffer = 1000
B_total = 0.1
k_NaK = 0.05  # Na+/K+ ATPase rate
k_mito_in = 0.001  # Mitochondrial Ca2+ uptake rate
k_mito_out = 0.0001  # Mitochondrial Ca2+ release rate

# GPCR (A2A receptor) parameters
k_A2A = 0.01  # A2A receptor activation rate
A2A_conc = 0  # Initial A2A agonist concentration

def nernst_potential(z, c_out, c_in):
    return (R * T / (z * F)) * np.log(c_out / c_in)

E_K = nernst_potential(1, K_out, K_in)
E_Ca = nernst_potential(2, Ca_out, Ca_in)
E_Na = nernst_potential(1, Na_out, Na_in)

def calculate_currents_and_fluxes(V, Ca_i, Ca_ER, Ca_mito, m, h, A2A_act):
    # Ion channel currents
    I_Kir61 = g_Kir61 * (V - E_K) * 1e-9
    I_Kir22 = g_Kir22 * (V - E_K) * 1e-9  # Added Kir2.2 current
    I_TRPC1 = g_TRPC1 * (V - E_Ca) * 1e-9
    I_TRPC3 = g_TRPC3 * (V - E_Ca) * 1e-9 * A2A_act  # TRPC3 modulated by A2A activation
    I_CaL = g_CaL * m * h * (V - E_Ca) * 1e-9
    I_CaCC = g_CaCC * (Ca_i / (Ca_i + 0.0005)) * (V - E_K) * 1e-9
    I_leak = g_leak * (V - E_K) * 1e-9

    # Pump and exchanger currents
    I_NaK = k_NaK * (Na_in / (Na_in + 10)) * (K_out / (K_out + 1.5)) * 1e-9
    I_PMCA = k_PMCA * (Ca_i**2 / (Ca_i**2 + 0.0001**2)) * 1e-9  # Added PMCA current
    I_NCX = k_NCX * (Na_in**3 / (Na_in**3 + 87.5**3)) * (Ca_out / (Ca_out + 1)) * 1e-12

    # Intracellular Ca2+ fluxes
    J_SERCA = k_SERCA * (Ca_i**2 / (Ca_i**2 + Km_SERCA**2))
    J_ER_leak = k_leak_er * (Ca_ER - Ca_i)
    J_IP3R = k_IP3R * (IP3_conc / (IP3_conc + 0.0001)) * (Ca_i / (Ca_i + 0.0002)) * (Ca_ER - Ca_i)
    J_RyR = k_RyR * (Ca_i**2 / (Ca_i**2 + 0.0003**2)) * (Ca_ER - Ca_i)
    J_mito_in = k_mito_in * Ca_i  # Mitochondrial Ca2+ uptake
    J_mito_out = k_mito_out * Ca_mito  # Mitochondrial Ca2+ release

    return I_Kir61, I_Kir22, I_TRPC1, I_TRPC3, I_CaL, I_CaCC, I_leak, I_NCX, I_NaK, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR, J_mito_in, J_mito_out

def model(t, y):
    V, Ca_i, Ca_ER, Ca_mito, m, h, B, A2A_act = y
    
    I_Kir61, I_Kir22, I_TRPC1, I_TRPC3, I_CaL, I_CaCC, I_leak, I_NCX, I_NaK, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR, J_mito_in, J_mito_out = calculate_currents_and_fluxes(V, Ca_i, Ca_ER, Ca_mito, m, h, A2A_act)
    
    dV_dt = -(I_Kir61 + I_Kir22 + I_TRPC1 + I_TRPC3 + I_CaL + I_CaCC + I_leak + I_NCX + I_NaK + I_PMCA) / Cm
    
    dCa_i_dt = (
        (-I_CaL - I_TRPC1 - I_TRPC3 - I_PMCA + 2*I_NCX) / (2 * F * Vcyto)
        + (J_IP3R + J_RyR - J_SERCA) / Vcyto
        + J_ER_leak
        - J_mito_in + J_mito_out * (Vmito / Vcyto)
        - k_buffer * (Ca_i * (B_total - B) - (0.0001 * B))
    )
    
    dCa_ER_dt = (-J_IP3R - J_RyR + J_SERCA) / Ver - J_ER_leak * (Vcyto / Ver)
    
    dCa_mito_dt = (J_mito_in - J_mito_out) * (Vcyto / Vmito)
    
    m_inf = 1 / (1 + np.exp(-(V + 20) / 5))
    tau_m = 1 / (0.1 * np.exp((V + 20) / 20) + 0.1 * np.exp(-(V + 20) / 20))
    dm_dt = (m_inf - m) / tau_m

    h_inf = 1 / (1 + np.exp((V + 30) / 5)) * (1 / (1 + (Ca_i / 0.0005)**2))
    tau_h = 20
    dh_dt = (h_inf - h) / tau_h

    dB_dt = k_buffer * (Ca_i * (B_total - B) - (0.0001 * B))

    dA2A_act_dt = k_A2A * (A2A_conc - A2A_act)  # A2A receptor activation

    return [dV_dt, dCa_i_dt, dCa_ER_dt, dCa_mito_dt, dm_dt, dh_dt, dB_dt, dA2A_act_dt]

t_span = (0, 5000)  # Simulation time: 5000 ms
y0 = [-70, 0.0001, 0.5, 0.0001, 0.05, 0.95, 0.08, 0]  # Initial conditions

# Run simulation
sol = solve_ivp(model, t_span, y0, method='LSODA', dense_output=True, max_step=1)

t = sol.t
V = sol.y[0]
Ca_i = sol.y[1]
Ca_ER = sol.y[2]
Ca_mito = sol.y[3]
m = sol.y[4]
h = sol.y[5]
B = sol.y[6]
A2A_act = sol.y[7]

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 25))

axs[0].plot(t, V)
axs[0].set_ylabel('Membrane Potential (mV)')
axs[0].set_title('Pericyte Membrane Potential')

axs[1].plot(t, Ca_i * 1000)
axs[1].set_ylabel('Cytosolic Ca2+ (μM)')
axs[1].set_title('Cytosolic Calcium Concentration')
axs[1].set_yscale('log')

axs[2].plot(t, Ca_ER)
axs[2].set_ylabel('ER Ca2+ (mM)')
axs[2].set_title('ER Calcium Concentration')

axs[3].plot(t, Ca_mito)
axs[3].set_ylabel('Mitochondrial Ca2+ (mM)')
axs[3].set_title('Mitochondrial Calcium Concentration')

axs[4].plot(t, m, label='m (activation)')
axs[4].plot(t, h, label='h (inactivation)')
axs[4].set_ylabel('Gating variables')
axs[4].set_title('L-type Ca Channel Gating')
axs[4].legend()

for ax in axs:
    ax.set_xlabel('Time (ms)')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Calculate currents and fluxes
I_Kir61, I_Kir22, I_TRPC1, I_TRPC3, I_CaL, I_CaCC, I_leak, I_NCX, I_NaK, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR, J_mito_in, J_mito_out = zip(*[calculate_currents_and_fluxes(v, ca_i, ca_er, ca_mito, m_val, h_val, a2a) for v, ca_i, ca_er, ca_mito, m_val, h_val, a2a in zip(V, Ca_i, Ca_ER, Ca_mito, m, h, A2A_act)])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(t, I_Kir61, label='I_Kir61')
ax1.plot(t, I_Kir22, label='I_Kir22')
ax1.plot(t, I_TRPC1, label='I_TRPC1')
ax1.plot(t, I_TRPC3, label='I_TRPC3')
ax1.plot(t, I_CaL, label='I_CaL')
ax1.plot(t, I_CaCC, label='I_CaCC')
ax1.plot(t, I_leak, label='I_leak')
ax1.plot(t, I_NCX, label='I_NCX')
ax1.plot(t, I_NaK, label='I_NaK')
ax1.plot(t, I_PMCA, label='I_PMCA')
ax1.set_ylabel('Current (A)')
ax1.set_title('Membrane Currents')
ax1.legend()
ax1.grid(True)

ax2.plot(t, J_SERCA, label='J_SERCA')
ax2.plot(t, J_ER_leak, label='J_ER_leak')
ax2.plot(t, J_IP3R, label='J_IP3R')
ax2.plot(t, J_RyR, label='J_RyR')
ax2.plot(t, J_mito_in, label='J_mito_in')
ax2.plot(t, J_mito_out, label='J_mito_out')
ax2.set_ylabel('Flux (mol/s)')
ax2.set_title('Calcium Fluxes')
ax2.legend()
ax2.grid(True)

for ax in (ax1, ax2):
    ax.set_xlabel('Time (ms)')

plt.tight_layout()
plt.show()