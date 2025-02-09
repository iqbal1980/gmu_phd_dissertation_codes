#!/usr/bin/env python
"""
Multicell Reaction–Diffusion Model for Pericytes with:
1) Voltage diffusion
2) Calcium diffusion
3) Time-dependent IP3 production

Includes debug prints showing progress of solve_ivp in real time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import time

###############################################################################
# 1. PHYSICAL CONSTANTS, CELL GEOMETRY, AND ION PARAMETERS
###############################################################################
faraday_constant = 96485       # C/mol
gas_constant = 8314            # J/(mol*K)
temperature = 310              # K

# Cell geometry and capacitance
membrane_capacitance = 0.94    # pF
cell_volume = 2e-12            # L
Vcyto = cell_volume
ER_volume = cell_volume * 0.5   # 50% of cell volume

# Ion concentrations (in mM) and valences
valence_K = 1;   valence_Ca = 2;   valence_Na = 1;   valence_Cl = -1
K_out = 6.26;    K_in  = 140
Ca_out = 2.0;    Ca_in = 0.0001
Na_out = 140;    Na_in = 15.38
Cl_out = 110;    Cl_in = 9.65

###############################################################################
# 2. PARAMETERS FROM THE DETAILED SINGLE-CELL MODEL
###############################################################################
# IP3R parameters:
a1 = 400.0;    a2 = 0.2;    a3 = 400.0;    a4 = 0.2;    a5 = 20.0
d1 = 0.13;     d2 = 1.049;  d3 = 0.9434;   d4 = 0.1445; d5 = 82.34e-3
v1 = 90.0
b1 = a1 * d1;  b2 = a2 * d2;  b3 = a3 * d3;  b4 = a4 * d4;  b5 = a5 * d5

# PMCA parameters:
vu = 1540000.0;  vm = 2200000.0
aku = 0.303;     akmp = 0.14;   aru = 1.8;   arm = 2.1
p1 = 0.1;  p2 = 0.01

# Buffer parameters:
Bscyt = 500e-3   # mM (cytosol)
aKscyt = 0.1
Bser = 2000e-3   # mM (ER)
aKser = 1.0
Bm = 111e-3      # mM
aKm = 0.123

# Ca handling:
resting_calcium = 0.001
k_serca = 0.005       # SERCA rate
Km_serca = 0.001      # SERCA activation (~1 µM)
leak_rate_er = 0.01   # ER leak

# IP3 kinetics (we'll keep the same constants, but production will be time-dependent below):
prodip3 = 0.001
V2ip3 = 0.0125
ak2ip3 = 6.0
V3ip3 = 0.09
ak3ip3 = 0.1
ak4ip3 = 1.0

# Mitochondrial parameters:
Vmito = 0.08 * cell_volume
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV / 1000.0
alphm = 0.2; alphi = 1.0
Vnc = 1.836
aNa = 5000.0; akna = 8000.0; akca = 8.0

###############################################################################
# 3. INITIAL CONDITIONS AND SCALING FACTORS
###############################################################################
# Each cell has 15 state variables:
# [V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m]
initial_voltage = -70         # mV
initial_calcium = Ca_in       # mM
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.0001
Ca_m_initial = 0.0001

# IP3R state fractions (normalized)
x000_initial = 0.27;  x100_initial = 0.039;  x010_initial = 0.29;  x001_initial = 0.17
x110_initial = 0.042; x101_initial = 0.0033; x011_initial = 0.18;  x111_initial = 0.0035
total_ip3r = (x000_initial + x100_initial + x010_initial + x001_initial +
              x110_initial + x101_initial + x011_initial + x111_initial)
x000_initial /= total_ip3r; x100_initial /= total_ip3r
x010_initial /= total_ip3r; x001_initial /= total_ip3r
x110_initial /= total_ip3r; x101_initial /= total_ip3r
x011_initial /= total_ip3r; x111_initial /= total_ip3r

# RNA scaling factors:
rna_data = {'Kir6.1': 15.0, 'TRPC3': 266.99, 'IP3R1': 209.82, 'CaL': 99.48, 'CaCC': 329.91, 'Nav': 3.02}
reference = {'Kir6.1': 10.0, 'TRPC3': 100.0, 'IP3R': 200.0, 'CaL': 50.0, 'CaCC': 100.0, 'Nav': 3.0}
translation_efficiency = {'Kir6.1': 0.7, 'TRPC3': 0.5, 'IP3R': 0.8, 'CaL': 0.7, 'CaCC': 0.6, 'Nav': 0.8}

scaling_Kir   = (rna_data['Kir6.1'] / reference['Kir6.1']) * translation_efficiency['Kir6.1']
scaling_TRPC3 = (rna_data['TRPC3'] / reference['TRPC3'])   * translation_efficiency['TRPC3']
scaling_IP3R1 = (rna_data['IP3R1'] / reference['IP3R'])    * translation_efficiency['IP3R']
scaling_CaL   = (rna_data['CaL']    / reference['CaL'])     * translation_efficiency['CaL']
scaling_CaCC  = (rna_data['CaCC']   / reference['CaCC'])    * translation_efficiency['CaCC']
scaling_Nav   = (rna_data['Nav']    / reference['Nav'])     * translation_efficiency['Nav']

###############################################################################
# 4. HELPER FUNCTIONS FOR BUFFERING AND FLUXES
###############################################################################
def calculate_buffering_factors_vec(Ca_i, Ca_ER):
    Ca_i = np.maximum(Ca_i, 1e-12)
    Ca_ER = np.maximum(Ca_ER, 1e-12)
    beta_cyt = 1.0 / (1.0 + (Bscyt * aKscyt)/((aKscyt + Ca_i)**2)
                           + (Bm    * aKm   )/((aKm    + Ca_i)**2))
    beta_er  = 1.0 / (1.0 + (Bser  * aKser )/((aKser  + Ca_ER)**2)
                           + (Bm    * aKm   )/((aKm    + Ca_ER)**2))
    return beta_cyt, beta_er

def calculate_ip3r_states_vec(Ca_i, IP3,
                              x000, x100, x010, x001,
                              x110, x101, x011, x111):
    # Safeguards
    Ca_i = np.maximum(Ca_i, 1e-12)
    IP3  = np.maximum(IP3,  1e-12)
    f1  = b5*x010 - a5*Ca_i*x000
    f2  = b1*x100 - a1*IP3*x000
    f3  = b4*x001 - a4*Ca_i*x000
    f4  = b5*x110 - a5*Ca_i*x100
    f5  = b2*x101 - a2*Ca_i*x100
    f6  = b1*x110 - a1*IP3*x010
    f7  = b4*x011 - a4*Ca_i*x010
    f8  = b5*x011 - a5*Ca_i*x001
    f9  = b3*x101 - a3*IP3*x001
    f10 = b2*x111 - a2*Ca_i*x110
    f11 = b5*x111 - a5*Ca_i*x101
    f12 = b3*x111 - a3*IP3*x011

    dx000 = f1  + f2  + f3
    dx100 = f4  + f5  - f2
    dx010 = -f1 + f6  + f7
    dx001 = f8  - f3  + f9
    dx110 = -f4 - f6  + f10
    dx101 = f11 - f9  - f5
    dx011 = -f8 - f7  + f12
    dx111 = -f11 - f12 - f10
    return (dx000, dx100, dx010, dx001,
            dx110, dx101, dx011, dx111)

def calculate_ip3r_flux_vec(x110, Ca_ER, Ca_i):
    Ca_i  = np.maximum(Ca_i,  1e-12)
    Ca_ER = np.maximum(Ca_ER, 1e-12)
    return v1 * (x110**3) * (Ca_ER - Ca_i)

def calculate_pmca_flux_vec(Ca_i, dpmca):
    Ca_i = np.clip(Ca_i, 1e-12, 1e-2)
    u4 = (vu*(Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5 = (vm*(Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1-dpmca)*u5) / 6.6253e5
    return cJpmca, u4, u5

def calculate_mito_fluxes_vec(Ca_i, Ca_m):
    Ca_i = np.maximum(Ca_i, 1e-12)
    Ca_m = np.maximum(Ca_m, 1e-12)
    bb = (valence_Ca * psi_volts * faraday_constant)/(gas_constant * temperature)
    exp_neg_bb = np.exp(-bb)
    exp_neg_bb = np.where(np.isnan(exp_neg_bb)|np.isinf(exp_neg_bb), 0.0, exp_neg_bb)

    # Mito uniporter
    J_uni = (Pmito / Vmito)*bb*((alphm*Ca_i*np.exp(-bb) - alphi*Ca_m)/(np.exp(-bb) - 1))

    # Na+/Ca2+ exchanger
    som = (aNa**3)*Ca_m/(akna**3 * akca)
    soe = (aNa**3)*Ca_i/(akna**3 * akca)
    B = np.exp(0.5*psi_volts*valence_Ca*faraday_constant/(gas_constant*temperature))
    denominator = (1 + (aNa**3 / akna**3) + Ca_m/akca + som +
                   (aNa**3 / akna**3) + Ca_i/akca + soe)
    J_nc = Vnc * (B*som - (1/B)*soe) / denominator
    return J_uni, J_nc

###############################################################################
# 5. TIME-DEPENDENT IP3 PRODUCTION
###############################################################################
def ip3_ode_vec(Ca_i, IP3, t):
    # Example: a 10x bigger IP3 production from t=100 ms to t=150 ms
    if 100 <= t <= 150:
        prod = 0.01
    else:
        prod = 0.001

    deg1 = V2ip3 * IP3 / (ak2ip3 + IP3)
    deg2 = V3ip3 * IP3 / (ak4ip3 + IP3)
    dIP3dt = prod - deg1 - deg2

    new_IP3 = IP3 + dIP3dt
    dIP3dt = np.where(new_IP3 < 1e-5, 1e-5 - IP3, dIP3dt)
    dIP3dt = np.where(new_IP3 > 0.001, 0.001 - IP3, dIP3dt)
    return dIP3dt

###############################################################################
# 6. MAIN REACTION (SINGLE-CELL) ODEs, BUT VECTORIZED FOR ALL CELLS
###############################################################################
def vectorized_reaction(U, t, params):
    """
    U has shape (N_cells, 15).
    Returns dU/dt in shape (N_cells, 15).
    """
    V    = U[:, 0]
    Ca_i = U[:, 1]
    ATP  = U[:, 2]
    dpmca= U[:, 3]
    Ca_ER= U[:, 4]
    x000 = U[:, 5]
    x100 = U[:, 6]
    x010 = U[:, 7]
    x001 = U[:, 8]
    x110 = U[:, 9]
    x101 = U[:,10]
    x011 = U[:,11]
    x111 = U[:,12]
    IP3  = U[:,13]
    Ca_m = U[:,14]

    # Buffering factors
    beta_cyt, beta_er = calculate_buffering_factors_vec(Ca_i, Ca_ER)

    # IP3R states
    (dx000, dx100, dx010, dx001,
     dx110, dx101, dx011, dx111) = calculate_ip3r_states_vec(
         Ca_i, IP3, x000, x100, x010, x001,
         x110, x101, x011, x111)

    # Channel currents:
    E_K    = params['reversal_potential_K']
    g_Kir  = params['conductance_Kir6.1']
    ATP_half = params.get('ATP_half', 4.0)
    n_ATP   = params.get('n_ATP', 2)
    f_ATP   = 1.0 / (1.0 + (ATP / ATP_half)**n_ATP)
    I_Kir6_1 = g_Kir * f_ATP * (V - E_K)

    base_conductance_TRPC3   = params.get('base_conductance_TRPC3', 0.001)
    scaling_TRPC3_local      = params.get('scaling_TRPC3', 1.0)
    I_TRPC3 = base_conductance_TRPC3 * scaling_TRPC3_local * (V - 0.0)

    d_inf = 1.0 / (1.0 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = (1.0 / (1.0 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL']))
             + params['amplitude_factor_CaL'] / (1.0 + np.exp((params['voltage_shift_CaL'] - V)
                                                             / params['slope_factor_CaL'])))
    I_CaL  = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])

    E_Cl  = params['reversal_potential_Cl']
    base_conductance_CaCC = params.get('base_conductance_CaCC', 0.001)
    scaling_CaCC_local    = params.get('scaling_CaCC', 1.0)
    g_CaCC = base_conductance_CaCC * scaling_CaCC_local
    activation = Ca_i / (Ca_i + 0.001)
    I_CaCC = g_CaCC * activation * (V - E_Cl)

    E_Na  = params['reversal_potential_Na']
    g_Nav = params['conductance_Nav1.2']
    V_half_m = params.get('V_half_m_Nav', -35)
    k_m      = params.get('k_m_Nav', 6)
    V_half_h = params.get('V_half_h_Nav', -60)
    k_h      = params.get('k_h_Nav', 6)
    m_inf = 1.0 / (1.0 + np.exp(-(V - V_half_m)/k_m))
    h_inf = 1.0 / (1.0 + np.exp((V - V_half_h)/k_h))
    I_Nav1_2 = g_Nav * (m_inf**3) * h_inf * (V - E_Na)

    # PMCA
    cJpmca, _, _ = calculate_pmca_flux_vec(Ca_i, dpmca)
    I_PMCA = cJpmca * valence_Ca * faraday_constant * Vcyto * 1e6

    # SERCA, leak, IP3R, RyR
    J_SERCA   = params['k_serca'] * (Ca_i**2)/(Ca_i**2 + params['Km_serca']**2)
    J_ER_leak = params['leak_rate_er']*(Ca_ER - Ca_i)
    J_IP3R    = calculate_ip3r_flux_vec(x110, Ca_ER, Ca_i)*params['conductance_IP3R1']
    J_RyR     = params['conductance_RyR']*(Ca_i/(Ca_i + 0.3))*(Ca_ER - Ca_i)

    # Membrane potential derivative
    dV_dt = -(I_Kir6_1 + I_TRPC3 + I_CaL + I_CaCC + I_Nav1_2)/membrane_capacitance

    # Ca flux from channels
    f_TRPC    = 0.1  # fraction of TRPC current that is Ca current
    Ca_influx = (-I_CaL - f_TRPC*I_TRPC3) / (2*faraday_constant*cell_volume)
    Ca_efflux = -I_PMCA / (2*faraday_constant*cell_volume)
    dCa_dt    = beta_cyt*(Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)

    # ER Ca derivative
    dCa_ER_dt = beta_er*((Vcyto/ER_volume)*(J_SERCA - J_ER_leak - J_IP3R - J_RyR))

    # dpmca gating
    w1 = p1*Ca_i
    w2 = p2
    denom     = np.maximum(w1 + w2, 1e-12)
    taom      = 1.0/denom
    dpmcainf  = w2/denom
    ddpmca_dt = (dpmcainf - dpmca)/taom

    # IP3 derivative
    dIP3_dt = ip3_ode_vec(Ca_i, IP3, t)

    # Mitochondrial Ca
    J_uni, J_nc = calculate_mito_fluxes_vec(Ca_i, Ca_m)
    dCa_m_dt    = J_uni - J_nc

    # Assemble
    dU_dt = np.column_stack([
        dV_dt,
        dCa_dt,
        np.zeros_like(V),  # ATP not changing in this example
        ddpmca_dt,
        dCa_ER_dt,
        dx000, dx100, dx010, dx001,
        dx110, dx101, dx011, dx111,
        dIP3_dt,
        dCa_m_dt
    ])
    return dU_dt

###############################################################################
# 7. SPATIAL LAPLACIAN FOR 1D VOLTAGE + CALCIUM DIFFUSION
###############################################################################
N_cells = 500
dx = 1.0
diagonals = [
    np.ones(N_cells-1),
    -2*np.ones(N_cells),
    np.ones(N_cells-1)
]
offsets = [-1, 0, 1]
L = diags(diagonals, offsets, shape=(N_cells, N_cells)).toarray() / (dx**2)

# Gap-junction coupling for Voltage
g_gap = 20.0
D_V   = g_gap / membrane_capacitance  # "Diffusion coefficient" for V

# Calcium diffusion coefficient:
D_ca  = 0.05  # adjust as needed

###############################################################################
# 8. STIMULATION (INJECTION) PARAMETERS
###############################################################################
stim_cell_start = 140
stim_cell_end   = 360
stim_cell_start = max(0, stim_cell_start)
stim_cell_end   = min(N_cells-1, stim_cell_end)
injection_mask  = np.zeros(N_cells)
injection_mask[stim_cell_start:stim_cell_end+1] = 1
t_inj_start  = 100   # ms
t_inj_end    = 300   # ms
injection_amplitude = -70  # pA

###############################################################################
# 9. GLOBAL PARAMETER DICTIONARY (INCLUDES REVERSAL POTENTIALS)
###############################################################################
def compute_reversal_potentials(K_out, K_in, Ca_out, Ca_in,
                                Na_out, Na_in, Cl_out, Cl_in):
    E_K  = (gas_constant*temperature/(valence_K*faraday_constant))*np.log(K_out/K_in)
    E_Ca = (gas_constant*temperature/(valence_Ca*faraday_constant))*np.log(Ca_out/Ca_in)
    E_Na = (gas_constant*temperature/(valence_Na*faraday_constant))*np.log(Na_out/Na_in)
    E_Cl = (gas_constant*temperature/(valence_Cl*faraday_constant))*np.log(Cl_out/Cl_in)
    return E_K, E_Ca, E_Na, E_Cl

E_K, E_Ca, E_Na, E_Cl = compute_reversal_potentials(
    K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in
)

params = {
    'reversal_potential_K':  E_K,
    'reversal_potential_Ca': E_Ca,
    'reversal_potential_Na': E_Na,
    'reversal_potential_Cl': E_Cl,

    'conductance_Kir6.1': 0.025 * scaling_Kir,
    'ATP_half': 4.0,
    'n_ATP': 2,

    'base_conductance_TRPC3': 0.001,
    'scaling_TRPC3': scaling_TRPC3,

    'activation_midpoint_CaL':  -40,
    'activation_slope_CaL':     4,
    'inactivation_midpoint_CaL':-45,
    'inactivation_slope_CaL':   5,
    'voltage_shift_CaL':        50,
    'slope_factor_CaL':         20,
    'amplitude_factor_CaL':     0.6,
    'conductance_CaL':          0.00015 * scaling_CaL,

    'conductance_IP3R1': 0.1 * scaling_IP3R1,
    'conductance_RyR':   0.01,
    'conductance_Nav1.2':0.005 * scaling_Nav,

    'leak_rate_er': leak_rate_er,
    'k_serca':      k_serca,
    'Km_serca':     Km_serca
}

###############################################################################
# 10. FULL REACTION-DIFFUSION FUNCTION + DEBUG PRINT
###############################################################################
# We'll track real elapsed time for debug prints:
start_time = None
last_debug_print = None
simulation_duration = 600  # ms (declared here so we can see in debug)

def f_full(t, U_flat, params):
    """
    U_flat is shape (N_cells*15,).
    Reshape -> (N_cells, 15) to apply vectorized_reaction,
    then add diffusion for V and Ca,
    plus a debug print that fires ~once per second of real time.
    """
    global last_debug_print, start_time

    # Debug: print progress if at least 1s of real time has elapsed
    current_time = time.time()
    if (current_time - last_debug_print) >= 1.0:
        elapsed = current_time - start_time
        fraction = (t / simulation_duration)*100.0
        print(f"[Debug] t = {t:.2f} ms ({fraction:.1f}% of {simulation_duration} ms),"
              f" elapsed real = {elapsed:.1f} s")
        last_debug_print = current_time

    U = U_flat.reshape((N_cells, 15))
    dU_reac = vectorized_reaction(U, t, params)  # shape (N_cells, 15)

    # Voltage diffusion
    V  = U[:, 0]
    diff_V = D_V * L.dot(V)
    dU_reac[:, 0] += diff_V

    # Calcium diffusion
    Ca = U[:, 1]
    diff_Ca = D_ca * L.dot(Ca)
    dU_reac[:, 1] += diff_Ca

    # Injection current
    I_inj = np.where((t_inj_start <= t) & (t <= t_inj_end),
                     injection_mask*injection_amplitude, 0.0)
    dU_reac[:, 0] -= I_inj / membrane_capacitance

    return dU_reac.flatten()

###############################################################################
# 11. TIME DISCRETIZATION AND SOLVER
###############################################################################
time_points = 1000
t_eval = np.linspace(0, simulation_duration, time_points)

# Build initial state
initial_state = np.zeros((N_cells, 15))
initial_state[:, 0]  = initial_voltage
initial_state[:, 1]  = initial_calcium
initial_state[:, 2]  = initial_atp
initial_state[:, 3]  = initial_dpmca
initial_state[:, 4]  = Ca_ER_initial
initial_state[:, 5]  = x000_initial
initial_state[:, 6]  = x100_initial
initial_state[:, 7]  = x010_initial
initial_state[:, 8]  = x001_initial
initial_state[:, 9]  = x110_initial
initial_state[:,10]  = x101_initial
initial_state[:,11]  = x011_initial
initial_state[:,12]  = x111_initial
initial_state[:,13]  = IP3_initial
initial_state[:,14]  = Ca_m_initial
U0 = initial_state.flatten()

# Initialize debug timers
start_time = time.time()
last_debug_print = start_time

# Solve
sol = solve_ivp(
    lambda t, U: f_full(t, U, params),
    (0, simulation_duration),
    U0,
    t_eval=t_eval,
    method='BDF'
)

# Final debug
end_time = time.time()
print(f"\nSimulation complete in {end_time - start_time:.2f} seconds of real time.\n")

# Reshape solution
U_sol = sol.y.reshape((N_cells, 15, -1))

###############################################################################
# 12. PLOTTING
###############################################################################
num_to_plot = min(5, N_cells)
plot_indices = np.linspace(0, N_cells - 1, num_to_plot, dtype=int)

plt.figure(figsize=(10,6))
for i in plot_indices:
    plt.plot(sol.t, U_sol[i, 0, :], label=f'Cell {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential in Selected Cells (With Ca Diffusion & IP3 Pulse)')
plt.legend()
plt.grid(True)
plt.show()

# Also plot Ca
plt.figure(figsize=(10,6))
for i in plot_indices:
    plt.plot(sol.t, U_sol[i, 1, :], label=f'Cell {i} Ca_i')
plt.xlabel('Time (ms)')
plt.ylabel('Ca_i (mM)')
plt.title('Intracellular Ca in Selected Cells')
plt.legend()
plt.grid(True)
plt.show()
