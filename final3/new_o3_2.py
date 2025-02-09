import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

###############################################################################
# 1. PHYSICAL CONSTANTS, CELL GEOMETRY, AND ION PARAMETERS (from detailed model)
###############################################################################

# Global physical constants
faraday_constant = 96485       # C/mol
gas_constant = 8314            # J/(mol*K)
temperature = 310              # K

# Cell geometry and capacitance
membrane_capacitance = 0.94    # pF
cell_volume = 2e-12            # L
Vcyto = cell_volume
ER_volume = cell_volume * 0.5   # 50% of cell volume

# Ion valences and concentrations (in mM)
valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1

K_out = 6.26
K_in  = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

###############################################################################
# 2. PARAMETERS FOR IP₃R, PMCA, BUFFERING, MITOCHONDRIAL FLUXES, etc.
###############################################################################

# IP₃R (De Young–Keizer) parameters:
a1 = 400.0; a2 = 0.2; a3 = 400.0; a4 = 0.2; a5 = 20.0
d1 = 0.13; d2 = 1.049; d3 = 0.9434; d4 = 0.1445; d5 = 82.34e-3
v1 = 90.0
b1 = a1 * d1; b2 = a2 * d2; b3 = a3 * d3; b4 = a4 * d4; b5 = a5 * d5

# PMCA parameters
vu = 1540000.0; vm = 2200000.0
aku = 0.303; akmp = 0.14; aru = 1.8; arm = 2.1
p1 = 0.1; p2 = 0.01

# Buffer parameters
Bscyt = 500e-3   # mM (cytosolic)
aKscyt = 0.1
Bser = 2000e-3   # mM (ER)
aKser = 1.0
Bm = 111e-3      # mM
aKm = 0.123

# Other intracellular Ca²⁺ handling parameters
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
k_serca = 0.005       # Maximal SERCA rate
Km_serca = 0.001      # Activation ~1 µM Ca
leak_rate_er = 0.01   # ER leak rate

# IP₃ kinetics parameters
prodip3 = 0.001
V2ip3 = 0.0125
ak2ip3 = 6.0
V3ip3 = 0.09
ak3ip3 = 0.1; ak4ip3 = 1.0

# Mitochondrial parameters
Vmito = Vcyto * 0.08
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV / 1000.0
alphm = 0.2; alphi = 1.0
Vnc = 1.836
aNa = 5000.0; akna = 8000.0; akca = 8.0

###############################################################################
# 3. INITIAL CONDITIONS (Single Cell) AND RNA SCALING FACTORS
###############################################################################

# Initial conditions for the 15 state variables:
# [V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m]
initial_voltage = -70         # mV
initial_calcium = Ca_in        # mM
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.0001
Ca_m_initial = 0.0001

# IP₃R states (normalized)
x000_initial = 0.27
x100_initial = 0.039
x010_initial = 0.29
x001_initial = 0.17
x110_initial = 0.042
x101_initial = 0.0033
x011_initial = 0.18
x111_initial = 0.0035
total = x000_initial + x100_initial + x010_initial + x001_initial + \
        x110_initial + x101_initial + x011_initial + x111_initial
x000_initial /= total; x100_initial /= total; x010_initial /= total; x001_initial /= total
x110_initial /= total; x101_initial /= total; x011_initial /= total; x111_initial /= total

# RNA expression data and scaling factors:
rna_data = {
    'Kir6.1': 15.0,
    'TRPC3': 266.99,
    'IP3R1': 209.82,
    'CaL': 99.48,
    'CaCC': 329.91,
    'Nav': 3.02
}
reference = {
    'Kir6.1': 10.0,
    'TRPC3': 100.0,
    'IP3R': 200.0,
    'CaL': 50.0,
    'CaCC': 100.0,
    'Nav': 3.0
}
translation_efficiency = {
    'Kir6.1': 0.7,
    'TRPC3': 0.5,
    'IP3R': 0.8,
    'CaL': 0.7,
    'CaCC': 0.6,
    'Nav': 0.8
}

scaling_Kir   = (rna_data['Kir6.1'] / reference['Kir6.1']) * translation_efficiency['Kir6.1']
scaling_TRPC3 = (rna_data['TRPC3'] / reference['TRPC3']) * translation_efficiency['TRPC3']
scaling_IP3R1 = (rna_data['IP3R1'] / reference['IP3R']) * translation_efficiency['IP3R']
scaling_CaL   = (rna_data['CaL'] / reference['CaL']) * translation_efficiency['CaL']
scaling_CaCC  = (rna_data['CaCC'] / reference['CaCC']) * translation_efficiency['CaCC']
scaling_Nav   = (rna_data['Nav'] / reference['Nav']) * translation_efficiency['Nav']

###############################################################################
# 4. FUNCTION DEFINITIONS (CHANNEL CURRENTS, FLUXES, ETC.)
###############################################################################

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    E_K = (gas_constant * temperature / (valence_K * faraday_constant)) * np.log(K_out / K_in)
    E_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * np.log(Ca_out / Ca_i)
    E_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * np.log(Na_out / Na_in)
    E_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * np.log(Cl_out / Cl_in)
    return E_K, E_Ca, E_Na, E_Cl

def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    beta_cyt = 1.0 / (1.0 + (Bscyt * aKscyt) / ((aKscyt + Ca_i)**2) + (Bm * aKm) / ((aKm + Ca_i)**2))
    beta_er  = 1.0 / (1.0 + (Bser * aKser) / ((aKser + Ca_ER)**2) + (Bm * aKm) / ((aKm + Ca_ER)**2))
    return beta_cyt, beta_er

def calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111):
    Ca_i = max(Ca_i, 1e-12)
    IP3 = max(IP3, 1e-12)
    f1 = b5 * x010 - a5 * Ca_i * x000
    f2 = b1 * x100 - a1 * IP3 * x000
    f3 = b4 * x001 - a4 * Ca_i * x000
    f4 = b5 * x110 - a5 * Ca_i * x100
    f5 = b2 * x101 - a2 * Ca_i * x100
    f6 = b1 * x110 - a1 * IP3 * x010
    f7 = b4 * x011 - a4 * Ca_i * x010
    f8 = b5 * x011 - a5 * Ca_i * x001
    f9 = b3 * x101 - a3 * IP3 * x001
    f10 = b2 * x111 - a2 * Ca_i * x110
    f11 = b5 * x111 - a5 * Ca_i * x101
    f12 = b3 * x111 - a3 * IP3 * x011
    dx000 = f1 + f2 + f3
    dx100 = f4 + f5 - f2
    dx010 = -f1 + f6 + f7
    dx001 = f8 - f3 + f9
    dx110 = -f4 - f6 + f10
    dx101 = f11 - f9 - f5
    dx011 = -f8 - f7 + f12
    dx111 = -f11 - f12 - f10
    return dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    return v1 * (x110 ** 3) * (Ca_ER - Ca_i)

# --- Modified PMCA flux function ---
def calculate_pmca_flux(Ca_i, dpmca):
    # Clamp Ca_i from below and above to avoid overflow in exponentiation:
    Ca_i = np.clip(Ca_i, 1e-12, 1e-2)
    u4 = (vu * (Ca_i ** aru)) / (Ca_i ** aru + aku ** aru)
    u5 = (vm * (Ca_i ** arm)) / (Ca_i ** arm + akmp ** arm)
    cJpmca = (dpmca * u4 + (1 - dpmca) * u5) / (6.6253e5)
    return cJpmca, u4, u5

def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i = max(Ca_i, 1e-12)
    Ca_m = max(Ca_m, 1e-12)
    bb = (valence_Ca * psi_volts * faraday_constant) / (gas_constant * temperature)
    exp_neg_bb = np.exp(-bb)
    if np.isinf(exp_neg_bb) or np.isnan(exp_neg_bb):
        exp_neg_bb = 0.0
    J_uni = (Pmito / Vmito) * bb * ((alphm * Ca_i * np.exp(-bb) - alphi * Ca_m) / (np.exp(-bb) - 1))
    som = (aNa ** 3) * Ca_m / (akna ** 3 * akca)
    soe = (aNa ** 3) * Ca_i / (akna ** 3 * akca)
    B = np.exp(0.5 * psi_volts * valence_Ca * faraday_constant / (gas_constant * temperature))
    denominator = (1 + (aNa ** 3 / akna ** 3) + Ca_m / akca + som + (aNa ** 3 / akna ** 3) + Ca_i / akca + soe)
    J_nc = Vnc * (B * som - (1 / B) * soe) / denominator
    return J_uni, J_nc

def ip3_ode(Ca_i, IP3):
    Ca_i = max(Ca_i, 1e-12)
    IP3 = max(IP3, 1e-12)
    prod = prodip3  
    deg1 = V2ip3 * IP3 / (ak2ip3 + IP3)
    deg2 = V3ip3 * IP3 / (ak4ip3 + IP3)
    dIP3dt = prod - deg1 - deg2
    # Clamp IP3 between 1e-5 and 0.001 mM:
    if IP3 + dIP3dt < 1e-5:
        dIP3dt = 1e-5 - IP3
    elif IP3 + dIP3dt > 0.001:
        dIP3dt = 0.001 - IP3
    return dIP3dt

# Channel current functions:
def calculate_Kir6_1_current(V, ATP, params):
    E_K = params['reversal_potential_K']
    g_Kir = params['conductance_Kir6.1']
    ATP_half = params.get('ATP_half', 4.0)
    n_ATP = params.get('n_ATP', 2)
    f_ATP = 1.0 / (1.0 + (ATP / ATP_half)**n_ATP)
    return g_Kir * f_ATP * (V - E_K)

def calculate_TRPC3_current(V, params):
    reversal_potential_TRPC3 = 0.0
    base_conductance_TRPC3 = params.get('base_conductance_TRPC3', 0.001)
    scaling_factor = params.get('scaling_TRPC3', 1.0)
    return base_conductance_TRPC3 * scaling_factor * (V - reversal_potential_TRPC3)

def calculate_CaCC_current(V, Ca_i, params):
    E_Cl = params['reversal_potential_Cl']
    base_conductance_CaCC = params.get('base_conductance_CaCC', 0.001)
    scaling_factor = params.get('scaling_CaCC', 1.0)
    g_CaCC = base_conductance_CaCC * scaling_factor
    activation = Ca_i / (Ca_i + 0.001)
    return g_CaCC * activation * (V - E_Cl)

def calculate_Nav1_2_current(V, params):
    E_Na = params['reversal_potential_Na']
    g_Nav = params['conductance_Nav1.2']
    V_half_m = params.get('V_half_m_Nav', -35)
    k_m = params.get('k_m_Nav', 6)
    V_half_h = params.get('V_half_h_Nav', -60)
    k_h = params.get('k_h_Nav', 6)
    m_inf = 1.0 / (1.0 + np.exp(-(V - V_half_m) / k_m))
    h_inf = 1.0 / (1.0 + np.exp((V - V_half_h) / k_h))
    return g_Nav * (m_inf**3) * h_inf * (V - E_Na)

def calculate_currents(V, Ca_i, ATP, dpmca, Ca_ER, x110, IP3, params):
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    
    I_Kir6_1 = calculate_Kir6_1_current(V, ATP, params)
    I_TRPC3 = calculate_TRPC3_current(V, params)
    
    # L-type Ca2+ channel (Cav1.2) gating dynamics:
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_CaL']) / params['activation_slope_CaL']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_CaL']) / params['inactivation_slope_CaL'])) \
            + params['amplitude_factor_CaL'] / (1 + np.exp((params['voltage_shift_CaL'] - V) / params['slope_factor_CaL']))
    I_CaL = params['conductance_CaL'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    
    I_CaCC = calculate_CaCC_current(V, Ca_i, params)
    I_Nav1_2 = calculate_Nav1_2_current(V, params)
    
    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca * valence_Ca * faraday_constant * Vcyto * 1e6
    J_SERCA = params['k_serca'] * (Ca_i ** 2) / (Ca_i ** 2 + params['Km_serca'] ** 2)
    J_ER_leak = params['leak_rate_er'] * (Ca_ER - Ca_i)
    J_IP3R = calculate_ip3r_flux(x110, Ca_ER, Ca_i) * params['conductance_IP3R1']
    J_RyR = params['conductance_RyR'] * (Ca_i / (Ca_i + 0.3)) * (Ca_ER - Ca_i)
    
    return (I_Kir6_1, I_TRPC3, I_CaL, I_CaCC, I_Nav1_2,
            d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR)

###############################################################################
# 5. SINGLE CELL DYNAMICS (15 ODEs) 
###############################################################################

def single_cell_derivs(t, y, params):
    """
    Computes the derivatives for a single cell.
    The state vector y has 15 elements:
      [V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m]
    """
    V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m = y
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
        calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111)
    (I_Kir6_1, I_TRPC3, I_CaL, I_CaCC, I_Nav1_2,
     d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR) = \
         calculate_currents(V, Ca_i, ATP, dpmca, Ca_ER, x110, IP3, params)
    
    # Voltage dynamics (mV/ms)
    dV_dt = -(I_Kir6_1 + I_TRPC3 + I_CaL + I_CaCC + I_Nav1_2) / membrane_capacitance
    
    # Cytosolic calcium dynamics (mM/ms)
    f_TRPC = 0.1
    Ca_influx = (-I_CaL - f_TRPC * I_TRPC3) / (2 * faraday_constant * cell_volume)
    Ca_efflux = -I_PMCA / (2 * faraday_constant * cell_volume)
    dCa_dt = beta_cyt * (Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
    
    # ER calcium dynamics:
    dCa_ER_dt = beta_er * (Vcyto / ER_volume) * (J_SERCA - J_ER_leak - J_IP3R - J_RyR)
    
    # PMCA modulation dynamics (modified to avoid division by very small denominators)
    w1 = p1 * Ca_i
    w2 = p2
    denom = w1 + w2
    if denom < 1e-12:
        denom = 1e-12  # prevent division by near zero
    taom = 1 / denom
    dpmcainf = w2 / denom
    ddpmca_dt = (dpmcainf - dpmca) / taom  # This was the source of overflow warnings
    
    # IP₃ dynamics:
    dIP3_dt = ip3_ode(Ca_i, IP3)
    
    # Mitochondrial calcium dynamics:
    J_uni, J_nc = calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt = J_uni - J_nc
    
    return np.array([dV_dt, dCa_dt, 0, ddpmca_dt, dCa_ER_dt,
                     dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111,
                     dIP3_dt, dCa_m_dt])

###############################################################################
# 6. NUMBA-ACCELERATED COUPLING CALCULATION
###############################################################################

@njit
def compute_coupling(V_all, g_gap, dx, C_m):
    N = V_all.shape[0]
    coupling = np.empty(N)
    for i in range(N):
        if i == 0:
            coupling[i] = g_gap / (dx**2 * C_m) * (V_all[i+1] - V_all[i])
        elif i == N - 1:
            coupling[i] = g_gap / (dx**2 * C_m) * (V_all[i-1] - V_all[i])
        else:
            coupling[i] = g_gap / (dx**2 * C_m) * (V_all[i+1] + V_all[i-1] - 2 * V_all[i])
    return coupling

###############################################################################
# 7. MULTICELL MODEL: COUPLING & CURRENT INJECTION
###############################################################################

def multicell_model(t, Y_flat, params, injection_mask):
    """
    Y_flat: flattened state vector for all cells.
    Each cell has n_states (15) variables.
    injection_mask: an array of length N_cells (1 for cells to receive injection, 0 otherwise)
    """
    N = params['N_cells']
    n_states = params['n_states']  # should be 15
    Y = Y_flat.reshape((N, n_states))
    dY = np.zeros_like(Y)
    
    # Compute gap junction coupling using the numba-accelerated function:
    V_all = Y[:, 0]
    coupling = compute_coupling(V_all, params['g_gap'], params['dx'], membrane_capacitance)
    
    # Compute injection current (only active within the injection time window)
    I_injection = np.zeros(N)
    if params['t_inj_start'] <= t <= params['t_inj_end']:
        I_injection = injection_mask * params['injection_amplitude']
    
    # Loop over each cell and compute its derivatives
    for i in range(N):
        y_cell = Y[i, :]
        dy = single_cell_derivs(t, y_cell, params)
        # Add coupling and subtract injection current (divided by capacitance) from the voltage derivative.
        dy[0] += coupling[i] - I_injection[i] / membrane_capacitance
        dY[i, :] = dy
    return dY.flatten()

###############################################################################
# 8. SET UP MULTICELL SIMULATION PARAMETERS, INJECTION MASK, & INITIAL CONDITIONS
###############################################################################

# For testing, we now use a reduced number of cells and time points.
N_cells = 200           # Total number of cells
n_states = 15          # Each cell has 15 state variables
simulation_duration = 600  # ms
time_points = 50         # Number of time points for evaluation

# Specify stimulation (current injection) cell indices.
# These parameters determine which cells are stimulated.
stim_cell_start = 35   # Starting index (inclusive)
stim_cell_end = 55     # Ending index (inclusive)

# Ensure the indices are within bounds:
stim_cell_start = max(0, stim_cell_start)
stim_cell_end = min(N_cells - 1, stim_cell_end)

# Create the injection mask dynamically
injection_mask = np.zeros(N_cells)
injection_mask[stim_cell_start:stim_cell_end+1] = 1  # +1 because slice end is exclusive

# Global simulation parameters (including stimulation time window)
params = {
    'N_cells': N_cells,
    'n_states': n_states,
    'g_gap': 20.0,      # Gap junction conductance
    'dx': 1.0,          # Spatial step between cells
    't_inj_start': 100, # Injection start time (ms)
    't_inj_end': 400,   # Injection end time (ms)
    'injection_amplitude': -70,  # Injection current amplitude (pA)
    # L-type Ca2+ channel parameters:
    'activation_midpoint_CaL': -40,
    'activation_slope_CaL': 4,
    'inactivation_midpoint_CaL': -45,
    'inactivation_slope_CaL': 5,
    'voltage_shift_CaL': 50,
    'slope_factor_CaL': 20,
    'amplitude_factor_CaL': 0.6,
    'conductance_CaL': 0.00015 * scaling_CaL,
    # TRPC3 parameters:
    'base_conductance_TRPC3': 0.001,
    'scaling_TRPC3': scaling_TRPC3,
    # IP3 receptor and RyR conductances:
    'conductance_IP3R1': 0.1 * scaling_IP3R1,
    'conductance_RyR': 0.01,
    # Nav1.2 parameters:
    'conductance_Nav1.2': 0.005 * scaling_Nav,
    # For Kir6.1:
    'conductance_Kir6.1': 0.025 * scaling_Kir,
    'ATP_half': 4.0,
    'n_ATP': 2,
    # Ca handling parameters:
    'k_serca': k_serca,
    'Km_serca': Km_serca,
    'leak_rate_er': leak_rate_er
}

# Compute reversal potentials and update params.
E_K, E_Ca, E_Na, E_Cl = calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in)
params.update({
    'reversal_potential_K': E_K,
    'reversal_potential_Ca': E_Ca,
    'reversal_potential_Na': E_Na,
    'reversal_potential_Cl': E_Cl
})

# Create the initial state for all cells (each cell is identical initially)
initial_state = np.zeros((N_cells, n_states))
initial_state[:, 0] = initial_voltage  # Membrane potential
initial_state[:, 1] = initial_calcium  # Cytosolic Ca²⁺
initial_state[:, 2] = initial_atp      # ATP
initial_state[:, 3] = initial_dpmca    # PMCA modulation
initial_state[:, 4] = Ca_ER_initial    # ER Ca²⁺
initial_state[:, 5] = x000_initial
initial_state[:, 6] = x100_initial
initial_state[:, 7] = x010_initial
initial_state[:, 8] = x001_initial
initial_state[:, 9] = x110_initial
initial_state[:, 10] = x101_initial
initial_state[:, 11] = x011_initial
initial_state[:, 12] = x111_initial
initial_state[:, 13] = IP3_initial
initial_state[:, 14] = Ca_m_initial

y0 = initial_state.flatten()

###############################################################################
# 9. SOLVE THE COUPLED ODE SYSTEM
###############################################################################

t_span = (0, simulation_duration)
t_eval = np.linspace(t_span[0], t_span[1], time_points)

sol = solve_ivp(fun=lambda t, y: multicell_model(t, y, params, injection_mask),
                t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')

# Reshape solution for easier analysis:
# solution shape: (N_cells, n_states, time_points)
solution = sol.y.reshape((N_cells, n_states, -1))

###############################################################################
# 10. DYNAMIC PLOTTING OF RESULTS
###############################################################################

# Dynamically select up to 5 cell indices to plot (evenly spaced)
num_to_plot = min(5, N_cells)  # Plot at most 5 cells
plot_indices = np.linspace(0, N_cells - 1, num_to_plot, dtype=int)

plt.figure(figsize=(10, 6))
for i in plot_indices:
    plt.plot(sol.t, solution[i, 0, :], label=f'Cell {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential in Selected Cells')
plt.legend()
plt.grid(True)
plt.show()
