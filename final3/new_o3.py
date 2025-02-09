import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##########################################
# Physical constants and cell geometry
faraday_constant = 96485       # C/mol
gas_constant = 8314            # J/(mol*K)
temperature = 310              # K

membrane_capacitance = 0.94    # pF
cell_volume = 2e-12            # L
Vcyto = cell_volume
# Increase ER volume from 20% to 50% of cell volume
ER_volume = cell_volume * 0.5

##########################################
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

##########################################
# De Young-Keizer IP3R1 Parameters (for detailed IP3 flux)
a1 = 400.0; a2 = 0.2; a3 = 400.0; a4 = 0.2; a5 = 20.0
d1 = 0.13; d2 = 1.049; d3 = 0.9434; d4 = 0.1445; d5 = 82.34e-3
v1 = 90.0
b1 = a1 * d1; b2 = a2 * d2; b3 = a3 * d3; b4 = a4 * d4; b5 = a5 * d5

##########################################
# PMCA parameters
vu = 1540000.0; vm = 2200000.0
aku = 0.303; akmp = 0.14; aru = 1.8; arm = 2.1
p1 = 0.1; p2 = 0.01

##########################################
# Buffer parameters
# Increase cytosolic buffer capacity from 225 µM to 500 µM (i.e. 0.5 mM)
Bscyt = 500e-3   # mM
aKscyt = 0.1
# Convert ER buffer from µM to mM (if needed)
Bser = 2000e-3   # mM
aKser = 1.0
Bm = 111e-3      # mM
aKm = 0.123

##########################################
# Other intracellular Ca2+ handling parameters
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
# Adjust SERCA and leak so that ER Ca remains near 0.5 mM:
k_serca = 0.005       # Reduced maximal SERCA rate
Km_serca = 0.001      # SERCA active at ~1 µM Ca
leak_rate_er = 0.01   # Increased leak

##########################################
# IP3 kinetics parameters (adjusted for a lower resting IP₃)
prodip3 = 0.001
V2ip3 = 0.0125
ak2ip3 = 6.0
V3ip3 = 0.09
ak3ip3 = 0.1; ak4ip3 = 1.0

##########################################
# Mitochondrial parameters
Vmito = Vcyto * 0.08
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV / 1000.0
alphm = 0.2; alphi = 1.0
Vnc = 1.836
aNa = 5000.0; akna = 8000.0; akca = 8.0

##########################################
# Time simulation parameters
simulation_duration = 500   # ms
time_points = 1000

##########################################
# Initial conditions for the state vector (15 states)
# y = [V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m]
initial_voltage = -70         # mV
initial_calcium = Ca_in        # mM
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.0001           # mM
Ca_m_initial = 0.0001         # mM

# IP3R1 states (normalized)
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

##########################################
# RNA Expression Data and Scaling Factors
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

scaling_Kir = (rna_data['Kir6.1'] / reference['Kir6.1']) * translation_efficiency['Kir6.1']
scaling_TRPC3 = (rna_data['TRPC3'] / reference['TRPC3']) * translation_efficiency['TRPC3']
scaling_IP3R1 = (rna_data['IP3R1'] / reference['IP3R']) * translation_efficiency['IP3R']
scaling_CaL = (rna_data['CaL'] / reference['CaL']) * translation_efficiency['CaL']
scaling_CaCC = (rna_data['CaCC'] / reference['CaCC']) * translation_efficiency['CaCC']
scaling_Nav = (rna_data['Nav'] / reference['Nav']) * translation_efficiency['Nav']

##########################################
# Functions for reversal potentials, buffering, and channel/flux kinetics
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

def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i = max(Ca_i, 1e-12)
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

##########################################
# Channel current functions
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
    
    # L-type Ca2+ (Cav1.2) gating
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

##########################################
# Full ODE model (15 states)
def model(t, y, params):
    # y = [V, Ca_i, ATP, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3, Ca_m]
    (V, Ca_i, ATP, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m) = y
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
         calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111)
    
    (I_Kir6_1, I_TRPC3, I_CaL, I_CaCC, I_Nav1_2,
     d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, J_IP3R, J_RyR) = \
         calculate_currents(V, Ca_i, ATP, dpmca, Ca_ER, x110, IP3, params)
    
    # Membrane potential dynamics (pA/pF gives mV/ms):
    dV_dt = -(I_Kir6_1 + I_TRPC3 + I_CaL + I_CaCC + I_Nav1_2) / membrane_capacitance
    
    # Calcium dynamics (mM/ms):
    # Influx via L-type channels (plus 10% of TRPC3) and efflux via PMCA.
    f_TRPC = 0.1
    Ca_influx = (-I_CaL - f_TRPC * I_TRPC3) / (2 * faraday_constant * cell_volume)
    Ca_efflux = -I_PMCA / (2 * faraday_constant * cell_volume)
    dCa_dt = beta_cyt * (Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
    
    dCa_ER_dt = beta_er * (Vcyto / ER_volume) * (J_SERCA - J_ER_leak - J_IP3R - J_RyR)
    
    # PMCA modulation dynamics:
    w1 = p1 * Ca_i
    w2 = p2
    taom = 1 / (w1 + w2)
    dpmcainf = w2 / (w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca) / taom
    
    dIP3_dt = ip3_ode(Ca_i, IP3)
    
    # Mitochondrial Ca²⁺ dynamics:
    J_uni, J_nc = calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt = J_uni - J_nc
    
    return [dV_dt, dCa_dt, 0, ddpmca_dt, dCa_ER_dt,
            dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111,
            dIP3_dt, dCa_m_dt]

##########################################
# Run simulation
def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'],
          params['initial_calcium'],
          params['initial_atp'],
          params['initial_dpmca'],
          params['Ca_ER_initial'],
          x000_initial, x100_initial, x010_initial, x001_initial, x110_initial,
          x101_initial, x011_initial, x111_initial,
          IP3_initial,
          Ca_m_initial]
    # Use LSODA for faster and adaptive integration.
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA', args=(params,), dense_output=True)
    return sol

##########################################
# Plotting and CSV export functions
def plot_and_export(sol, params):
    t = sol.t
    V = sol.y[0]
    Ca_i = sol.y[1]
    Ca_ER = sol.y[4]
    IP3 = sol.y[13]
    dpmca = sol.y[3]
    Ca_m = sol.y[14]
    
    # Figure 1: Cellular States
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 15))
    axs1[0, 0].plot(t, V, 'b')
    axs1[0, 0].set_title('Membrane Potential (mV)')
    axs1[0, 1].plot(t, Ca_i, 'orange')
    axs1[0, 1].set_title('Cytosolic Ca²⁺ (mM)')
    axs1[1, 0].plot(t, Ca_ER, 'red')
    axs1[1, 0].set_title('ER Ca²⁺ (mM)')
    axs1[1, 1].plot(t, IP3, 'magenta')
    axs1[1, 1].set_title('IP₃ (mM)')
    axs1[2, 0].plot(t, dpmca, 'cyan')
    axs1[2, 0].set_title('PMCA Modulation')
    axs1[2, 1].plot(t, Ca_m, 'brown')
    axs1[2, 1].set_title('Mitochondrial Ca²⁺ (mM)')
    for ax in axs1.flat:
        ax.grid(True)
    fig1.tight_layout()
    plt.show()
    
    # Figure 2: Currents and ER Fluxes
    n_points = len(t)
    I_Kir6_1_arr = np.zeros(n_points)
    I_TRPC3_arr = np.zeros(n_points)
    I_CaL_arr = np.zeros(n_points)
    I_CaCC_arr = np.zeros(n_points)
    I_Nav1_2_arr = np.zeros(n_points)
    I_PMCA_arr = np.zeros(n_points)
    J_SERCA_arr = np.zeros(n_points)
    J_ER_leak_arr = np.zeros(n_points)
    J_IP3R_arr = np.zeros(n_points)
    J_RyR_arr = np.zeros(n_points)
    d_inf_arr = np.zeros(n_points)
    f_inf_arr = np.zeros(n_points)
    
    for i in range(n_points):
        current_V = V[i]
        current_Ca_i = max(Ca_i[i], 1e-12)
        current_Ca_ER = max(Ca_ER[i], 1e-12)
        current_IP3 = IP3[i]
        current_dpmca = dpmca[i]
        # x110 is the 10th state (index 9)
        current_x110 = sol.y[9][i]
        (I_Kir6_1_arr[i], I_TRPC3_arr[i], I_CaL_arr[i], I_CaCC_arr[i],
         I_Nav1_2_arr[i], d_inf_arr[i], f_inf_arr[i], I_PMCA_arr[i],
         J_SERCA_arr[i], J_ER_leak_arr[i], J_IP3R_arr[i], J_RyR_arr[i]) = \
             calculate_currents(current_V, current_Ca_i, params['initial_atp'],
                                current_dpmca, current_Ca_ER, current_x110, current_IP3, params)
    
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 15))
    axs2[0, 0].plot(t, I_Kir6_1_arr, label='Kir6.1')
    axs2[0, 0].plot(t, I_TRPC3_arr, label='TRPC3')
    axs2[0, 0].set_title('Kir6.1 & TRPC3 Currents (pA)')
    axs2[0, 0].legend(); axs2[0, 0].grid(True)
    axs2[0, 1].plot(t, I_CaL_arr, 'g', label='Cav1.2')
    axs2[0, 1].set_title('Cav1.2 Current (pA)')
    axs2[0, 1].legend(); axs2[0, 1].grid(True)
    axs2[1, 0].plot(t, I_CaCC_arr, 'm', label='CaCC')
    axs2[1, 0].set_title('CaCC Current (pA)')
    axs2[1, 0].legend(); axs2[1, 0].grid(True)
    axs2[1, 1].plot(t, I_Nav1_2_arr, 'r', label='Nav1.2')
    axs2[1, 1].set_title('Nav1.2 Current (pA)')
    axs2[1, 1].legend(); axs2[1, 1].grid(True)
    axs2[2, 0].plot(t, I_PMCA_arr, 'c', label='I_PMCA')
    axs2[2, 0].set_title('PMCA Current (pA)')
    axs2[2, 0].legend(); axs2[2, 0].grid(True)
    axs2[2, 1].plot(t, J_SERCA_arr, 'k', label='J_SERCA')
    axs2[2, 1].plot(t, J_ER_leak_arr, 'y', label='J_ER_leak')
    axs2[2, 1].plot(t, J_IP3R_arr, 'b', label='J_IP3R')
    axs2[2, 1].plot(t, J_RyR_arr, 'orange', label='J_RyR')
    axs2[2, 1].set_title('ER Fluxes (mM/ms)')
    axs2[2, 1].legend(); axs2[2, 1].grid(True)
    fig2.tight_layout()
    plt.show()
    
    # Export states every 10 ms
    t_csv = np.arange(0, params['simulation_duration'] + 1e-9, 10)
    states_interp = sol.sol(t_csv)
    # Columns: time, V, Ca_i, Ca_ER, IP3, dpmca, Ca_m
    states_data = np.column_stack((t_csv, states_interp[0, :], states_interp[1, :],
                                     states_interp[4, :], states_interp[13, :],
                                     states_interp[3, :], states_interp[14, :]))
    header_states = "time,V,Ca_i,Ca_ER,IP3,dpmca,Ca_m"
    np.savetxt("states_output.csv", states_data, delimiter=",", header=header_states, comments="")
    
    # Compute currents every 10 ms:
    n_csv = len(t_csv)
    I_Kir6_1_csv = np.zeros(n_csv)
    I_TRPC3_csv = np.zeros(n_csv)
    I_CaL_csv = np.zeros(n_csv)
    I_CaCC_csv = np.zeros(n_csv)
    I_Nav1_2_csv = np.zeros(n_csv)
    I_PMCA_csv = np.zeros(n_csv)
    J_SERCA_csv = np.zeros(n_csv)
    J_ER_leak_csv = np.zeros(n_csv)
    J_IP3R_csv = np.zeros(n_csv)
    J_RyR_csv = np.zeros(n_csv)
    d_inf_csv = np.zeros(n_csv)
    f_inf_csv = np.zeros(n_csv)
    
    states_for_currents = sol.sol(t_csv)
    for i in range(n_csv):
        V_i = states_for_currents[0, i]
        Ca_i_i = max(states_for_currents[1, i], 1e-12)
        Ca_ER_i = max(states_for_currents[4, i], 1e-12)
        IP3_i = states_for_currents[13, i]
        dpmca_i = states_for_currents[3, i]
        x110_i = states_for_currents[9, i]
        (I_Kir6_1_csv[i], I_TRPC3_csv[i], I_CaL_csv[i], I_CaCC_csv[i],
         I_Nav1_2_csv[i], d_inf_csv[i], f_inf_csv[i], I_PMCA_csv[i],
         J_SERCA_csv[i], J_ER_leak_csv[i], J_IP3R_csv[i], J_RyR_csv[i]) = \
             calculate_currents(V_i, Ca_i_i, params['initial_atp'],
                                dpmca_i, Ca_ER_i, x110_i, IP3_i, params)
    
    currents_data = np.column_stack((t_csv, I_Kir6_1_csv, I_TRPC3_csv, I_CaL_csv,
                                       I_CaCC_csv, I_Nav1_2_csv, I_PMCA_csv,
                                       J_SERCA_csv, J_ER_leak_csv, J_IP3R_csv,
                                       J_RyR_csv, d_inf_csv, f_inf_csv))
    header_currents = ("time,I_Kir6_1,I_TRPC3,I_CaL,I_CaCC,I_Nav1_2,"
                         "I_PMCA,J_SERCA,J_ER_leak,J_IP3R,J_RyR,d_inf,f_inf")
    np.savetxt("currents_output.csv", currents_data, delimiter=",", header=header_currents, comments="")

##########################################
# Parameter dictionary setup
params = {
    'K_out': K_out, 'K_in': K_in, 'Ca_out': Ca_out, 'Ca_in': Ca_in,
    'Na_out': Na_out, 'Na_in': Na_in, 'Cl_out': Cl_out, 'Cl_in': Cl_in,
    
    # Scale channel conductances with RNA data:
    'conductance_Kir6.1': 0.025 * scaling_Kir,  # ATP-sensitive K+ channel
    'base_conductance_TRPC3': 0.001,
    'scaling_TRPC3': scaling_TRPC3,
    # Use a further reduced L-type Ca2+ conductance:
    'conductance_CaL': 0.00015 * scaling_CaL,
    'base_conductance_CaCC': 0.001,
    'scaling_CaCC': scaling_CaCC,
    'conductance_IP3R1': 0.1 * scaling_IP3R1,
    'conductance_RyR': 0.01,
    'conductance_Nav1.2': 0.005 * scaling_Nav,
    
    'k_serca': k_serca, 'Km_serca': Km_serca,
    'leak_rate_er': leak_rate_er, 'k_ncx': 0.0,
    'calcium_extrusion_rate': calcium_extrusion_rate,
    'resting_calcium': resting_calcium,
    
    'prodip3': prodip3, 'V2ip3': V2ip3, 'ak2ip3': ak2ip3,
    'V3ip3': V3ip3, 'ak3ip3': ak3ip3, 'ak4ip3': ak4ip3,
    
    'vu': vu, 'vm': vm, 'aku': aku, 'akmp': akmp, 'aru': aru, 'arm': arm,
    'initial_dpmca': initial_dpmca,
    
    # Voltage‐dependent parameters for L‐type Ca2+ channels:
    'activation_midpoint_CaL': -40,
    'activation_slope_CaL': 4,
    'inactivation_midpoint_CaL': -45,
    'inactivation_slope_CaL': 5,
    'voltage_shift_CaL': 50,
    'slope_factor_CaL': 20,
    'amplitude_factor_CaL': 0.6,
    
    # Additional parameters for Nav1.2 gating:
    'V_half_m_Nav': -35, 'k_m_Nav': 6,
    'V_half_h_Nav': -60, 'k_h_Nav': 6,
    
    # For Kir6.1 ATP sensitivity:
    'ATP_half': 4.0, 'n_ATP': 2,
    
    'simulation_duration': simulation_duration,
    'time_points': time_points,
    'initial_voltage': initial_voltage,
    'initial_calcium': initial_calcium,
    'initial_atp': initial_atp,
    'Ca_ER_initial': Ca_ER_initial
}

# Calculate reversal potentials and update parameters.
E_K, E_Ca, E_Na, E_Cl = calculate_reversal_potentials(params['K_out'], params['K_in'],
                                                       params['Ca_out'], params['Ca_in'],
                                                       params['Na_out'], params['Na_in'],
                                                       params['Cl_out'], params['Cl_in'])
params.update({
    'reversal_potential_K': E_K,
    'reversal_potential_Ca': E_Ca,
    'reversal_potential_Na': E_Na,
    'reversal_potential_Cl': E_Cl
})

##########################################
# Run simulation and export plots/CSV
sol = run_simulation(params)
plot_and_export(sol, params)
