import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Numba imports
from numba import jit, njit
# We’ll use `@jit` with `forceobj=True` on the multi_cell_model 
# to handle the dictionary in object mode, 
# and we can try `@njit` for the single-cell ODE if we remove dictionary usage.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


###############################################################################
#                            SINGLE-CELL MODEL
###############################################################################
faraday_constant = 96485   # C/mol
gas_constant = 8314        # J/(mol*K)
temperature = 310          # K

membrane_capacitance = 0.94  # pF
cell_volume = 2e-12          # L
Vcyto = cell_volume
ER_volume = cell_volume*0.2

valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2  # Ca2+

K_out = 6.26
K_in = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

a1, a2, a3, a4, a5 = 400.0, 0.2, 400.0, 0.2, 20.0
d1, d2, d3, d4, d5 = 0.13, 1.049, 0.9434, 0.1445, 82.34e-3
v1 = 90.0
b1 = a1*d1
b2 = a2*d2
b3 = a3*d3
b4 = a4*d4
b5 = a5*d5

vu, vm = 1540000.0, 2200000.0
aku, akmp = 0.303, 0.14
aru, arm = 1.8, 2.1
p1, p2 = 0.1, 0.01

Bscyt, aKscyt = 225.0, 0.1
Bser, aKser = 2000.0, 1.0
Bm, aKm = 111.0, 0.123

conductance_Kir61 = 0.025
conductance_TRPC1 = 0.001
conductance_CaL = 0.0005
conductance_CaCC = 0.001
conductance_leak = 0.01
conductance_IP3R1 = 0.1
conductance_IP3R2 = 0.05
conductance_RyR = 0.01
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
calcium_activation_threshold_CaCC = 0.0005
k_serca = 0.1
Km_serca = 0.5
leak_rate_er = 0.05
k_ncx = 0.001

prodip3 = 0.01
V2ip3 = 12.5
ak2ip3 = 6.0
V3ip3 = 0.9
ak3ip3 = 0.1
ak4ip3 = 1.0

Vmito = 2e-12 * 0.08
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV/1000.0
alphm = 0.2
alphi = 1.0
Vnc = 1.836
aNa = 5000.0
akna = 8000.0
akca = 8.0

simulation_duration = 100
time_points = 1000

initial_voltage = -70
initial_calcium = Ca_in
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.1
Ca_m_initial = 0.0001

x000_initial = 0.27
x100_initial = 0.039
x010_initial = 0.29
x001_initial = 0.17
x110_initial = 0.042
x101_initial = 0.0033
x011_initial = 0.18
x111_initial = 0.0035
total1 = (x000_initial + x100_initial + x010_initial + x001_initial +
          x110_initial + x101_initial + x011_initial + x111_initial)
x000_initial /= total1
x100_initial /= total1
x010_initial /= total1
x001_initial /= total1
x110_initial /= total1
x101_initial /= total1
x011_initial /= total1
x111_initial /= total1

x0002_initial = 0.27
x1002_initial = 0.039
x0102_initial = 0.29
x0012_initial = 0.17
x1102_initial = 0.042
x1012_initial = 0.0033
x0112_initial = 0.18
x1112_initial = 0.0035
total2 = (x0002_initial + x1002_initial + x0102_initial + x0012_initial +
          x1102_initial + x1012_initial + x0112_initial + x1112_initial)
x0002_initial /= total2
x1002_initial /= total2
x0102_initial /= total2
x0012_initial /= total2
x1102_initial /= total2
x1012_initial /= total2
x0112_initial /= total2
x1112_initial /= total2


# For best performance, we'd put these helpers in nopython mode too.
# But they rely on many global variables and dictionary usage. We'll attempt object mode.
@jit(forceobj=True)
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    revK = (gas_constant*temperature/(valence_K*faraday_constant))*np.log(K_out/K_in)
    revCa = (gas_constant*temperature/(valence_Ca*faraday_constant))*np.log(Ca_out/Ca_i)
    revNa = (gas_constant*temperature/(valence_Na*faraday_constant))*np.log(Na_out/Na_in)
    revCl = (gas_constant*temperature/(valence_Cl*faraday_constant))*np.log(Cl_out/Cl_in)
    return revK, revCa, revNa, revCl

@jit(forceobj=True)
def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    beta_cyt = 1.0/(1.0 + (Bscyt*aKscyt)/((aKscyt+Ca_i)**2) + (Bm*aKm)/((aKm+Ca_i)**2))
    beta_er  = 1.0/(1.0 + (Bser*aKser)/((aKser+Ca_ER)**2) + (Bm*aKm)/((aKm+Ca_ER)**2))
    return beta_cyt, beta_er

@jit(forceobj=True)
def calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111):
    Ca_i = max(Ca_i,1e-12)
    IP3  = max(IP3,1e-12)
    
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
    
    dx000 = f1 + f2 + f3
    dx100 = f4 + f5 - f2
    dx010 = -f1 + f6 + f7
    dx001 = f8 - f3 + f9
    dx110 = -f4 - f6 + f10
    dx101 = f11 - f9 - f5
    dx011 = -f8 - f7 + f12
    dx111 = -f11 - f12 - f10
    
    return dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111

@jit(forceobj=True)
def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i  = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    return v1*(x110**3)*(Ca_ER - Ca_i)

@jit(forceobj=True)
def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i = max(Ca_i,1e-12)
    u4 = (vu*(Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5 = (vm*(Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1.0 - dpmca)*u5) / 6.6253e5
    return cJpmca, u4, u5

@jit(forceobj=True)
def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i = max(Ca_i,1e-12)
    Ca_m = max(Ca_m,1e-12)
    bb = (z*psi_volts*faraday_constant)/(gas_constant*temperature)
    exp_neg_bb = np.exp(-bb)
    if np.isinf(exp_neg_bb) or np.isnan(exp_neg_bb):
        exp_neg_bb = 0.0
    J_uni = (Pmito/Vmito)*bb*((alphm*Ca_i*np.exp(-bb) - alphi*Ca_m)/(exp_neg_bb - 1))
    
    som = (aNa**3)*Ca_m/(akna**3*akca)
    soe = (aNa**3)*Ca_i/(akna**3*akca)
    B = np.exp(0.5*psi_volts*z*faraday_constant/(gas_constant*temperature))
    denominator = (1 + (aNa**3/(akna**3)) + Ca_m/akca + som +
                   (aNa**3/(akna**3)) + Ca_i/akca + soe)
    J_nc = Vnc*(B*som - (1.0/B)*soe)/denominator
    return J_uni, J_nc

@jit(forceobj=True)
def ip3_ode(Ca_i, IP3):
    Ca_i = max(Ca_i,1e-12)
    IP3  = max(IP3,1e-12)
    term1 = prodip3
    term2 = V2ip3 / (1.0 + (ak2ip3/IP3))
    term3 = (V3ip3 / (1.0 + (ak3ip3/IP3))) * (1.0/(1.0 + (ak4ip3/(Ca_i))))
    dIP3dt = term1 - term2 - term3
    new_IP3 = IP3 + dIP3dt
    if new_IP3 > 0.01:
        dIP3dt = (0.01 - IP3)
    if new_IP3 < 1e-12:
        dIP3dt = (1e-12 - IP3)
    return dIP3dt

@jit(forceobj=True)
def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params):
    Ca_i  = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    d_inf = 1.0/(1.0 + np.exp(-(V-params['activation_midpoint_CaL'])/params['activation_slope_CaL']))
    f_inf = (1.0/(1.0 + np.exp((V - params['inactivation_midpoint_CaL'])/params['inactivation_slope_CaL']))
             + params['amplitude_factor_CaL']/(1.0 + np.exp((params['voltage_shift_CaL'] - V)/params['slope_factor_CaL'])))
    
    I_Kir61 = (params['conductance_Kir61']*atp*np.sqrt(params['K_out']/params['reference_K'])
               * (V - params['reversal_potential_K'])
               / (1.0 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61'])
                               / params['voltage_slope_Kir61'])))
    
    I_TRPC1 = params['conductance_TRPC1']*(V - params['reversal_potential_Ca'])
    I_CaCC  = (params['conductance_CaCC']*(Ca_i/(Ca_i + params['calcium_activation_threshold_CaCC']))
               * (V - params['reversal_potential_Cl']))
    I_CaL   = params['conductance_CaL']*d_inf*f_inf*(V - params['reversal_potential_Ca'])
    I_leak  = params['conductance_leak']*(V - params['reversal_potential_K'])
    
    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca*z*faraday_constant*Vcyto*1e6
    
    J_SERCA = params['k_serca']*(Ca_i**2)/(Ca_i**2 + params['Km_serca']**2)
    J_ER_leak = params['leak_rate_er']*(Ca_ER - Ca_i)
    I_NCX = (params['k_ncx']
             * (params['Na_in']**3/(params['Na_in']**3+87.5**3))
             * (params['Ca_out']/(params['Ca_out']+1.0)))
    
    J_IP3R1 = calculate_ip3r_flux(x110, Ca_ER, Ca_i)*params['conductance_IP3R1']
    J_IP3R2 = calculate_ip3r_flux(x1102, Ca_ER, Ca_i)*params['conductance_IP3R2']
    J_IP3R = J_IP3R1 + J_IP3R2
    J_RyR = params['conductance_RyR']*(Ca_i/(Ca_i+0.3))*(Ca_ER - Ca_i)
    
    return (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf,
            I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)

# The single-cell ODE also reads from a dictionary => we must do object mode if we keep that approach
@jit(forceobj=True)
def model(t, y, params):
    (V, Ca_i, atp, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m,
     x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112) = y
    
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    
    dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
        calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111)
    dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, dx1112 = \
        calculate_ip3r_states(Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)
    
    (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf,
     I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR) = \
        calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params)
    
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX)/membrane_capacitance
    
    Ca_influx = -I_CaL/(2.0*z*faraday_constant*cell_volume)
    Ca_efflux = (-I_PMCA - I_NCX)/(2.0*z*faraday_constant*cell_volume)
    
    dCa_dt = beta_cyt*(Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
    dCa_ER_dt = beta_er*((Vcyto/ER_volume)*(J_SERCA - J_ER_leak - J_IP3R - J_RyR))
    
    w1 = p1*Ca_i
    w2 = p2
    taom = 1.0/(w1 + w2)
    dpmcainf = w2/(w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca)/taom
    
    dIP3_dt = ip3_ode(Ca_i, IP3)
    
    J_uni, J_nc = calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt = J_uni - J_nc
    
    return [
        dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
        dx000, dx100, dx010, dx001, x110, x101, x011, x111,
        dIP3_dt, dCa_m_dt,
        dx0002, dx1002, dx0102, x0012, x1102, x1012, x0112, x1112
    ]


def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [
        params['initial_voltage'],
        params['initial_calcium'],
        params['initial_atp'],
        params['initial_dpmca'],
        params['Ca_ER_initial'],
        x000_initial, x100_initial, x010_initial, x001_initial, x110_initial,
        x101_initial, x011_initial, x111_initial,
        IP3_initial,
        Ca_m_initial,
        x0002_initial, x1002_initial, x0102_initial, x0012_initial,
        x1102_initial, x1012_initial, x0112_initial, x1112_initial
    ]
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
    return sol

###############################################################################
# Plotting is mostly unaffected by Numba, so we leave it alone
###############################################################################
def plot_results_with_two_figures(sol, params):
    ...
    # (Same as your original plotting code, omitted for brevity)
    ...


###############################################################################
# MULTI-CELL WRAPPER WITH GAP-JUNCTIONS + STIM, JIT-compiled in object mode
###############################################################################
@jit(forceobj=True)
def multi_cell_model(t, Y, params_multi):
    """
    We'll do object mode because of the dictionary usage in `params_multi`.
    """
    Ng       = params_multi['Ng']
    nvar     = params_multi['nvar']
    g_gap    = params_multi['g_gap']
    dx       = params_multi['dx']
    cm       = params_multi['cm']
    cellparams = params_multi['cell_params']

    I_app_val  = params_multi.get('I_app_val', 0.0)
    stim_cell  = params_multi.get('stim_cell', 0)
    stim_start = params_multi.get('stim_start', 100.0)
    stim_end   = params_multi.get('stim_end', 400.0)

    dYdt = np.zeros_like(Y)
    
    for i in range(Ng):
        idx_start = i * nvar
        idx_end   = idx_start + nvar
        
        y_i = Y[idx_start:idx_end]
        
        dyi = model(t, y_i, cellparams)  # single-cell derivative

        dVdt_single = dyi[0]
        V_i = y_i[0]
        
        if i == 0:
            V_left = V_i
        else:
            V_left = Y[(i-1)*nvar + 0]
        
        if i == Ng-1:
            V_right = V_i
        else:
            V_right = Y[(i+1)*nvar + 0]
        
        I_gj = (g_gap / (dx**2)) * (V_left + V_right - 2.0*V_i)
        dVdt_multi = dVdt_single + (I_gj / cm)

        if (stim_start <= t <= stim_end) and (i == stim_cell):
            dVdt_multi += (I_app_val / cm)
        
        dyi[0] = dVdt_multi
        dYdt[idx_start:idx_end] = dyi
    
    return dYdt


def run_multi_cell_simulation(params_multi, t_span=(0,600), dt=0.1):
    Ng    = params_multi['Ng']
    nvar  = params_multi['nvar']
    t_eval = np.arange(t_span[0], t_span[1]+dt, dt)

    single_cell_ic = [
        params_multi['cell_params']['initial_voltage'],
        params_multi['cell_params']['initial_calcium'],
        params_multi['cell_params']['initial_atp'],
        params_multi['cell_params']['initial_dpmca'],
        params_multi['cell_params']['Ca_ER_initial'],
        x000_initial, x100_initial, x010_initial, x001_initial, x110_initial,
        x101_initial, x011_initial, x111_initial,
        IP3_initial,
        Ca_m_initial,
        x0002_initial, x1002_initial, x0102_initial, x0012_initial,
        x1102_initial, x1012_initial, x0112_initial, x1112_initial
    ]
    Y0 = np.tile(single_cell_ic, Ng)
    
    sol = solve_ivp(
        lambda t, Y: multi_cell_model(t, Y, params_multi),
        (t_span[0], t_span[1]),
        Y0, t_eval=t_eval, method='RK45'
    )
    return sol

###############################################################################
# EXAMPLE MAIN
###############################################################################
if __name__ == "__main__":
    # 1) Prepare single-cell params
    params_cell = {
        'K_out': K_out,
        'K_in': K_in,
        'Ca_out': Ca_out,
        'Ca_in': Ca_in,
        'Na_out': Na_out,
        'Na_in': Na_in,
        'Cl_out': Cl_out,
        'Cl_in': Cl_in,
        'conductance_Kir61': conductance_Kir61,
        'conductance_TRPC1': conductance_TRPC1,
        'conductance_CaCC': conductance_CaCC,
        'conductance_CaL': conductance_CaL,
        'conductance_leak': conductance_leak,
        'conductance_IP3R1': conductance_IP3R1,
        'conductance_IP3R2': conductance_IP3R2,
        'conductance_RyR': conductance_RyR,
        'k_serca': k_serca,
        'Km_serca': Km_serca,
        'leak_rate_er': leak_rate_er,
        'k_ncx': k_ncx,
        'calcium_extrusion_rate': calcium_extrusion_rate,
        'resting_calcium': resting_calcium,
        'calcium_activation_threshold_CaCC': calcium_activation_threshold_CaCC,
        'reference_K': 5.4,
        'voltage_slope_Kir61': 6,
        'voltage_shift_Kir61': 15,
        'activation_midpoint_CaL': -40,
        'activation_slope_CaL': 4,
        'inactivation_midpoint_CaL': -45,
        'inactivation_slope_CaL': 5,
        'voltage_shift_CaL': 50,
        'slope_factor_CaL': 20,
        'amplitude_factor_CaL': 0.6,
        'vu': vu,
        'vm': vm,
        'aku': aku,
        'akmp': akmp,
        'aru': aru,
        'arm': arm,
        'initial_dpmca': initial_dpmca,
        'simulation_duration': simulation_duration,
        'time_points': time_points,
        'initial_voltage': initial_voltage,
        'initial_calcium': initial_calcium,
        'initial_atp': initial_atp,
        'Ca_ER_initial': Ca_ER_initial
    }
    revK, revCa, revNa, revCl = calculate_reversal_potentials(
        params_cell['K_out'], params_cell['K_in'],
        params_cell['Ca_out'], params_cell['Ca_in'],
        params_cell['Na_out'], params_cell['Na_in'],
        params_cell['Cl_out'], params_cell['Cl_in']
    )
    params_cell.update({
        'reversal_potential_K':  revK,
        'reversal_potential_Ca': revCa,
        'reversal_potential_Na': revNa,
        'reversal_potential_Cl': revCl
    })

    # 2) Single-cell run
    sol_single = run_simulation(params_cell)
    # plot_results_with_two_figures(sol_single, params_cell)  # if desired

    # 3) Multi-cell run with stimulus
    params_multi = {
        'Ng': 5,
        'nvar': 23,
        'g_gap': 0.02,
        'dx': 1.0,
        'cm': membrane_capacitance,
        'cell_params': params_cell,
        'I_app_val': 50.0,
        'stim_cell': 2,
        'stim_start': 100.0,
        'stim_end': 400.0
    }

    sol_multi = run_multi_cell_simulation(params_multi, t_span=(0,300), dt=1.0)
    t_multi = sol_multi.t
    Y_multi = sol_multi.y

    plt.figure()
    for i_cell in range(params_multi['Ng']):
        idxV = i_cell*params_multi['nvar'] + 0
        plt.plot(t_multi, Y_multi[idxV, :], label=f'Cell {i_cell}')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Multi-cell Pericyte Model with Numba (object mode) + Stim")
    plt.legend()
    plt.show()
