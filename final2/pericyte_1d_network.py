import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##############################################################################
# 1) Global Physical and Model Constants
##############################################################################

faraday_constant = 96485   # C/mol
gas_constant = 8314        # J/(mol*K)
temperature = 310          # K

membrane_capacitance = 0.94  # pF
cell_volume = 2e-12          # L
Vcyto = cell_volume
ER_volume = cell_volume * 0.2

valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2  # Ca2+

K_out = 6.26
K_in  = 140
Ca_out = 2.0
Ca_in  = 0.0001
Na_out = 140
Na_in  = 15.38
Cl_out = 110
Cl_in  = 9.65

# De Young-Keizer IP3R parameters
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
v1 = 90.0
b1 = a1 * d1
b2 = a2 * d2
b3 = a3 * d3
b4 = a4 * d4
b5 = a5 * d5

# PMCA parameters
vu   = 1540000.0
vm   = 2200000.0
aku  = 0.303
akmp = 0.14
aru  = 1.8
arm  = 2.1
p1   = 0.1
p2   = 0.01

# Buffer parameters
Bscyt  = 225.0
aKscyt = 0.1
Bser   = 2000.0
aKser  = 1.0
Bm     = 111.0
aKm    = 0.123

# Other channel/pump conductances
conductance_Kir61  = 0.025
conductance_TRPC1  = 0.001
conductance_CaL    = 0.0002  # Reduced from 0.0005 for a more stable rest
conductance_CaCC   = 0.001
conductance_leak   = 0.005   # Slightly reduced
conductance_IP3R1  = 0.1
conductance_IP3R2  = 0.05
conductance_RyR    = 0.01
k_serca            = 0.1
Km_serca           = 0.5
leak_rate_er       = 0.05
k_ncx              = 0.0005  # Reduced from 0.001

# IP3 kinetics
prodip3  = 0.01
V2ip3    = 12.5
ak2ip3   = 6.0
V3ip3    = 0.9
ak3ip3   = 0.1
ak4ip3   = 1.0

# Mitochondrial parameters
Vmito   = cell_volume * 0.08
Pmito   = 2.776e-20
psi_mV  = 160.0
psi_volts = psi_mV / 1000.0
alphm   = 0.2
alphi   = 1.0
Vnc     = 1.836
aNa     = 5000.0
akna    = 8000.0
akca    = 8.0

##############################################################################
# 2) Initial IP3R States (Normalized)
##############################################################################

x000_initial  = 0.27
x100_initial  = 0.039
x010_initial  = 0.29
x001_initial  = 0.17
x110_initial  = 0.042
x101_initial  = 0.0033
x011_initial  = 0.18
x111_initial  = 0.0035

total1 = (x000_initial + x100_initial + x010_initial + x001_initial
          + x110_initial + x101_initial + x011_initial + x111_initial)
x000_initial /= total1
x100_initial /= total1
x010_initial /= total1
x001_initial /= total1
x110_initial /= total1
x101_initial /= total1
x011_initial /= total1
x111_initial /= total1

# IP3R2
x0002_initial = 0.27
x1002_initial = 0.039
x0102_initial = 0.29
x0012_initial = 0.17
x1102_initial = 0.042
x1012_initial = 0.0033
x0112_initial = 0.18
x1112_initial = 0.0035

total2 = (x0002_initial + x1002_initial + x0102_initial + x0012_initial
          + x1102_initial + x1012_initial + x0112_initial + x1112_initial)
x0002_initial /= total2
x1002_initial /= total2
x0102_initial /= total2
x0012_initial /= total2
x1102_initial /= total2
x1012_initial /= total2
x0112_initial /= total2
x1112_initial /= total2

##############################################################################
# 3) Helper Functions
##############################################################################

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in,
                                  Cl_out, Cl_in):
    rev_K = (gas_constant * temperature / (valence_K * faraday_constant)) * \
             np.log(K_out / K_in)
    rev_Ca = (gas_constant * temperature / (valence_Ca * faraday_constant)) * \
              np.log(Ca_out / Ca_i)
    rev_Na = (gas_constant * temperature / (valence_Na * faraday_constant)) * \
              np.log(Na_out / Na_in)
    rev_Cl = (gas_constant * temperature / (valence_Cl * faraday_constant)) * \
              np.log(Cl_out / Cl_in)
    return rev_K, rev_Ca, rev_Na, rev_Cl


def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i  = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    beta_cyt = 1.0 / (1.0 + (Bscyt * aKscyt)/((aKscyt + Ca_i)**2)
                            + (Bm * aKm)/((aKm + Ca_i)**2))
    beta_er = 1.0 / (1.0 + (Bser * aKser)/((aKser + Ca_ER)**2)
                            + (Bm   * aKm)/((aKm + Ca_ER)**2))
    return beta_cyt, beta_er


def calculate_ip3r_states(Ca_i, IP3,
                          x000, x100, x010, x001, x110, x101, x011, x111):
    Ca_i = max(Ca_i, 1e-12)
    IP3  = max(IP3,   1e-12)
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


def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i  = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    return v1 * (x110**3) * (Ca_ER - Ca_i)


def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i = max(Ca_i, 1e-12)
    u4   = (vu * (Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5   = (vm * (Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1-dpmca)*u5) / (6.6253e5)
    return cJpmca, u4, u5


def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i = max(Ca_i, 1e-12)
    Ca_m = max(Ca_m, 1e-12)
    bb = (z*psi_volts*faraday_constant)/(gas_constant*temperature)
    exp_neg_bb = np.exp(-bb)
    if np.isinf(exp_neg_bb) or np.isnan(exp_neg_bb):
        exp_neg_bb = 0.0

    J_uni = (Pmito / Vmito) * bb * ((alphm * Ca_i * np.exp(-bb) - alphi * Ca_m)
                                   / (np.exp(-bb) - 1))

    som = (aNa**3)*Ca_m / (akna**3*akca)
    soe = (aNa**3)*Ca_i / (akna**3*akca)
    B = np.exp(0.5*psi_volts*z*faraday_constant/(gas_constant*temperature))
    denominator = (1 + (aNa**3/(akna**3)) + Ca_m/akca + som
                     + (aNa**3/(akna**3)) + Ca_i/akca + soe)
    J_nc = Vnc*(B*som - (1/B)*soe)/denominator

    return J_uni, J_nc


def ip3_ode(Ca_i, IP3):
    Ca_i = max(Ca_i, 1e-12)
    IP3  = max(IP3,  1e-12)
    term1 = prodip3
    term2 = V2ip3 / (1 + (ak2ip3 / IP3))
    term3 = (V3ip3 / (1 + (ak3ip3 / IP3))) * (1 / (1 + (ak4ip3 / Ca_i)))
    dIP3dt = term1 - term2 - term3

    new_IP3 = IP3 + dIP3dt
    if new_IP3 > 0.01:
        dIP3dt = (0.01 - IP3)
    if new_IP3 < 1e-12:
        dIP3dt = (1e-12 - IP3)
    return dIP3dt

##############################################################################
# 4) A Helper Function for Time-Limited Current Injection
##############################################################################

def injected_current(t, cell_index, x_inject, y_inject, amplitude,
                     t_on, t_off):
    """
    Returns 'amplitude' pA if cell_index is between x_inject & y_inject
    AND time is in [t_on, t_off]. Otherwise returns 0.
    """
    if (x_inject <= cell_index <= y_inject) and (t_on <= t <= t_off):
        return amplitude
    else:
        return 0.0

##############################################################################
# 5) Calculate Currents for Single Cell
##############################################################################

def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params):
    Ca_i  = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER,1e-12)

    revK   = params['reversal_potential_K']
    revCa  = params['reversal_potential_Ca']
    revCl  = params['reversal_potential_Cl']
    K_ext  = params['K_out']
    K_ref  = params['reference_K']

    # Gating for L-type Ca channel
    d_inf = 1/(1 + np.exp(-(V - params['activation_midpoint_CaL'])
                          / params['activation_slope_CaL']))
    f_inf = 1/(1 + np.exp((V - params['inactivation_midpoint_CaL'])
                          / params['inactivation_slope_CaL'])) + \
            params['amplitude_factor_CaL']/(1 + np.exp(
                (params['voltage_shift_CaL'] - V)/params['slope_factor_CaL']))

    I_Kir61 = params['conductance_Kir61'] * atp * np.sqrt(K_ext / K_ref) * \
              (V - revK) / (1 + np.exp((V - revK - params['voltage_shift_Kir61'])
                                       / params['voltage_slope_Kir61']))

    I_TRPC1 = params['conductance_TRPC1'] * (V - revCa)

    I_CaCC  = params['conductance_CaCC'] * (Ca_i / (Ca_i +
                    params['calcium_activation_threshold_CaCC'])) * (V - revCl)
    I_CaL   = params['conductance_CaL'] * d_inf * f_inf * (V - revCa)
    I_leak  = params['conductance_leak'] * (V - revK)

    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca * z * faraday_constant * Vcyto * 1e6  # pA

    J_SERCA   = params['k_serca']*(Ca_i**2)/(Ca_i**2+params['Km_serca']**2)
    J_ER_leak = params['leak_rate_er']*(Ca_ER - Ca_i)
    I_NCX     = params['k_ncx']*(Na_in**3/(Na_in**3+87.5**3))*(Ca_out/(Ca_out+1))

    # IP3R flux
    J_IP3R1 = calculate_ip3r_flux(x110,  Ca_ER, Ca_i)*params['conductance_IP3R1']
    J_IP3R2 = calculate_ip3r_flux(x1102, Ca_ER, Ca_i)*params['conductance_IP3R2']
    J_IP3R  = J_IP3R1 + J_IP3R2

    J_RyR = params['conductance_RyR']*(Ca_i/(Ca_i+0.3))*(Ca_ER - Ca_i)

    return (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA,
            J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)

##############################################################################
# 6) 1D Network Model ODE
##############################################################################

def model_1d(t, y, params):
    N = params['num_cells']
    dydt = np.zeros_like(y)
    num_states_per_cell = 23

    g_gap       = params['g_gap']
    x_inject    = params['x_inject']
    y_inject    = params['y_inject']
    I_app_amp   = params['I_app_amplitude']
    I_app_t_on  = params['I_app_t_on']
    I_app_t_off = params['I_app_t_off']

    for i in range(N):
        i_start = num_states_per_cell * i

        V_i    = y[i_start + 0 ]
        Ca_i   = y[i_start + 1 ]
        atp_i  = y[i_start + 2 ]
        dpmca_i= y[i_start + 3 ]
        CaER_i = y[i_start + 4 ]
        x000_i = y[i_start + 5 ]
        x100_i = y[i_start + 6 ]
        x010_i = y[i_start + 7 ]
        x001_i = y[i_start + 8 ]
        x110_i = y[i_start + 9 ]
        x101_i = y[i_start + 10]
        x011_i = y[i_start + 11]
        x111_i = y[i_start + 12]
        IP3_i  = y[i_start + 13]
        Ca_m_i = y[i_start + 14]
        x0002_i= y[i_start + 15]
        x1002_i= y[i_start + 16]
        x0102_i= y[i_start + 17]
        x0012_i= y[i_start + 18]
        x1102_i= y[i_start + 19]
        x1012_i= y[i_start + 20]
        x0112_i= y[i_start + 21]
        x1112_i= y[i_start + 22]

        # Buffering
        beta_cyt, beta_er = calculate_buffering_factors(Ca_i, CaER_i)

        # IP3R states
        (dx000, dx100, dx010, dx001,
         dx110, dx101, dx011, dx111) = \
            calculate_ip3r_states(Ca_i, IP3_i,
                                  x000_i, x100_i, x010_i, x001_i,
                                  x110_i, x101_i, x011_i, x111_i)

        (dx0002, dx1002, dx0102, dx0012,
         dx1102, dx1012, dx0112, dx1112) = \
            calculate_ip3r_states(Ca_i, IP3_i,
                                  x0002_i, x1002_i, x0102_i, x0012_i,
                                  x1102_i, x1012_i, x0112_i, x1112_i)

        # Single-cell fluxes
        (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA,
         J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR) = \
            calculate_currents(V_i, Ca_i, atp_i, dpmca_i,
                               CaER_i, x110_i, x1102_i, IP3_i, params)

        dVdt_single = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak
                       - I_PMCA - I_NCX) / membrane_capacitance

        J_uni_i, J_nc_i = calculate_mito_fluxes(Ca_i, Ca_m_i)

        Ca_influx = -I_CaL/(2 * faraday_constant * cell_volume)
        Ca_efflux = (-I_PMCA - I_NCX)/(2 * faraday_constant * cell_volume)
        dCa_dt_single = beta_cyt * (Ca_influx + Ca_efflux
                                    + J_ER_leak + J_IP3R + J_RyR
                                    - J_SERCA)

        dCa_ER_dt_single = beta_er * (Vcyto/ER_volume) * (
            J_SERCA - J_ER_leak - J_IP3R - J_RyR)

        w1 = p1 * Ca_i
        w2 = p2
        taom = 1.0 / (w1 + w2)
        dpmcainf = w2 / (w1 + w2)
        ddpmca_dt = (dpmcainf - dpmca_i) / taom

        dIP3_dt = ip3_ode(Ca_i, IP3_i)
        dCa_m_dt = J_uni_i - J_nc_i

        # Gap-Junction Coupling
        I_gap = 0.0
        if i > 0:
            V_left = y[num_states_per_cell*(i-1)]
            I_gap += g_gap*(V_left - V_i)
        if i < N-1:
            V_right = y[num_states_per_cell*(i+1)]
            I_gap += g_gap*(V_right - V_i)

        dVdt_coupled = dVdt_single + I_gap / membrane_capacitance

        # Time-limited patch current
        i_inject = injected_current(t, i, x_inject, y_inject,
                                    I_app_amp, I_app_t_on, I_app_t_off)
        dVdt_coupled += i_inject / membrane_capacitance

        # Store derivatives
        dydt[i_start + 0]  = dVdt_coupled
        dydt[i_start + 1]  = dCa_dt_single
        dydt[i_start + 2]  = 0.0  # if atp is fixed
        dydt[i_start + 3]  = ddpmca_dt
        dydt[i_start + 4]  = dCa_ER_dt_single
        dydt[i_start + 5]  = dx000
        dydt[i_start + 6]  = dx100
        dydt[i_start + 7]  = dx010
        dydt[i_start + 8]  = dx001
        dydt[i_start + 9]  = dx110
        dydt[i_start + 10] = dx101
        dydt[i_start + 11] = dx011
        dydt[i_start + 12] = dx111
        dydt[i_start + 13] = dIP3_dt
        dydt[i_start + 14] = dCa_m_dt
        dydt[i_start + 15] = dx0002
        dydt[i_start + 16] = dx1002
        dydt[i_start + 17] = dx0102
        dydt[i_start + 18] = dx0012
        dydt[i_start + 19] = dx1102
        dydt[i_start + 20] = dx1012
        dydt[i_start + 21] = dx0112
        dydt[i_start + 22] = dx1112

    return dydt

##############################################################################
# 7) Run the Simulation
##############################################################################

def run_1d_network_simulation(params):
    N = params['num_cells']
    num_states_per_cell = 23

    # Build the initial condition vector
    Y0 = np.zeros(num_states_per_cell * N)

    for i in range(N):
        idx = num_states_per_cell*i
        Y0[idx + 0]  = params['initial_voltage']   # V
        Y0[idx + 1]  = params['initial_calcium']   # Ca_i
        Y0[idx + 2]  = params['initial_atp']       # atp
        Y0[idx + 3]  = params['initial_dpmca']     # dpmca
        Y0[idx + 4]  = params['Ca_ER_initial']     # Ca_ER
        Y0[idx + 5]  = x000_initial
        Y0[idx + 6]  = x100_initial
        Y0[idx + 7]  = x010_initial
        Y0[idx + 8]  = x001_initial
        Y0[idx + 9]  = x110_initial
        Y0[idx + 10] = x101_initial
        Y0[idx + 11] = x011_initial
        Y0[idx + 12] = x111_initial
        Y0[idx + 13] = 0.1      # IP3
        Y0[idx + 14] = 0.0001   # Ca_m
        Y0[idx + 15] = x0002_initial
        Y0[idx + 16] = x1002_initial
        Y0[idx + 17] = x0102_initial
        Y0[idx + 18] = x0012_initial
        Y0[idx + 19] = x1102_initial
        Y0[idx + 20] = x1012_initial
        Y0[idx + 21] = x0112_initial
        Y0[idx + 22] = x1112_initial

    # Solve
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])

    sol = solve_ivp(fun=model_1d,
                    t_span=t_span,
                    y0=Y0,
                    t_eval=t_eval,
                    args=(params,),
                    method='RK45')
    return sol

##############################################################################
# 8) Plotting
##############################################################################

def plot_1d_results(sol, params):
    t = sol.t
    Y = sol.y
    N = params['num_cells']
    num_states_per_cell = 23

    fig, ax = plt.subplots(figsize=(9,5))
    # Example: plot a few cells' membrane potential
    cells_to_plot = [0, N//2, N-1]
    for cell_idx in cells_to_plot:
        V = Y[num_states_per_cell*cell_idx + 0, :]
        ax.plot(t, V, label=f'Cell {cell_idx}')

    ax.set_title('Membrane Potential in a 1D Pericyte Chain (Improved)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

##############################################################################
# 9) Main Execution
##############################################################################

if __name__ == '__main__':

    revK, revCa, revNa, revCl = calculate_reversal_potentials(
        K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in
    )

    params = {
        'num_cells':  200,
        'simulation_duration': 10.0,   # seconds
        'time_points': 1000,

        # Single-cell resting conditions
        'initial_voltage':  -70.0,     # more negative start
        'initial_calcium':  0.0001,    # 100 nM
        'initial_atp':      1.0,       # moderate ATP => some Kir open
        'initial_dpmca':    1.0,
        'Ca_ER_initial':    0.5,

        # Ion reversal potentials
        'reversal_potential_K':  revK,
        'reversal_potential_Ca': revCa,
        'reversal_potential_Cl': revCl,
        'reversal_potential_Na': revNa,

        'K_out': K_out,
        'reference_K': 5.4,

        'voltage_slope_Kir61':   6.0,
        'voltage_shift_Kir61':  15.0,

        'activation_midpoint_CaL':   -40,
        'activation_slope_CaL':       4,
        'inactivation_midpoint_CaL': -45,
        'inactivation_slope_CaL':     5,
        'voltage_shift_CaL':         50,
        'slope_factor_CaL':          20,
        'amplitude_factor_CaL':      0.6,

        'conductance_Kir61': conductance_Kir61,
        'conductance_TRPC1': conductance_TRPC1,
        'conductance_CaCC':  conductance_CaCC,
        'conductance_CaL':   conductance_CaL,
        'conductance_leak':  conductance_leak,
        'conductance_IP3R1': conductance_IP3R1,
        'conductance_IP3R2': conductance_IP3R2,
        'conductance_RyR':   conductance_RyR,

        'k_serca':       k_serca,
        'Km_serca':      Km_serca,
        'leak_rate_er':  leak_rate_er,
        'k_ncx':         k_ncx,
        'calcium_activation_threshold_CaCC': 0.0005,

        # Gap-junction coupling
        'g_gap': 0.002,  

        # Time-limited patch current
        'x_inject':       5,       # first cell to inject
        'y_inject':       10,      # last cell to inject
        'I_app_amplitude': 5.0,    # pA amplitude
        'I_app_t_on':     2.0,     # start injecting at t=2s
        'I_app_t_off':    3.0      # stop injecting at t=3s
    }

    sol = run_1d_network_simulation(params)
    plot_1d_results(sol, params)
