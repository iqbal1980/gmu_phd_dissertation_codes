import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
#                            SINGLE-CELL MODEL
###############################################################################
# ------------------ Physical constants and cell parameters ------------------ #
faraday_constant = 96485   # C/mol
gas_constant = 8314        # J/(mol*K)
temperature = 310          # K

membrane_capacitance = 0.94  # pF
cell_volume = 2e-12          # L
Vcyto = cell_volume
ER_volume = cell_volume*0.2

# Ion valences
valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2  # Ca2+

# Concentrations in mM
K_out = 6.26
K_in = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

# De Young-Keizer IP3R Parameters
a1, a2, a3, a4, a5 = 400.0, 0.2, 400.0, 0.2, 20.0
d1, d2, d3, d4, d5 = 0.13, 1.049, 0.9434, 0.1445, 82.34e-3
v1 = 90.0
b1 = a1*d1
b2 = a2*d2
b3 = a3*d3
b4 = a4*d4
b5 = a5*d5

# PMCA parameters
vu, vm = 1540000.0, 2200000.0
aku, akmp = 0.303, 0.14
aru, arm = 1.8, 2.1
p1, p2 = 0.1, 0.01

# Buffer parameters
Bscyt, aKscyt = 225.0, 0.1
Bser, aKser = 2000.0, 1.0
Bm, aKm = 111.0, 0.123

# Other channel/pump/leak parameters
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

# IP3 kinetics
prodip3 = 0.01
V2ip3 = 12.5
ak2ip3 = 6.0
V3ip3 = 0.9
ak3ip3 = 0.1
ak4ip3 = 1.0

# Mitochondrial parameters
Vmito = Vcyto*0.08
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV/1000.0
alphm = 0.2
alphi = 1.0
Vnc = 1.836
aNa = 5000.0
akna = 8000.0
akca = 8.0

# Time and initial conditions
simulation_duration = 100
time_points = 1000

initial_voltage = -70
initial_calcium = Ca_in
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.1
Ca_m_initial = 0.0001

# Normalize IP3R1 states
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

# Normalize IP3R2 states
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

###############################################################################
#                      HELPER FUNCTIONS FOR SINGLE-CELL MODEL
###############################################################################
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    revK = (gas_constant*temperature/(valence_K*faraday_constant))*np.log(K_out/K_in)
    revCa = (gas_constant*temperature/(valence_Ca*faraday_constant))*np.log(Ca_out/Ca_i)
    revNa = (gas_constant*temperature/(valence_Na*faraday_constant))*np.log(Na_out/Na_in)
    revCl = (gas_constant*temperature/(valence_Cl*faraday_constant))*np.log(Cl_out/Cl_in)
    return revK, revCa, revNa, revCl

def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    beta_cyt = 1.0/(1.0 + (Bscyt*aKscyt)/((aKscyt+Ca_i)**2) + (Bm*aKm)/((aKm+Ca_i)**2))
    beta_er  = 1.0/(1.0 + (Bser*aKser)/((aKser+Ca_ER)**2) + (Bm*aKm)/((aKm+Ca_ER)**2))
    return beta_cyt, beta_er

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

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i  = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    return v1*(x110**3)*(Ca_ER - Ca_i)

def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i = max(Ca_i,1e-12)
    u4 = (vu*(Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5 = (vm*(Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1.0 - dpmca)*u5) / 6.6253e5
    return cJpmca, u4, u5

def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i = max(Ca_i,1e-12)
    Ca_m = max(Ca_m,1e-12)
    bb = (z*psi_volts*faraday_constant)/(gas_constant*temperature)
    exp_neg_bb = np.exp(-bb)
    if np.isinf(exp_neg_bb) or np.isnan(exp_neg_bb):
        exp_neg_bb = 0.0
    # J_uni
    J_uni = (Pmito/Vmito)*bb*((alphm*Ca_i*np.exp(-bb) - alphi*Ca_m)/(exp_neg_bb - 1))
    
    som = (aNa**3)*Ca_m/(akna**3*akca)
    soe = (aNa**3)*Ca_i/(akna**3*akca)
    B = np.exp(0.5*psi_volts*z*faraday_constant/(gas_constant*temperature))
    denominator = (1 + (aNa**3/(akna**3)) + Ca_m/akca + som +
                   (aNa**3/(akna**3)) + Ca_i/akca + soe)
    J_nc = Vnc*(B*som - (1.0/B)*soe)/denominator
    return J_uni, J_nc

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

def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params):
    """
    Returns:
      I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA,
      J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR
    """
    Ca_i  = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    
    # Example gating for CaL
    d_inf = 1.0/(1.0 + np.exp(-(V-params['activation_midpoint_CaL'])/params['activation_slope_CaL']))
    f_inf = (1.0/(1.0 + np.exp((V - params['inactivation_midpoint_CaL'])/params['inactivation_slope_CaL']))
             + params['amplitude_factor_CaL']/(1.0 + np.exp((params['voltage_shift_CaL'] - V)/params['slope_factor_CaL'])))
    
    # Kir6.1-like current
    I_Kir61 = (params['conductance_Kir61']*atp*np.sqrt(params['K_out']/params['reference_K'])
               * (V - params['reversal_potential_K'])
               / (1.0 + np.exp((V - params['reversal_potential_K'] - params['voltage_shift_Kir61'])
                               / params['voltage_slope_Kir61'])))
    
    I_TRPC1 = params['conductance_TRPC1']*(V - params['reversal_potential_Ca'])
    I_CaCC  = (params['conductance_CaCC']*(Ca_i/(Ca_i + params['calcium_activation_threshold_CaCC']))
               * (V - params['reversal_potential_Cl']))
    I_CaL   = params['conductance_CaL']*d_inf*f_inf*(V - params['reversal_potential_Ca'])
    I_leak  = params['conductance_leak']*(V - params['reversal_potential_K'])
    
    # PMCA flux => current
    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca*z*faraday_constant*Vcyto*1e6
    
    # SERCA
    J_SERCA = params['k_serca']*(Ca_i**2)/(Ca_i**2 + params['Km_serca']**2)
    # ER leak
    J_ER_leak = params['leak_rate_er']*(Ca_ER - Ca_i)
    
    # NCX (simplistic)
    I_NCX = (params['k_ncx']
             * (params['Na_in']**3/(params['Na_in']**3+87.5**3))
             * (params['Ca_out']/(params['Ca_out']+1.0)))
    
    # IP3R flux
    J_IP3R1 = calculate_ip3r_flux(x110, Ca_ER, Ca_i)*params['conductance_IP3R1']
    J_IP3R2 = calculate_ip3r_flux(x1102, Ca_ER, Ca_i)*params['conductance_IP3R2']
    J_IP3R = J_IP3R1 + J_IP3R2
    
    # RyR
    J_RyR = params['conductance_RyR']*(Ca_i/(Ca_i+0.3))*(Ca_ER - Ca_i)
    
    return (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf,
            I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)

def model(t, y, params):
    """
    Single-cell ODE system with updated channels, IP3R1, IP3R2, SERCA, RyR, Mito flux, etc.
    y is a 23-element array:
       [V, Ca_i, atp, dpmca, Ca_ER,
        x000, x100, x010, x001, x110, x101, x011, x111,
        IP3, Ca_m,
        x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112]
    """
    (V, Ca_i, atp, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m,
     x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112) = y
    
    # Buffering
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    
    # IP3R states
    dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
        calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111)
    dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, dx1112 = \
        calculate_ip3r_states(Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)
    
    # Currents & fluxes
    (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf,
     I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR) = \
        calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params)
    
    # Membrane voltage derivative from sum of ionic currents
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX) / membrane_capacitance
    
    # Ca2+ flux terms
    Ca_influx = -I_CaL/(2.0*z*faraday_constant*cell_volume)
    Ca_efflux = (-I_PMCA - I_NCX)/(2.0*z*faraday_constant*cell_volume)
    
    dCa_dt = beta_cyt*(Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
    
    dCa_ER_dt = beta_er*((Vcyto/ER_volume)*(J_SERCA - J_ER_leak - J_IP3R - J_RyR))
    
    # PMCA regulatory var
    w1 = p1*Ca_i
    w2 = p2
    taom = 1.0/(w1 + w2)
    dpmcainf = w2/(w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca)/taom
    
    # IP3 dynamics
    dIP3_dt = ip3_ode(Ca_i, IP3)
    
    # Mito flux
    J_uni, J_nc = calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt = J_uni - J_nc
    
    return [
        dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
        dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111,
        dIP3_dt, dCa_m_dt,
        dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, x1112
    ]

def run_simulation(params):
    """
    Run the single-cell simulation (if desired).
    """
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

def plot_results_with_two_figures(sol, params):
    """
    Plot single-cell simulation results (like your original code).
    """
    V      = sol.y[0]
    Ca_i   = sol.y[1]
    dpmca  = sol.y[3]
    Ca_ER  = sol.y[4]
    x000   = sol.y[5]
    x100   = sol.y[6]
    x010   = sol.y[7]
    x001   = sol.y[8]
    x110   = sol.y[9]
    x101   = sol.y[10]
    x011   = sol.y[11]
    x111   = sol.y[12]
    IP3    = sol.y[13]
    Ca_m   = sol.y[14]
    x0002  = sol.y[15]
    x1002  = sol.y[16]
    x0102  = sol.y[17]
    x0012  = sol.y[18]
    x1102  = sol.y[19]
    x1012  = sol.y[20]
    x0112  = sol.y[21]
    x1112  = sol.y[22]
    
    # Recompute fluxes for plotting
    I_Kir61_arr, I_TRPC1_arr, I_CaCC_arr, I_CaL_arr, I_leak_arr = \
        (np.zeros_like(sol.t) for _ in range(5))
    d_inf_arr, f_inf_arr = np.zeros_like(sol.t), np.zeros_like(sol.t)
    I_PMCA_arr, J_SERCA_arr, J_ER_leak_arr, I_NCX_arr = \
        (np.zeros_like(sol.t) for _ in range(4))
    J_IP3R_arr, J_RyR_arr = np.zeros_like(sol.t), np.zeros_like(sol.t)
    beta_cyt_arr, beta_er_arr = np.zeros_like(sol.t), np.zeros_like(sol.t)
    J_uni_arr, J_nc_arr = np.zeros_like(sol.t), np.zeros_like(sol.t)
    
    for i, t_val in enumerate(sol.t):
        Cai_val  = max(Ca_i[i], 1e-12)
        CaER_val = max(Ca_ER[i], 1e-12)
        beta_cyt_arr[i], beta_er_arr[i] = calculate_buffering_factors(Cai_val, CaER_val)
        
        (I_Kir61_arr[i], I_TRPC1_arr[i], I_CaCC_arr[i], I_CaL_arr[i], I_leak_arr[i],
         d_inf_arr[i], f_inf_arr[i], I_PMCA_arr[i],
         J_SERCA_arr[i], J_ER_leak_arr[i], I_NCX_arr[i],
         J_IP3R_tmp, J_RyR_tmp) = calculate_currents(
            V[i], Cai_val, params['initial_atp'], dpmca[i], CaER_val,
            x110[i], x1102[i], IP3[i], params
        )
        J_IP3R_arr[i] = J_IP3R_tmp
        J_RyR_arr[i]  = J_RyR_tmp
        
        # Mito flux
        J_uni_arr[i], J_nc_arr[i] = calculate_mito_fluxes(Cai_val, Ca_m[i])
    
    # Figure 1: states
    fig1, axs1 = plt.subplots(10,1,figsize=(12,50))
    fig1.suptitle('Cell Dynamics (Single Cell) - Phase 5', fontsize=16, fontweight='bold')
    
    axs1[0].plot(sol.t, V, color='blue')
    axs1[0].set_title('Membrane Potential (mV)')
    axs1[1].plot(sol.t, Ca_i, color='orange')
    axs1[1].set_title('Cytosolic Ca2+ (mM)')
    axs1[2].plot(sol.t, Ca_ER, color='red')
    axs1[2].set_title('ER Ca2+ (mM)')
    axs1[3].plot(sol.t, IP3, color='magenta')
    axs1[3].set_title('IP3 (mM)')
    axs1[4].plot(sol.t, dpmca, color='cyan')
    axs1[4].set_title('dPMCA State')
    axs1[5].plot(sol.t, beta_cyt_arr, label='Cytosolic', color='green')
    axs1[5].plot(sol.t, beta_er_arr, label='ER', color='blue')
    axs1[5].set_title('Buffering Factors')
    axs1[5].legend()
    axs1[6].plot(sol.t, x110, label='x110 (IP3R1)', color='purple')
    axs1[6].set_title('IP3R1 x110 State')
    axs1[6].legend()
    
    axs1[7].plot(sol.t, x000, label='x000', linestyle='--')
    axs1[7].plot(sol.t, x100, label='x100', linestyle='--')
    axs1[7].plot(sol.t, x010, label='x010', linestyle='--')
    axs1[7].plot(sol.t, x001, label='x001', linestyle='--')
    axs1[7].plot(sol.t, x101, label='x101', linestyle='--')
    axs1[7].plot(sol.t, x011, label='x011', linestyle='--')
    axs1[7].plot(sol.t, x111, label='x111', linestyle='--')
    axs1[7].set_title('Other IP3R1 States')
    axs1[7].legend(ncol=4)
    
    axs1[8].plot(sol.t, x0002, label='x0002', linestyle='--')
    axs1[8].plot(sol.t, x1002, label='x1002', linestyle='--')
    axs1[8].plot(sol.t, x0102, label='x0102', linestyle='--')
    axs1[8].plot(sol.t, x0012, label='x0012', linestyle='--')
    axs1[8].plot(sol.t, x1102, label='x1102', linestyle='--')
    axs1[8].plot(sol.t, x1012, label='x1012', linestyle='--')
    axs1[8].plot(sol.t, x0112, label='x0112', linestyle='--')
    axs1[8].plot(sol.t, x1112, label='x1112', linestyle='--')
    axs1[8].set_title('All IP3R2 States')
    axs1[8].legend(ncol=4)
    
    axs1[9].plot(sol.t, Ca_m, label='Ca_m', color='brown')
    axs1[9].set_title('Mitochondrial Ca2+ (mM)')
    axs1[9].legend()
    
    fig1.tight_layout()
    plt.show()

    # Figure 2: currents & fluxes
    fig2, axs2 = plt.subplots(5,1,figsize=(12,25))
    fig2.suptitle('Currents, Fluxes, Gating (Single Cell)', fontsize=16, fontweight='bold')
    
    axs2[0].plot(sol.t, I_Kir61_arr, label='I_Kir61')
    axs2[0].plot(sol.t, I_TRPC1_arr, label='I_TRPC1')
    axs2[0].plot(sol.t, I_CaCC_arr, label='I_CaCC')
    axs2[0].plot(sol.t, I_CaL_arr, label='I_CaL')
    axs2[0].plot(sol.t, I_leak_arr, label='I_leak')
    axs2[0].plot(sol.t, I_PMCA_arr, label='I_PMCA')
    axs2[0].plot(sol.t, I_NCX_arr, label='I_NCX', color='magenta')
    axs2[0].set_title('Membrane Currents (pA)')
    axs2[0].legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    axs2[1].plot(sol.t, J_SERCA_arr, label='J_SERCA', color='purple')
    axs2[1].plot(sol.t, J_ER_leak_arr, label='J_ER_leak', color='cyan')
    axs2[1].plot(sol.t, J_IP3R_arr, label='J_IP3R', color='pink')
    axs2[1].plot(sol.t, J_RyR_arr, label='J_RyR', color='brown')
    axs2[1].set_title('ER Ca2+ Fluxes (mM/ms)')
    axs2[1].legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    axs2[2].plot(sol.t, J_uni_arr, label='J_uni', color='green')
    axs2[2].plot(sol.t, J_nc_arr, label='J_nc', color='orange')
    axs2[2].set_title('Mito Ca2+ Fluxes (mM/ms)')
    axs2[2].legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    axs2[3].plot(sol.t, d_inf_arr, label='d_inf', linestyle='--', color='purple')
    axs2[3].plot(sol.t, f_inf_arr, label='f_inf', linestyle='--', color='brown')
    axs2[3].set_title('Gating Variables (CaL)')
    axs2[3].legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    axs2[4].plot(sol.t, dpmca, label='dPMCA', color='cyan')
    axs2[4].set_title('PMCA Modulation State')
    axs2[4].legend(loc='center left', bbox_to_anchor=(1,0.5))
    fig2.tight_layout()
    plt.show()


###############################################################################
#                 MULTI-CELL WRAPPER WITH GAP-JUNCTIONS + STIM
###############################################################################
def multi_cell_model(t, Y, params_multi):
    """
    Extends the single-cell model to Ng cells coupled via gap junctions,
    AND adds a time-dependent current injection (I_app) into one cell
    for a specified time window.
    """
    Ng       = params_multi['Ng']          # number of cells
    nvar     = params_multi['nvar']        # 23
    g_gap    = params_multi['g_gap']       # gap-junction conductance
    dx       = params_multi['dx']          # distance between cells
    cm       = params_multi['cm']          # membrane capacitance
    cellparams = params_multi['cell_params']  # single-cell param dict

    # Stimulus info
    I_app_val  = params_multi.get('I_app_val', 0.0)     # pA
    stim_cell  = params_multi.get('stim_cell', 0)       # index of cell
    stim_start = params_multi.get('stim_start', 100.0)  # ms
    stim_end   = params_multi.get('stim_end', 400.0)    # ms

    dYdt = np.zeros_like(Y)
    
    for i in range(Ng):
        idx_start = i * nvar
        idx_end   = idx_start + nvar
        
        # Extract single-cell state for cell i
        y_i = Y[idx_start:idx_end]
        
        # Single-cell derivative
        dyi = model(t, y_i, cellparams)
        
        # The first element in dyi is dV/dt from single-cell perspective
        dVdt_single = dyi[0]
        
        # Current V
        V_i = y_i[0]
        
        # Identify neighbors for gap-junction
        if i == 0:
            V_left = V_i
        else:
            V_left = Y[(i-1)*nvar + 0]
        
        if i == Ng-1:
            V_right = V_i
        else:
            V_right = Y[(i+1)*nvar + 0]
        
        # Gap-junction current
        I_gj = (g_gap / (dx**2)) * (V_left + V_right - 2.0*V_i)
        
        # Convert gap-junction current to dV/dt
        dVdt_multi = dVdt_single + (I_gj / cm)
        
        # ------------------------------------------------
        #  ADD STIMULUS CURRENT if time and cell match
        # ------------------------------------------------
        if (stim_start <= t <= stim_end) and (i == stim_cell):
            # I_app_val in pA => add I_app_val / cm to dV/dt
            dVdt_multi += (I_app_val / cm)
        
        # Store new dV/dt
        dyi[0] = dVdt_multi
        
        # Save final derivatives
        dYdt[idx_start:idx_end] = dyi
    
    return dYdt

def run_multi_cell_simulation(params_multi, t_span=(0,600), dt=0.1):
    """
    Runs the multi-cell simulation using solve_ivp, now with optional stimulus.
    """
    Ng    = params_multi['Ng']
    nvar  = params_multi['nvar']  # 23
    t_eval = np.arange(t_span[0], t_span[1]+dt, dt)

    # Single-cell initial conditions
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
        fun=lambda t, Y: multi_cell_model(t, Y, params_multi),
        t_span=(t_span[0], t_span[1]),
        y0=Y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol

###############################################################################
#                            EXAMPLE MAIN
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
    # Compute reversal potentials
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
    plot_results_with_two_figures(sol_single, params_cell)

    # 3) Multi-cell run with STIMULUS
    params_multi = {
        'Ng':  5,            # 5 cells
        'nvar': 23,          # 23 states per cell
        'g_gap': 0.02,       # gap-junction conductance
        'dx': 1.0,           # distance between cells
        'cm': membrane_capacitance,
        'cell_params': params_cell,
        # Stimulus details:
        'I_app_val': 50.0,     # pA (positive => depolarizing)
        'stim_cell': 2,        # which cell gets the injected current
        'stim_start': 100.0,   # ms
        'stim_end': 400.0      # ms
    }

    sol_multi = run_multi_cell_simulation(params_multi, t_span=(0,600), dt=1.0)
    t_multi = sol_multi.t
    Y_multi = sol_multi.y  # shape = (Ng*nvar, len(t_multi))

    # Plot membrane potential of each cell in the chain
    plt.figure()
    for i_cell in range(params_multi['Ng']):
        idxV = i_cell*params_multi['nvar'] + 0  # V is the 0th state in each cell
        plt.plot(t_multi, Y_multi[idxV, :], label=f'Cell {i_cell}')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Multi-cell Pericyte Model with Stimulated Cell")
    plt.legend()
    plt.show()
