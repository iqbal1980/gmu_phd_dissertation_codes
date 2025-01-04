import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit

###############################################################################
# 1) GLOBAL CONSTANTS
###############################################################################
faraday_constant = 96485
gas_constant     = 8314
temperature      = 310

membrane_capacitance = 0.94
cell_volume    = 2e-12
Vcyto          = cell_volume
ER_volume      = cell_volume*0.2
z = 2  # Ca2+ valence

# Buffers
Bscyt, aKscyt = 225.0, 0.1
Bser, aKser   = 2000.0, 1.0
Bm, aKm       = 111.0, 0.123

# PMCA
p1, p2 = 0.1, 0.01

EXP_CAP = 50.0  # clamp exponent to avoid overflow

###############################################################################
# 2) SAFE EXPONENTIAL CLAMP
###############################################################################
@jit(forceobj=True)
def safe_exp(x, cap=EXP_CAP):
    """Clamps x into [-cap, cap], then calls math.exp(x)."""
    if x > cap:
        x = cap
    elif x < -cap:
        x = -cap
    return math.exp(x)

###############################################################################
# 3) CALCULATE_REVERSAL_POTENTIALS
###############################################################################
valence_K  = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1

@jit(forceobj=True)
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    """Returns revK, revCa, revNa, revCl in Volts."""
    valK = (gas_constant * temperature) / (valence_K * faraday_constant)
    revK  = valK * math.log(K_out / K_in)

    valCa= (gas_constant * temperature) / (valence_Ca * faraday_constant)
    revCa= valCa * math.log(Ca_out / Ca_i)

    valNa= (gas_constant * temperature) / (valence_Na * faraday_constant)
    revNa= valNa * math.log(Na_out / Na_in)

    valCl= (gas_constant * temperature) / (abs(valence_Cl)* faraday_constant)
    revCl= valCl * math.log(Cl_out / Cl_in)
    return revK, revCa, revNa, revCl

###############################################################################
# 4) HELPER BUFFERS & IP3R
###############################################################################
@jit(forceobj=True)
def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i  = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    top1 = (Bscyt*aKscyt)/((aKscyt+Ca_i)**2)
    top2 = (Bm*aKm)/((aKm+Ca_i)**2)
    beta_cyt= 1.0/(1.0 + top1 + top2)

    top3= (Bser*aKser)/((aKser+Ca_ER)**2)
    top4= (Bm*aKm)/((aKm+Ca_ER)**2)
    beta_er= 1.0/(1.0 + top3 + top4)
    return beta_cyt, beta_er

# De Young-Keizer IP3R
a1, a2, a3, a4, a5 = 400.0, 0.2, 400.0, 0.2, 20.0
d1, d2, d3, d4, d5 = 0.13, 1.049, 0.9434, 0.1445, 82.34e-3
v1= 90.0
b1= a1*d1
b2= a2*d2
b3= a3*d3
b4= a4*d4
b5= a5*d5

@jit(forceobj=True)
def calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111):
    Ca_i= max(Ca_i,1e-12)
    IP3 = max(IP3,1e-12)

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

    dx000= f1 + f2 + f3
    dx100= f4 + f5 - f2
    dx010= -f1+ f6 + f7
    dx001= f8 - f3+ f9
    dx110= -f4- f6+ f10
    dx101= f11- f9- f5
    dx011= -f8- f7+ f12
    dx111= -f11- f12- f10
    return dx000,dx100,dx010,dx001,dx110,dx101,dx011,dx111

@jit(forceobj=True)
def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i= max(Ca_i,1e-12)
    Ca_ER= max(Ca_ER,1e-12)
    return v1*(x110**3)*(Ca_ER- Ca_i)

###############################################################################
# 5) Mito flux, IP3 ODE
###############################################################################
Pmito= 2.776e-20
psi_mV=160.0
psi_volts= psi_mV/1000.0
alphm=0.2
alphi=1.0
Vnc=1.836
aNa=5000.0
akna=8000.0
akca=8.0

@jit(forceobj=True)
def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i= max(Ca_i,1e-12)
    # Hard-coded from your code
    vu=1540000.0
    vm=2200000.0
    aku=0.303
    akmp=0.14
    aru=1.8
    arm=2.1
    u4= (vu*(Ca_i**aru))/(Ca_i**aru+ aku**aru)
    u5= (vm*(Ca_i**arm))/(Ca_i**arm+ akmp**arm)
    cJpmca= (dpmca*u4+ (1.0- dpmca)*u5)/6.6253e5
    return cJpmca, u4, u5

@jit(forceobj=True)
def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i= max(Ca_i,1e-12)
    Ca_m= max(Ca_m,1e-12)
    bb= (z* psi_volts* faraday_constant)/(gas_constant* temperature)
    e_neg= safe_exp(-bb)
    if math.isinf(e_neg) or math.isnan(e_neg):
        e_neg=0.0

    # J_uni => same as your code except "cell_volume*0.08" = Vmito
    J_uni= (Pmito/ (cell_volume* 0.08))* bb*((alphm* Ca_i* e_neg- alphi* Ca_m)/(e_neg-1))

    som= (aNa**3)* Ca_m/(akna**3* akca)
    soe= (aNa**3)* Ca_i/(akna**3* akca)
    B= safe_exp(0.5* psi_volts* z* faraday_constant/(gas_constant* temperature))
    denom= (1+(aNa**3/(akna**3))+ Ca_m/akca+ som+(aNa**3/(akna**3))+ Ca_i/akca+ soe)
    J_nc= Vnc*( B* som- (1.0/B)* soe)/ denom
    return J_uni, J_nc

# IP3 ODE
prodip3=0.01
V2ip3=12.5
ak2ip3=6.0
V3ip3=0.9
ak3ip3=0.1
ak4ip3=1.0

@jit(forceobj=True)
def ip3_ode(Ca_i, IP3):
    Ca_i= max(Ca_i,1e-12)
    IP3=  max(IP3,1e-12)
    term1= prodip3
    denom2= (1.0+ (ak2ip3/ IP3))
    term2= V2ip3/ denom2
    denom3a= (1.0+ (ak3ip3/ IP3))
    denom3b= (1.0+ (ak4ip3/ Ca_i))
    term3= (V3ip3/ denom3a)* (1.0/ denom3b)

    dIP3dt= term1- term2- term3
    new_IP3= IP3+ dIP3dt
    if new_IP3>0.01:
        dIP3dt= (0.01- IP3)
    if new_IP3<1e-12:
        dIP3dt= (1e-12- IP3)
    return dIP3dt

###############################################################################
# 6) Currents (with smaller Kir and the same CaL)
###############################################################################
k_serca=0.1
Km_serca=0.5
leak_rate_er=0.05
k_ncx=0.001

# *** KEY ADJUSTMENT #1: reduce Kir conductance from 0.025 -> 0.01 ***
conductance_Kir61= 0.01

conductance_TRPC1=0.001
conductance_CaCC= 0.001
conductance_CaL=  0.0005
conductance_leak=0.01
conductance_IP3R1=0.1
conductance_IP3R2=0.05
conductance_RyR=  0.01
calcium_extrusion_rate=100.0
resting_calcium=0.001
calcium_activation_threshold_CaCC=0.0005

@jit(forceobj=True)
def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params):
    Ca_i= max(Ca_i,1e-12)
    Ca_ER= max(Ca_ER,1e-12)

    # CaL gating
    argA= -(V- params['activation_midpoint_CaL'])/ params['activation_slope_CaL']
    eA= safe_exp(argA)
    d_inf= 1.0/(1.0+ eA)

    argB= (V- params['inactivation_midpoint_CaL'])/ params['inactivation_slope_CaL']
    eB= safe_exp(argB)
    partB= 1.0/(1.0+ eB)

    argC= (params['voltage_shift_CaL']- V)/ params['slope_factor_CaL']
    eC= safe_exp(argC)
    partC= params['amplitude_factor_CaL']/ (1.0+ eC)
    f_inf= partB+ partC

    # *** KEY ADJUSTMENT #2: we use a smaller Kir conductance from the dictionary. ***
    # Kir
    argK= (V- params['reversal_potential_K']- params['voltage_shift_Kir61'])/ params['voltage_slope_Kir61']
    eK= safe_exp(argK)
    denomKir= 1.0+ eK
    # replaced "params['conductance_Kir61']" from the dictionary. We'll keep it in dictionary
    # so that it's 0.01 now. 
    I_Kir61= (params['conductance_Kir61']* atp* math.sqrt(params['K_out']/ params['reference_K'])
              *(V- params['reversal_potential_K'])/ denomKir)

    I_TRPC1= params['conductance_TRPC1']*(V- params['reversal_potential_Ca'])
    # CaCC
    fracCa= Ca_i/(Ca_i+ params['calcium_activation_threshold_CaCC'])
    I_CaCC= params['conductance_CaCC']* fracCa* (V- params['reversal_potential_Cl'])

    # CaL
    I_CaL= params['conductance_CaL']* d_inf* f_inf*(V- params['reversal_potential_Ca'])
    I_leak= params['conductance_leak']*(V- params['reversal_potential_K'])

    # PMCA
    cJpmca,_,_= calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA= cJpmca* z* faraday_constant* cell_volume* 1e6

    # SERCA
    J_SERCA= params['k_serca']*(Ca_i**2)/( Ca_i**2+ params['Km_serca']**2)
    # ER leak
    J_ER_leak= params['leak_rate_er']*(Ca_ER- Ca_i)
    # NCX
    iNcxNum= (params['Na_in']**3)/(params['Na_in']**3+ 87.5**3)
    iNcxDen= (params['Ca_out']/(params['Ca_out']+1.0))
    I_NCX= params['k_ncx']* iNcxNum* iNcxDen

    # IP3R
    J_IP3R1= calculate_ip3r_flux(x110,  Ca_ER, Ca_i)* params['conductance_IP3R1']
    J_IP3R2= calculate_ip3r_flux(x1102, Ca_ER, Ca_i)* params['conductance_IP3R2']
    J_IP3R= J_IP3R1+ J_IP3R2

    # RyR
    partRy= Ca_i/(Ca_i+ 0.3)
    J_RyR= params['conductance_RyR']* partRy* (Ca_ER- Ca_i)

    return (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak,
            d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)

###############################################################################
# 7) BUILD SINGLE-CELL MODEL WITH PRINT
###############################################################################
def build_single_cell_model(debug_dt):
    """
    Returns a function that prints time every debug_dt ms.
    """
    last_print= [float('-inf')]

    @jit(forceobj=True)
    def model_func(t, y, params):
        if t - last_print[0]>= debug_dt:
            print("[model DEBUG] t=%.4f" % t)
            last_print[0]= t

        (V, Ca_i, atp, dpmca, Ca_ER,
         x000, x100, x010, x001, x110, x101, x011, x111,
         IP3, Ca_m,
         x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)= y

        beta_cyt, beta_er= calculate_buffering_factors(Ca_i, Ca_ER)
        (dx000,dx100,dx010,dx001,
         dx110,dx101,dx011,dx111)= calculate_ip3r_states(
            Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111
        )
        (dx0002,dx1002,dx0102,dx0012,
         dx1102,dx1012,dx0112,dx1112)= calculate_ip3r_states(
            Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112
        )

        (I_Kir61,I_TRPC1,I_CaCC,I_CaL,I_leak,
         d_inf,f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)= \
            calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params)

        dV_dt= (-I_Kir61- I_TRPC1- I_CaCC- I_CaL
                - I_leak- I_PMCA- I_NCX)/ membrane_capacitance

        Ca_influx= -I_CaL/(2.0*z* faraday_constant* cell_volume)
        Ca_efflux= (-I_PMCA - I_NCX)/(2.0*z* faraday_constant* cell_volume)
        dCa_dt= beta_cyt*(Ca_influx+ Ca_efflux+ J_ER_leak+ J_IP3R+ J_RyR - J_SERCA)
        dCa_ER_dt= beta_er*((Vcyto/ER_volume)*(J_SERCA- J_ER_leak- J_IP3R- J_RyR))

        w1= p1* Ca_i
        w2= p2
        taom= 1.0/(w1+ w2)
        dpmcainf= w2/(w1+ w2)
        ddpmca_dt= (dpmcainf- dpmca)/ taom

        dIP3_dt= ip3_ode(Ca_i, IP3)
        J_uni, J_nc= calculate_mito_fluxes(Ca_i, Ca_m)
        dCa_m_dt= J_uni- J_nc

        return [
            dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
            dx000,dx100,dx010,dx001, x110, x101, x011, x111,
            dIP3_dt, dCa_m_dt,
            dx0002,dx1002,dx0102, x0012, x1102, x1012, x0112, x1112
        ]

    return model_func

###############################################################################
# 8) RUN_SINGLE_CELL
###############################################################################
def run_simulation(params):
    print("[DEBUG] Starting single-cell simulation with final time = %d ms using BDF solver..." %
          params['simulation_duration'])

    model_func= build_single_cell_model(debug_dt=1.0)
    t_span= (0, params['simulation_duration'])
    t_eval= np.linspace(0, params['simulation_duration'], params['time_points'])

    y0= [
        params['initial_voltage'],
        params['initial_calcium'],
        params['initial_atp'],
        params['initial_dpmca'],
        params['Ca_ER_initial'],
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035,
        0.1,
        0.0001,
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035
    ]

    start_time= time.perf_counter()
    sol= solve_ivp(
        lambda t, Y: model_func(t, Y, params),
        t_span, y0,
        method='BDF',
        t_eval=t_eval,
        max_step=1.0,
        rtol=1e-5,
        atol=1e-8
    )
    end_time= time.perf_counter()
    print("[DEBUG] Single-cell simulation completed in %.4f s." % (end_time- start_time))
    return sol

###############################################################################
# 9) PLOT_SINGLE_CELL
###############################################################################
def plot_results_with_two_figures(sol, params):
    print("[DEBUG] Plotting single-cell results (brief).")
    plt.figure()
    plt.plot(sol.t, sol.y[0], label="Vm")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Single-Cell Potential (with debug prints in model)")
    plt.legend()
    plt.show()

###############################################################################
# 10) SINGLE_CELL_MODEL_NO_PRINTS for multi-cell
###############################################################################
@jit(forceobj=True)
def single_cell_model_no_prints(t, y, params):
    (V, Ca_i, atp, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m,
     x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)= y

    beta_cyt, beta_er= calculate_buffering_factors(Ca_i, Ca_ER)
    (dx000,dx100,dx010,dx001,
     dx110,dx101,dx011,dx111)= calculate_ip3r_states(
        Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111
    )
    (dx0002, dx1002, dx0102, dx0012,
     dx1102, dx1012, dx0112, dx1112)= calculate_ip3r_states(
        Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112
    )

    (I_Kir61,I_TRPC1,I_CaCC,I_CaL,I_leak,
     d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)= \
        calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, params)

    dV_dt= (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL
            - I_leak - I_PMCA - I_NCX)/ membrane_capacitance

    Ca_influx= -I_CaL/(2.0*z* faraday_constant* cell_volume)
    Ca_efflux= (-I_PMCA - I_NCX)/(2.0*z* faraday_constant* cell_volume)
    dCa_dt= beta_cyt*(Ca_influx+ Ca_efflux+ J_ER_leak+ J_IP3R+ J_RyR - J_SERCA)
    dCa_ER_dt= beta_er*((Vcyto/ER_volume)*(J_SERCA- J_ER_leak- J_IP3R- J_RyR))

    w1= p1* Ca_i
    w2= p2
    taom= 1.0/(w1+ w2)
    dpmcainf= w2/(w1+ w2)
    ddpmca_dt= (dpmcainf- dpmca)/ taom

    dIP3_dt= ip3_ode(Ca_i, IP3)
    J_uni, J_nc= calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt= J_uni- J_nc

    return [
        dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
        dx000,dx100,dx010,dx001, x110, x101, x011, x111,
        dIP3_dt, dCa_m_dt,
        dx0002,dx1002,dx0102, x0012, x1102, x1012, x0112, x1112
    ]

###############################################################################
# 11) MULTI_CELL_MODEL
###############################################################################
@jit(forceobj=True)
def multi_cell_model(t, Y, params_multi):
    Ng     = params_multi['Ng']
    nvar   = params_multi['nvar']
    g_gap  = params_multi['g_gap']
    dx     = params_multi['dx']
    cm     = params_multi['cm']
    cparams= params_multi['cell_params']

    # *** KEY ADJUSTMENT #3: we will use a bigger I_app_val => e.g. 300 pA. ***
    I_app_val= params_multi.get('I_app_val', 300.0)
    stim_cell= params_multi.get('stim_cell', 0)
    stim_start= params_multi.get('stim_start', 100.0)
    stim_end=   params_multi.get('stim_end', 400.0)

    dYdt= np.zeros_like(Y)
    for i in range(Ng):
        idx_start= i* nvar
        idx_end  = idx_start+ nvar
        y_i= Y[idx_start: idx_end]

        dyi= single_cell_model_no_prints(t, y_i, cparams)

        dVdt_single= dyi[0]
        V_i= y_i[0]

        if i==0:
            V_left= V_i
        else:
            V_left= Y[(i-1)* nvar +0]
        if i== Ng-1:
            V_right= V_i
        else:
            V_right= Y[(i+1)* nvar +0]

        I_gj= (g_gap/(dx**2))* (V_left+ V_right- 2.0* V_i)
        dVdt_multi= dVdt_single+ (I_gj/ cm)

        # Stim
        if (stim_start <= t <= stim_end) and (i== stim_cell):
            dVdt_multi+= (I_app_val/ cm)

        dyi[0]= dVdt_multi
        dYdt[idx_start: idx_end]= dyi

    return dYdt

###############################################################################
# 12) RUN_MULTI_CELL_SIMULATION
###############################################################################
def run_multi_cell_simulation(params_multi, t_span=(0,600), dt=1.0):
    print("[DEBUG] Starting multi-cell simulation with final time = %d ms using BDF solver..." %
          t_span[1])
    Ng   = params_multi['Ng']
    nvar = params_multi['nvar']
    t_eval= np.arange(t_span[0], t_span[1]+ dt, dt)

    ic= [
        params_multi['cell_params']['initial_voltage'],
        params_multi['cell_params']['initial_calcium'],
        params_multi['cell_params']['initial_atp'],
        params_multi['cell_params']['initial_dpmca'],
        params_multi['cell_params']['Ca_ER_initial'],
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035,
        0.1,
        0.0001,
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035
    ]
    Y0= np.tile(ic, Ng)

    start_time= time.perf_counter()
    sol= solve_ivp(
        lambda t, Y: multi_cell_model(t, Y, params_multi),
        (t_span[0], t_span[1]),
        Y0,
        t_eval= t_eval,
        method='BDF',
        max_step=1.0,
        rtol=1e-5,
        atol=1e-8
    )
    end_time= time.perf_counter()
    print("[DEBUG] Multi-cell simulation completed in %.4f s." % (end_time- start_time))
    return sol

###############################################################################
# 13) MAIN
###############################################################################
if __name__=="__main__":
    print("[DEBUG] Script started...")

    # Build single-cell param dictionary
    # NOTE: we changed conductance_Kir61 => 0.01
    # so let's store that in the dictionary
    params_cell= {
        'K_out':6.26, 'K_in':140, 'Ca_out':2.0, 'Ca_in':0.0001,
        'Na_out':140,'Na_in':15.38,'Cl_out':110,'Cl_in':9.65,
        'conductance_Kir61':0.01,     # <--- reduced Kir
        'conductance_TRPC1':0.001,
        'conductance_CaCC':0.001,
        'conductance_CaL':0.0005,
        'conductance_leak':0.01,
        'conductance_IP3R1':0.1,
        'conductance_IP3R2':0.05,
        'conductance_RyR':0.01,
        'k_serca':0.1,
        'Km_serca':0.5,
        'leak_rate_er':0.05,
        'k_ncx':0.001,
        'calcium_extrusion_rate':100.0,
        'resting_calcium':0.001,
        'calcium_activation_threshold_CaCC':0.0005,
        'reference_K':5.4,
        'voltage_slope_Kir61':6,
        'voltage_shift_Kir61':15,
        'activation_midpoint_CaL':-40,
        'activation_slope_CaL':4,
        'inactivation_midpoint_CaL':-45,
        'inactivation_slope_CaL':5,
        'voltage_shift_CaL':50,
        'slope_factor_CaL':20,
        'amplitude_factor_CaL':0.6,
        'simulation_duration':100,
        'time_points':1000,
        'initial_voltage':-70,
        'initial_calcium':0.0001,
        'initial_atp':4.4,
        'initial_dpmca':1.0,
        'Ca_ER_initial':0.5
    }

    # reversal potentials
    revK, revCa, revNa, revCl= calculate_reversal_potentials(
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

    # Single-cell run
    sol_single= run_simulation(params_cell)
    plot_results_with_two_figures(sol_single, params_cell)

    # Multi-cell
    # *** KEY ADJUSTMENT #3: bigger I_app_val => 300. 
    params_multi= {
        'Ng':5,
        'nvar':23,
        'g_gap':0.02,
        'dx':1.0,
        'cm':membrane_capacitance,
        'cell_params': params_cell,
        'I_app_val':300.0,   # <--- bigger stimulation
        'stim_cell':2,
        'stim_start':50.0,   # can also start earlier if you want
        'stim_end':150.0
    }
    print("[DEBUG] Starting multi-cell run...")
    sol_multi= run_multi_cell_simulation(params_multi, t_span=(0,300), dt=1.0)
    t_multi= sol_multi.t
    Y_multi= sol_multi.y
    print("[DEBUG] Plotting multi-cell results...")

    plt.figure()
    for i_cell in range(params_multi['Ng']):
        idxV= i_cell* params_multi['nvar']
        plt.plot(t_multi, Y_multi[idxV,:], label="Cell %d" % i_cell)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Multi-cell Pericyte Model (BDF solver, bigger I_app, reduced Kir)")
    plt.legend()
    plt.show()

    print("[DEBUG] All done!")
