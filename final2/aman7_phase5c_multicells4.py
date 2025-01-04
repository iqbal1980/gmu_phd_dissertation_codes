#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aman7_phase5c_multicells4_antiNaN.py

This version prevents Inf/NaN by:
1) Clamping the IP3R gating states between [0,1].
2) Clamping Ca_i, Ca_ER, Ca_m, IP3 to a max range.
3) Clamping IP3R flux if it exceeds a threshold.
4) Using method='LSODA' instead of 'BDF' for better numerical stability in some stiff+nonstiff systems.

Author: [Your Name]
Date: [Date]
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# 1) GLOBALS & SAFE OPS
###############################################################################
FARADAY_CONSTANT = 96485.0
GAS_CONSTANT     = 8314.0
TEMPERATURE      = 310.0

MEMBRANE_CAPACITANCE = 0.94  # pF
CELL_VOLUME    = 2e-12
V_CYTO         = CELL_VOLUME
ER_VOLUME      = CELL_VOLUME * 0.2
Z_CA           = 2.0

# Buffers
BSCYT, AKSCYT = 225.0, 0.1
BSER,  AKSER  = 2000.0, 1.0
BM,    AKM    = 111.0, 0.123

# PMCA
P1, P2 = 0.1, 0.01

# For exponent clamping
EXP_CAP = 50.0

def safe_exp(x, cap=EXP_CAP):
    if x > cap:
        x = cap
    elif x < -cap:
        x = -cap
    return math.exp(x)

###############################################################################
# 2) REVERSAL POTENTIALS
###############################################################################
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_in, Na_out, Na_in, Cl_out, Cl_in):
    valK  = (GAS_CONSTANT*TEMPERATURE)/(1.0*FARADAY_CONSTANT)
    revK  = valK* math.log(K_out/K_in)

    valCa= (GAS_CONSTANT*TEMPERATURE)/(2.0*FARADAY_CONSTANT)
    revCa= valCa* math.log(Ca_out/Ca_in)

    valNa= (GAS_CONSTANT*TEMPERATURE)/(1.0*FARADAY_CONSTANT)
    revNa= valNa* math.log(Na_out/Na_in)

    valCl= (GAS_CONSTANT*TEMPERATURE)/(1.0*FARADAY_CONSTANT)
    revCl= valCl* math.log(Cl_out/Cl_in)
    return revK, revCa, revNa, revCl

###############################################################################
# 3) BUFFER FACTORS & IP3R
###############################################################################
def calculate_buffering_factors(Ca_i, Ca_ER):
    Ca_i  = max(Ca_i,1e-12)
    Ca_ER = max(Ca_ER,1e-12)
    top1= (BSCYT*AKSCYT)/((AKSCYT+Ca_i)**2)
    top2= (BM*AKM)/((AKM+Ca_i)**2)
    beta_cyt= 1.0/(1.0+ top1+ top2)

    top3= (BSER*AKSER)/((AKSER+ Ca_ER)**2)
    top4= (BM*AKM)/((AKM+ Ca_ER)**2)
    beta_er= 1.0/(1.0+ top3+ top4)
    return beta_cyt, beta_er

# De Young-Keizer IP3R
A1, A2, A3, A4, A5 = 400.0, 0.2, 400.0, 0.2, 20.0
D1, D2, D3, D4, D5 = 0.13, 1.049, 0.9434, 0.1445, 0.08234
# If your v1 is too large => big flux => overflow. Let's reduce it from 90 -> 10
V1= 10.0  
B1= A1*D1
B2= A2*D2
B3= A3*D3
B4= A4*D4
B5= A5*D5

def clamp01(x):
    """Clamp a gating variable x into [0, 1]."""
    return max(0.0, min(1.0, x))

def calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111):
    Ca_i= max(Ca_i,1e-12)
    IP3 = max(IP3,1e-12)

    f1  = B5*x010 - A5*Ca_i*x000
    f2  = B1*x100 - A1*IP3*x000
    f3  = B4*x001 - A4*Ca_i*x000
    f4  = B5*x110 - A5*Ca_i*x100
    f5  = B2*x101 - A2*Ca_i*x100
    f6  = B1*x110 - A1*IP3*x010
    f7  = B4*x011 - A4*Ca_i*x010
    f8  = B5*x011 - A5*Ca_i*x001
    f9  = B3*x101 - A3*IP3*x001
    f10 = B2*x111 - A2*Ca_i*x110
    f11 = B5*x111 - A5*Ca_i*x101
    f12 = B3*x111 - A3*IP3*x011

    dx000= f1 + f2 + f3
    dx100= f4 + f5 - f2
    dx010= -f1+ f6 + f7
    dx001= f8 - f3+ f9
    dx110= -f4- f6+ f10
    dx101= f11- f9- f5
    dx011= -f8- f7+ f12
    dx111= -f11- f12- f10

    return (dx000,dx100,dx010,dx001,dx110,dx101,dx011,dx111)

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    Ca_i= max(Ca_i,1e-12)
    Ca_ER= max(Ca_ER,1e-12)
    flux= V1* (x110**3)* (Ca_ER- Ca_i)
    # clamp flux to avoid huge blowups
    if abs(flux) > 1e4:
        flux= max(min(flux, 1e4), -1e4)
    return flux

###############################################################################
# 4) MITO & IP3 ODE
###############################################################################
PMITO=2.776e-20
PSI_MV= 160.0
PSI_VOLTS= PSI_MV/1000.0
ALPHM= 0.2
ALPHI= 1.0
VNC= 1.836
ANA=5000.0
AKNA=8000.0
AKCA=8.0

def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i= max(Ca_i,1e-12)
    vu=1540000.0
    vm=2200000.0
    aku=0.303
    akmp=0.14
    aru=1.8
    arm=2.1
    u4= (vu*(Ca_i**aru))/(Ca_i**aru+ aku**aru)
    u5= (vm*(Ca_i**arm))/(Ca_i**arm+ akmp**arm)
    cJpmca= (dpmca*u4+ (1.0- dpmca)* u5)/6.6253e5
    return cJpmca, u4, u5

def calculate_mito_fluxes(Ca_i, Ca_m):
    Ca_i= max(Ca_i,1e-12)
    Ca_m= max(Ca_m,1e-12)
    bb= (Z_CA* PSI_VOLTS* FARADAY_CONSTANT)/(GAS_CONSTANT* TEMPERATURE)
    e_neg= safe_exp(-bb)
    if math.isinf(e_neg) or math.isnan(e_neg):
        e_neg= 0.0

    Vmito= CELL_VOLUME*0.08
    top_uni= (ALPHM* Ca_i* e_neg) - (ALPHI* Ca_m)
    J_uni= (PMITO/Vmito)* bb*(top_uni/(e_neg-1))

    som= (ANA**3)* Ca_m/( (AKNA**3)* AKCA )
    soe= (ANA**3)* Ca_i/( (AKNA**3)* AKCA )
    B= safe_exp( 0.5* PSI_VOLTS* Z_CA* FARADAY_CONSTANT/(GAS_CONSTANT* TEMPERATURE))
    denom= 1 + (ANA**3/(AKNA**3)) + Ca_m/AKCA + som + (ANA**3/(AKNA**3)) + Ca_i/AKCA + soe
    J_nc= VNC*( B* som - (1.0/B)* soe )/ denom
    return J_uni, J_nc

# IP3 ODE
PROD_IP3=0.01
V2IP3=  12.5
AK2IP3= 6.0
V3IP3=  0.9
AK3IP3= 0.1
AK4IP3= 1.0

def ip3_ode(Ca_i, IP3):
    Ca_i= max(Ca_i,1e-12)
    IP3=  max(IP3,1e-12)
    term1= PROD_IP3
    denom2= (1.0+ (AK2IP3/IP3))
    term2= V2IP3/ denom2

    denom3a= (1.0+ (AK3IP3/IP3))
    denom3b= (1.0+ (AK4IP3/Ca_i))
    term3= (V3IP3/ denom3a)* (1.0/ denom3b)

    dIP3dt= term1 - term2 - term3
    new_IP3= IP3+ dIP3dt
    if new_IP3>0.1:
        dIP3dt= (0.1- IP3)
    if new_IP3<1e-12:
        dIP3dt= (1e-12- IP3)
    return dIP3dt

###############################################################################
# 5) CURRENTS
###############################################################################
def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, p):
    Ca_i= max(Ca_i,1e-12)
    Ca_ER= max(Ca_ER,1e-12)

    # CaL gating
    argA= -(V- p['activation_midpoint_CaL'])/ p['activation_slope_CaL']
    d_inf= 1.0/(1.0+ safe_exp(argA))

    argB= (V- p['inactivation_midpoint_CaL'])/ p['inactivation_slope_CaL']
    partB= 1.0/(1.0+ safe_exp(argB))

    argC= (p['voltage_shift_CaL']- V)/ p['slope_factor_CaL']
    partC= p['amplitude_factor_CaL']/(1.0+ safe_exp(argC))
    f_inf= partB+ partC

    # Kir
    shiftKir= p['voltage_shift_Kir61']
    slopeKir= p['voltage_slope_Kir61']
    denomKir= 1.0+ safe_exp((V- p['reversal_potential_K']- shiftKir)/ slopeKir)
    I_Kir61= ( p['conductance_Kir61'] * atp * math.sqrt(p['K_out']/ p['reference_K'])
               * (V- p['reversal_potential_K'])/ denomKir )

    # TRPC
    I_TRPC1= p['conductance_TRPC1']*( V- p['reversal_potential_Ca'] )

    # CaCC
    fracCa= Ca_i/( Ca_i+ p['calcium_activation_threshold_CaCC'] )
    I_CaCC= p['conductance_CaCC']* fracCa* (V- p['reversal_potential_Cl'])

    # CaL
    I_CaL= p['conductance_CaL']* d_inf* f_inf* (V- p['reversal_potential_Ca'])

    # Leak
    I_leak= p['conductance_leak']*( V- p['reversal_potential_K'])

    # PMCA
    cJpmca,_,_= calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA= cJpmca* Z_CA* FARADAY_CONSTANT* CELL_VOLUME* 1e6

    # SERCA
    J_SERCA= p['k_serca']*(Ca_i**2)/(Ca_i**2+ p['Km_serca']**2)
    # ER leak
    J_ER_leak= p['leak_rate_er']*(Ca_ER- Ca_i)
    # NCX
    iNcxNum= (p['Na_in']**3)/(p['Na_in']**3+ 87.5**3)
    iNcxDen= (p['Ca_out']/(p['Ca_out']+ 1.0))
    I_NCX= p['k_ncx']* iNcxNum* iNcxDen

    # IP3R flux
    J_IP3R1= calculate_ip3r_flux(x110,  Ca_ER, Ca_i)* p['conductance_IP3R1']
    J_IP3R2= calculate_ip3r_flux(x1102, Ca_ER, Ca_i)* p['conductance_IP3R2']
    J_IP3R= J_IP3R1+ J_IP3R2

    # RyR
    partRy= Ca_i/( Ca_i+ 0.3)
    J_RyR= p['conductance_RyR']* partRy* (Ca_ER- Ca_i)

    return (
        I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak,
        d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR
    )

###############################################################################
# 6) SINGLE-CELL MODEL
###############################################################################
def single_cell_model_noNaN(t, y, p):
    """
    We clamp everything inside the function to avoid runaway states.
    """
    # parse
    (V, Ca_i, atp, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m,
     x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)= y

    # clamp states inside
    Ca_i   = max(1e-12, min(10.0, Ca_i))       # clamp Ca_i
    Ca_ER  = max(1e-12, min(10.0, Ca_ER))      # clamp ER
    Ca_m   = max(1e-12, min(10.0, Ca_m))       # clamp Mito Ca
    IP3    = max(1e-12, min(0.1, IP3))         # clamp IP3
    # clamp gating states: 0..1
    x000   = clamp01(x000)
    x100   = clamp01(x100)
    x010   = clamp01(x010)
    x001   = clamp01(x001)
    x110   = clamp01(x110)
    x101   = clamp01(x101)
    x011   = clamp01(x011)
    x111   = clamp01(x111)
    x0002  = clamp01(x0002)
    x1002  = clamp01(x1002)
    x0102  = clamp01(x0102)
    x0012  = clamp01(x0012)
    x1102  = clamp01(x1102)
    x1012  = clamp01(x1012)
    x0112  = clamp01(x0112)
    x1112  = clamp01(x1112)

    # Now compute derivatives
    (dx000,dx100,dx010,dx001,dx110,dx101,dx011,dx111) = calculate_ip3r_states(
        Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111
    )
    (dx0002,dx1002,dx0102,dx0012,dx1102,dx1012,dx0112,dx1112) = calculate_ip3r_states(
        Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112
    )

    (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak,
     d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR)= \
         calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, x1102, IP3, p)

    dV_dt= (
        - I_Kir61 - I_TRPC1 - I_CaCC - I_CaL
        - I_leak  - I_PMCA   - I_NCX
    )/ MEMBRANE_CAPACITANCE

    Ca_influx= -I_CaL/(2.0* Z_CA* FARADAY_CONSTANT* CELL_VOLUME)
    Ca_efflux= (-I_PMCA - I_NCX)/(2.0* Z_CA* FARADAY_CONSTANT* CELL_VOLUME)
    beta_cyt, beta_er= calculate_buffering_factors(Ca_i, Ca_ER)

    dCa_dt= beta_cyt*( Ca_influx+ Ca_efflux+ J_ER_leak+ J_IP3R+ J_RyR - J_SERCA)
    dCa_ER_dt= beta_er*( (V_CYTO/ER_VOLUME)*( J_SERCA- J_ER_leak- J_IP3R- J_RyR))

    w1= P1* Ca_i
    w2= P2
    taom= 1.0/(w1+ w2)
    dpmcainf= w2/(w1+ w2)
    ddpmca_dt= (dpmcainf- dpmca)/ taom

    dIP3_dt= ip3_ode(Ca_i, IP3)

    J_uni, J_nc= calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt= J_uni- J_nc

    return [
        dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
        dx000,dx100,dx010,dx001, dx110,dx101,dx011,dx111,
        dIP3_dt, dCa_m_dt,
        dx0002,dx1002,dx0102,dx0012,dx1102,dx1012,dx0112,dx1112
    ]

###############################################################################
# 7) SINGLE-CELL RUN & PLOT
###############################################################################
def run_single_cell_sim(params):
    # init
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
    t_span= (0, params['simulation_duration'])
    t_eval= np.linspace(0, params['simulation_duration'], params['time_points'])

    def rhs(t, Y):
        return single_cell_model_noNaN(t, Y, params)

    print("[INFO] Single-cell simulation started with method='LSODA' ...")
    start_time= time.perf_counter()
    sol= solve_ivp(rhs, t_span, y0,
                   t_eval=t_eval,
                   method='LSODA',  # more robust
                   max_step=0.5,    # smaller step => more stable
                   rtol=1e-6,
                   atol=1e-9)
    end_time= time.perf_counter()
    print(f"[INFO] Single-cell simulation completed in {end_time - start_time:.4f} s.")

    return sol

def plot_single_cell_results(sol, params):
    t= sol.t
    Y= sol.y

    V= Y[0,:]
    Ca_i= Y[1,:]
    dpmca= Y[3,:]
    Ca_ER= Y[4,:]
    x000= Y[5,:]
    x100= Y[6,:]
    x010= Y[7,:]
    x001= Y[8,:]
    x110= Y[9,:]
    x101= Y[10,:]
    x011= Y[11,:]
    x111= Y[12,:]
    IP3=  Y[13,:]
    Ca_m= Y[14,:]
    x0002= Y[15,:]
    x1002= Y[16,:]
    x0102= Y[17,:]
    x0012= Y[18,:]
    x1102=Y[19,:]
    x1012=Y[20,:]
    x0112=Y[21,:]
    x1112=Y[22,:]

    fig, axs= plt.subplots(5,1, figsize=(10,18), sharex=True)
    fig.suptitle("Single-Cell (Anti-NaN) LSODA", fontsize=14)

    axs[0].plot(t, V, 'b')
    axs[0].set_ylabel("Vm (mV)")
    axs[0].set_title("Membrane Potential")
    axs[0].grid(True)

    axs[1].plot(t, Ca_i, 'r', label="Ca_i")
    axs[1].plot(t, Ca_ER,'g', label="Ca_ER")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title("Cytosolic & ER Ca")

    axs[2].plot(t, IP3, 'm')
    axs[2].set_ylabel("IP3 (mM)")
    axs[2].grid(True)
    axs[2].set_title("IP3 Over Time")

    axs[3].plot(t, Ca_m, 'brown')
    axs[3].set_ylabel("Ca_m (mM)")
    axs[3].grid(True)
    axs[3].set_title("Mitochondrial Ca")

    sum_ip3r1= x000+ x100+ x010+ x001+ x110+ x101+ x011+ x111
    sum_ip3r2= x0002+ x1002+ x0102+ x0012+ x1102+ x1012+ x0112+ x1112
    axs[4].plot(t, sum_ip3r1, 'k', label="IP3R1 sum")
    axs[4].plot(t, sum_ip3r2, 'c', label="IP3R2 sum")
    axs[4].grid(True)
    axs[4].legend()
    axs[4].set_title("IP3R States Sums")

    axs[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

###############################################################################
# 8) MULTI-CELL PDE
###############################################################################
def multi_cell_rhs_antiNaN(t, Y, params_multi):
    Ng   = params_multi["Ng"]
    nvar = params_multi["nvar"]
    g_gap= params_multi["g_gap"]
    dx   = params_multi["dx"]
    cm   = params_multi["mem_cap"]
    cpar = params_multi["cell_params"]

    I_app_val= params_multi["I_app_val"]
    stim_cell= params_multi["stim_cell"]
    s_start= params_multi["stim_start"]
    s_end  = params_multi["stim_end"]

    dYdt= np.zeros_like(Y)
    for i in range(Ng):
        idx0= i*nvar
        idx1= idx0+ nvar
        y_i= Y[idx0: idx1]

        # single cell derivative
        dy_i= single_cell_model_noNaN(t, y_i, cpar)
        V_i= y_i[0]
        dV_i= dy_i[0]

        if i==0:
            V_left= V_i
        else:
            V_left= Y[(i-1)* nvar + 0]
        if i== Ng-1:
            V_right= V_i
        else:
            V_right= Y[(i+1)* nvar + 0]

        I_gj= g_gap*( (V_left + V_right - 2* V_i)/(dx**2))
        dV_multi= dV_i+ (I_gj/cm)

        if (s_start <= t <= s_end) and (i== stim_cell):
            dV_multi+= (I_app_val/cm)

        dy_i[0]= dV_multi
        dYdt[idx0:idx1]= dy_i

    return dYdt

def run_multi_cell_sim(params_multi):
    Ng   = params_multi["Ng"]
    nvar = params_multi["nvar"]
    t0, tf= params_multi["t_span"]
    dt    = params_multi["dt"]
    cpar  = params_multi["cell_params"]

    # build Y0
    ic_single= [
        cpar['initial_voltage'],
        cpar['initial_calcium'],
        cpar['initial_atp'],
        cpar['initial_dpmca'],
        cpar['Ca_ER_initial'],
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035,
        0.1,
        0.0001,
        0.27,0.039,0.29,0.17,0.042,0.0033,0.18,0.0035
    ]
    Y0= np.tile(ic_single, Ng)

    def rhs(t, Y):
        return multi_cell_rhs_antiNaN(t, Y, params_multi)

    t_eval= np.arange(t0, tf+ dt, dt)
    print("[INFO] Multi-cell PDE-like run (anti-NaN, method='LSODA') ...")
    start_t= time.perf_counter()
    sol= solve_ivp(rhs, (t0, tf), Y0, t_eval=t_eval,
                   method='LSODA', max_step=0.5,
                   rtol=1e-6, atol=1e-9)
    end_t= time.perf_counter()
    print(f"[INFO] Multi-cell simulation finished in {end_t- start_t:.4f}s.")
    return sol

def plot_multi_cell_results(sol, params_multi):
    t= sol.t
    Y= sol.y
    Ng   = params_multi["Ng"]
    nvar = params_multi["nvar"]

    plt.figure(figsize=(10,6))
    for i in range(Ng):
        idxV= i*nvar + 0
        Vm_i= Y[idxV,:]
        plt.plot(t, Vm_i, label=f"Cell {i}")
    plt.title(f"Multi-cell PDE: V (Stim cell={params_multi['stim_cell']})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    for i in range(Ng):
        idxCa= i*nvar + 1
        Ca_i= Y[idxCa,:]
        plt.plot(t, Ca_i, label=f"Cell {i} Ca_i")
    plt.title("Multi-cell PDE: Cytosolic Ca")
    plt.xlabel("Time (ms)")
    plt.ylabel("Ca_i (mM)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()

###############################################################################
# 9) MAIN
###############################################################################
if __name__=="__main__":
    cell_params= {
        'K_out':6.26, 'K_in':140.0, 'Ca_out':2.0, 'Ca_in':1e-4,
        'Na_out':140.0, 'Na_in':15.38, 'Cl_out':110.0, 'Cl_in':9.65,

        'conductance_Kir61':0.01,
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
        'voltage_slope_Kir61':6.0,
        'voltage_shift_Kir61':15.0,

        'activation_midpoint_CaL':-40.0,
        'activation_slope_CaL':4.0,
        'inactivation_midpoint_CaL':-45.0,
        'inactivation_slope_CaL':5.0,
        'voltage_shift_CaL':50.0,
        'slope_factor_CaL':20.0,
        'amplitude_factor_CaL':0.6,

        'simulation_duration':200.0,
        'time_points':2001,

        'initial_voltage':-70.0,
        'initial_calcium':1e-4,
        'initial_atp':4.4,
        'initial_dpmca':1.0,
        'Ca_ER_initial':0.5
    }

    # Reversals
    rk, rca, rna, rcl= calculate_reversal_potentials(
        cell_params['K_out'], cell_params['K_in'],
        cell_params['Ca_out'], cell_params['Ca_in'],
        cell_params['Na_out'], cell_params['Na_in'],
        cell_params['Cl_out'], cell_params['Cl_in']
    )
    cell_params['reversal_potential_K'] =  rk*1e3
    cell_params['reversal_potential_Ca']= rca*1e3
    cell_params['reversal_potential_Na']= rna*1e3
    cell_params['reversal_potential_Cl']= rcl*1e3

    # Single cell run
    sol_single= run_single_cell_sim(cell_params)
    plot_single_cell_results(sol_single, cell_params)

    # Multi cell
    params_multi= {
        "Ng":5,
        "nvar":23,
        "g_gap":0.02,
        "dx":1.0,
        "mem_cap": MEMBRANE_CAPACITANCE,
        "cell_params": cell_params,

        "I_app_val": -50.0,
        "stim_cell":2,
        "stim_start":50.0,
        "stim_end":150.0,

        "t_span":(0.0, 300.0),
        "dt":1.0
    }
    sol_multi= run_multi_cell_sim(params_multi)
    plot_multi_cell_results(sol_multi, params_multi)

    print("[INFO] Done. If you still see Inf/NaN, consider lowering v1 further or max_step further.")
