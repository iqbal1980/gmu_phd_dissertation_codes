#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multicell_pericyte_full_corrected_mirror_with_investigation.py

This script:
  1) Runs the "corrected" 23-state pericyte PDE-like model TWICE:
       (a) I_app_val = +70 pA  ("sail" shape)
       (b) I_app_val = -70 pA  ("mirror" shape)
     and *saves* plots as images (rather than displaying them).

  2) Then, it tries a "steady-state" approach to find a resting state
     with I_app=0 for 't_pre' ms, and then re-runs with +70 or -70 from that
     new "rest" state, also saving plots.

  3) Then, it does a simple param-sweep example: varying the CaL conductance
     by a few factors to see if that modifies the "mirror" shape.

Paper reference:
  "The electrotonic architecture of the retinal microvasculature:
   modulation by angiotensin II" 
  T. Zhang, D.M. Wu, G.-z. Xu, D.G. Puro
  J Physiol. 2011 589(Pt 9):2383–2399. 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# 1) Indices for the 23-state single-cell model
###############################################################################
STATE_V       = 0
STATE_CAI     = 1
STATE_ATP     = 2
STATE_DPMCA   = 3
STATE_CAER    = 4
STATE_X000    = 5
STATE_X100    = 6
STATE_X010    = 7
STATE_X001    = 8
STATE_X110    = 9
STATE_X101    = 10
STATE_X011    = 11
STATE_X111    = 12
STATE_IP3     = 13
STATE_CAM     = 14
STATE_X0002   = 15
STATE_X1002   = 16
STATE_X0102   = 17
STATE_X0012   = 18
STATE_X1102   = 19
STATE_X1012   = 20
STATE_X0112   = 21
STATE_X1112   = 22

###############################################################################
# 2) Single-Cell ODE: "23-state" detailed pericyte model
###############################################################################
def single_cell_rhs_full(t, y_cell, p, cell_index):
    """
    Returns dy/dt for one pericyte with 23 states.
    Also injects patch-clamp current if cell == p['stim_cell'] and
    t in [p['stim_start'], p['stim_end']].
    """
    # Unpack
    V      = y_cell[STATE_V]
    Ca_i   = y_cell[STATE_CAI]
    atp    = y_cell[STATE_ATP]
    dpmca  = y_cell[STATE_DPMCA]
    Ca_ER  = y_cell[STATE_CAER]
    x000   = y_cell[STATE_X000]
    x100   = y_cell[STATE_X100]
    x010   = y_cell[STATE_X010]
    x001   = y_cell[STATE_X001]
    x110   = y_cell[STATE_X110]
    x101   = y_cell[STATE_X101]
    x011   = y_cell[STATE_X011]
    x111   = y_cell[STATE_X111]
    IP3    = y_cell[STATE_IP3]
    Ca_m   = y_cell[STATE_CAM]
    x0002  = y_cell[STATE_X0002]
    x1002  = y_cell[STATE_X1002]
    x0102  = y_cell[STATE_X0102]
    x0012  = y_cell[STATE_X0012]
    x1102  = y_cell[STATE_X1102]
    x1012  = y_cell[STATE_X1012]
    x0112  = y_cell[STATE_X0112]
    x1112  = y_cell[STATE_X1112]

    # Avoid negative Ca
    Cai   = max(Ca_i, 1e-12)
    CaER  = max(Ca_ER,1e-12)

    # A) Buffering
    def buffering_factors(Ccyt, Cer):
        Bscyt = p['Bscyt']
        aKscyt= p['aKscyt']
        Bm    = p['Bm']
        aKm   = p['aKm']
        Bser  = p['Bser']
        aKser = p['aKser']

        beta_cyt= 1.0/(
            1.0
            + (Bscyt*aKscyt)/((aKscyt + Ccyt)**2)
            + (Bm*aKm)/((aKm + Ccyt)**2)
        )
        beta_er= 1.0/(
            1.0
            + (Bser*aKser)/((aKser + Cer)**2)
            + (Bm*aKm)/((aKm + Cer)**2)
        )
        return beta_cyt, beta_er

    beta_cyt, beta_er= buffering_factors(Cai, CaER)

    # B) IP3R gating
    a1,a2,a3,a4,a5 = p['a1'], p['a2'], p['a3'], p['a4'], p['a5']
    d1,d2,d3,d4,d5 = p['d1'], p['d2'], p['d3'], p['d4'], p['d5']
    b1= a1*d1; b2= a2*d2; b3= a3*d3; b4= a4*d4; b5= a5*d5

    def calc_ip3r_states(Cav, IP3v,
                         X000, X100, X010, X001,
                         X110, X101, X011, X111):
        Cav   = max(Cav,1e-12)
        IP3v  = max(IP3v,1e-12)
        f1    = b5*X010  - a5*Cav*X000
        f2    = b1*X100  - a1*IP3v*X000
        f3    = b4*X001  - a4*Cav*X000
        f4    = b5*X110  - a5*Cav*X100
        f5    = b2*X101  - a2*Cav*X100
        f6    = b1*X110  - a1*IP3v*X010
        f7    = b4*X011  - a4*Cav*X010
        f8    = b5*X011  - a5*Cav*X001
        f9    = b3*X101  - a3*IP3v*X001
        f10   = b2*X111  - a2*Cav*X110
        f11   = b5*X111  - a5*Cav*X101
        f12   = b3*X111  - a3*IP3v*X011

        dx000= f1+ f2+ f3
        dx100= f4+ f5- f2
        dx010= -f1 + f6+ f7
        dx001= f8- f3 + f9
        dx110= -f4- f6 + f10
        dx101= f11- f9- f5
        dx011= -f8- f7 + f12
        dx111= -f11- f12- f10
        return dx000,dx100,dx010,dx001,dx110,dx101,dx011,dx111

    # IP3R1
    dx000,dx100,dx010,dx001,dx110,dx101,dx011,dx111 = \
        calc_ip3r_states(Cai, IP3, x000,x100,x010,x001,
                         x110,x101,x011,x111)
    # IP3R2
    dx0002,dx1002,dx0102,dx0012,dx1102,dx1012,dx0112,dx1112 = \
        calc_ip3r_states(Cai, IP3, x0002,x1002,x0102,x0012,
                         x1102,x1012,x0112,x1112)

    # C) IP3 ODE
    def ip3_ode(Cav, I3):
        prd  = p['prodip3']
        V2i  = p['V2ip3']
        ak2i = p['ak2ip3']
        V3i  = p['V3ip3']
        ak3i = p['ak3ip3']
        ak4i = p['ak4ip3']
        Cav  = max(Cav,1e-12)
        I3   = max(I3,1e-12)
        term1= prd
        term2= V2i/(1.0+(ak2i/I3))
        term3= (V3i/(1.0+(ak3i/I3))) * (1.0/(1.0+(ak4i/Cav)))
        dI3dt= term1 - term2 - term3
        new_I3= I3 + dI3dt
        if new_I3>0.01:
            dI3dt= (0.01 - I3)
        if new_I3<1e-12:
            dI3dt= (1e-12 - I3)
        return dI3dt

    dIP3= ip3_ode(Cai, IP3)

    # D) Mito fluxes
    def calc_mito_fluxes(Cav, Camv):
        z   = p['z']
        F   = p['faraday_constant']
        R   = p['gas_constant']
        T   = p['temperature']
        psi = p['psi_volts']
        alphm= p['alphm']
        alphi= p['alphi']
        Pmito= p['Pmito']
        Vmito= p['Vmito']
        aNa  = p['aNa']
        akna = p['akna']
        akca = p['akca']
        Vnc  = p['Vnc']

        Cav  = max(Cav,1e-12)
        Camv = max(Camv,1e-12)
        bb   = (z*psi*F)/(R*T)
        e_neg= math.exp(-bb)
        if math.isinf(e_neg) or math.isnan(e_neg):
            e_neg=0.0
        top_uni= (alphm*Cav* e_neg) - (alphi*Camv)
        denom_= ( e_neg -1.0 )
        if abs(denom_)<1e-12:
            denom_=1e-12
        J_uni= (Pmito/Vmito)* bb*( top_uni/denom_ )

        som = (aNa**3)* Camv / ((akna**3)* akca)
        soe = (aNa**3)* Cav  / ((akna**3)* akca)
        B   = math.exp(0.5*psi*z*F/(R*T))
        denom= (1.0+(aNa**3/(akna**3))+Camv/akca+som
                +(aNa**3/(akna**3))+Cav/akca+soe)
        J_nc= Vnc*( B*som - (1.0/B)*soe )/ denom
        return J_uni, J_nc

    J_uni, J_nc= calc_mito_fluxes(Cai, Ca_m)

    # E) PMCA flux
    def calc_pmca_flux(Cav, dpm):
        vu   = p['vu']
        vm   = p['vm']
        aku  = p['aku']
        akmp = p['akmp']
        aru  = p['aru']
        arm  = p['arm']
        Cav  = max(Cav,1e-12)
        u4= (vu*(Cav**aru)) /(Cav**aru+ aku**aru)
        u5= (vm*(Cav**arm)) /(Cav**arm+ akmp**arm)
        cJpmca= (dpm*u4 + (1.0-dpm)*u5)/(6.6253e5)
        return cJpmca

    cJpmca= calc_pmca_flux(Cai, dpmca)

    # F) Ion Currents in Membrane eqn
    I_Kir61= (
        p['conductance_Kir61']* atp
        * math.sqrt(p['K_out']/ p['reference_K'])
        * (V - p['reversal_potential_K'])
        / (1.0+ math.exp(( V- p['reversal_potential_K']
                           - p['voltage_shift_Kir61']) / p['voltage_slope_Kir61']))
    )
    I_TRPC1= p['conductance_TRPC1']*(V- p['reversal_potential_Ca'])
    fracCa= Cai/(Cai+ p['calcium_activation_threshold_CaCC'])
    I_CaCC= p['conductance_CaCC']* fracCa*(V- p['reversal_potential_Cl'])

    def CaL_gating(Vv):
        d_inf= 1.0/(1.0+ math.exp(-(Vv- p['activation_midpoint_CaL'])
                                  / p['activation_slope_CaL']))
        f_tmp= 1.0/(1.0+ math.exp(
            (Vv- p['inactivation_midpoint_CaL'])/ p['inactivation_slope_CaL']))
        f_tmp+= p['amplitude_factor_CaL']/ (1.0+ math.exp(
                   (p['voltage_shift_CaL']- Vv)/ p['slope_factor_CaL']))
        return d_inf, f_tmp

    d_inf, f_inf= CaL_gating(V)
    I_CaL= p['conductance_CaL']* d_inf* f_inf*(V- p['reversal_potential_Ca'])
    I_leak= p['conductance_leak']*(V- p['reversal_potential_K'])
    I_PMCA= cJpmca* p['z']* p['faraday_constant']* p['Vcyto']*1e6
    I_NCX= (
        p['k_ncx']
        *(p['Na_in']**3/(p['Na_in']**3+ 87.5**3))
        *(p['Ca_out']/(p['Ca_out']+ 1.0))
    )

    I_sum= I_Kir61+ I_TRPC1+ I_CaCC+ I_CaL+ I_leak+ I_PMCA+ I_NCX

    # Patch clamp current
    I_app= 0.0
    if (cell_index == p['stim_cell']) and (p['stim_start'] <= t <= p['stim_end']):
        I_app= p['I_app_val']

    # dV/dt
    dVdt= - (I_sum + I_app)/ p['membrane_capacitance']

    # G) SERCA, ER leak, IP3R, RyR
    J_SERCA= p['k_serca']*(Cai**2)/(Cai**2+ p['Km_serca']**2)
    J_ER_leak= p['leak_rate_er']*(CaER- Cai)
    J_IP3R1= p['conductance_IP3R1']*(x110**3)*(CaER- Cai)
    J_IP3R2= p['conductance_IP3R2']*(x1102**3)*(CaER- Cai)
    J_IP3R= J_IP3R1+ J_IP3R2
    J_RyR= p['conductance_RyR']*(Cai/(Cai+ 0.3))*(CaER- Cai)

    Ca_influx= -I_CaL/(2.0* p['faraday_constant']* p['cell_volume'])
    Ca_efflux= -(I_PMCA + I_NCX)/(2.0* p['faraday_constant']* p['cell_volume'])
    dCa_dt= beta_cyt*(Ca_influx+ Ca_efflux+ J_ER_leak+ J_IP3R+ J_RyR- J_SERCA)

    dCa_ER_dt= beta_er*(
        (p['Vcyto']/ p['ER_volume'])
        *( J_SERCA- J_ER_leak- J_IP3R- J_RyR )
    )

    w1= p['p1']* Cai
    w2= p['p2']
    taom= 1.0/(w1+ w2)
    dpmcainf= w2/(w1+ w2)
    ddpmca= (dpmcainf- dpmca)/ taom

    # Mito
    dCa_m= (J_uni- J_nc)

    dy= np.zeros(23, dtype=float)
    dy[STATE_V]      = dVdt
    dy[STATE_CAI]    = dCa_dt
    dy[STATE_ATP]    = 0.0
    dy[STATE_DPMCA]  = ddpmca
    dy[STATE_CAER]   = dCa_ER_dt
    dy[STATE_X000]   = dx000
    dy[STATE_X100]   = dx100
    dy[STATE_X010]   = dx010
    dy[STATE_X001]   = dx001
    dy[STATE_X110]   = dx110
    dy[STATE_X101]   = dx101
    dy[STATE_X011]   = dx011
    dy[STATE_X111]   = dx111
    dy[STATE_IP3]    = dIP3
    dy[STATE_CAM]    = dCa_m
    dy[STATE_X0002]  = dx0002
    dy[STATE_X1002]  = dx1002
    dy[STATE_X0102]  = dx0102
    dy[STATE_X0012]  = dx0012
    dy[STATE_X1102]  = dx1102
    dy[STATE_X1012]  = dx1012
    dy[STATE_X0112]  = dx0112
    dy[STATE_X1112]  = dx1112

    return dy

###############################################################################
# 3) PDE-like approach
###############################################################################
def multicell_rhs(t, Y, p):
    N   = p['N_cells']
    nvar= 23
    g_gap= p['g_gap']
    dx  = p['dx']
    cm  = p['membrane_capacitance']

    dYdt= np.zeros_like(Y)
    for i in range(N):
        idx0= i*nvar
        idx1= idx0+ nvar
        y_i= Y[idx0: idx1]

        # Single cell
        dy_i= single_cell_rhs_full(t, y_i, p, cell_index=i)

        # Gap junction on V
        V_i= y_i[STATE_V]
        if i==0:
            V_left= V_i
        else:
            V_left= Y[(i-1)*nvar+ STATE_V]
        if i==N-1:
            V_right= V_i
        else:
            V_right= Y[(i+1)*nvar+ STATE_V]

        I_gj= g_gap*((V_left+ V_right - 2.0*V_i)/(dx**2))
        dVdt_original= dy_i[STATE_V]
        dVdt_with_gj= dVdt_original + (I_gj/ cm)
        dy_i[STATE_V]= dVdt_with_gj

        dYdt[idx0: idx1]= dy_i

    return dYdt

###############################################################################
# 4) Run multicell sim
###############################################################################
def run_multicell_sim(p, initial_Y=None):
    # Build initial condition
    N= p['N_cells']
    nvar= 23
    if initial_Y is None:
        # normal approach
        bigY0= []
        def init_single_cell():
            arr= np.zeros(nvar)
            arr[STATE_V]   = -70.0
            arr[STATE_CAI] = 1e-4
            arr[STATE_ATP] = 4.4
            arr[STATE_DPMCA]= 1.0
            arr[STATE_CAER] = 0.5
            arr[STATE_IP3]  = 0.1
            arr[STATE_CAM]  = 1e-4
            # IP3R1 gating
            x000=0.27; x100=0.039; x010=0.29; x001=0.17
            x110=0.042; x101=0.0033; x011=0.18; x111=0.0035
            tot= (x000+ x100+ x010+ x001+
                  x110+ x101+ x011+ x111)
            x000/=tot; x100/=tot; x010/=tot; x001/=tot
            x110/=tot; x101/=tot; x011/=tot; x111/=tot
            # IP3R2 gating
            x0002=0.27; x1002=0.039; x0102=0.29; x0012=0.17
            x1102=0.042; x1012=0.0033; x0112=0.18; x1112=0.0035
            tot2= (x0002+ x1002+ x0102+ x0012+
                   x1102+ x1012+ x0112+ x1112)
            x0002/=tot2; x1002/=tot2; x0102/=tot2; x0012/=tot2
            x1102/=tot2; x1012/=tot2; x0112/=tot2; x1112/=tot2
            arr[STATE_X000]= x000; arr[STATE_X100]= x100
            arr[STATE_X010]= x010; arr[STATE_X001]= x001
            arr[STATE_X110]= x110; arr[STATE_X101]= x101
            arr[STATE_X011]= x011; arr[STATE_X111]= x111
            arr[STATE_X0002]= x0002; arr[STATE_X1002]= x1002
            arr[STATE_X0102]= x0102; arr[STATE_X0012]= x0012
            arr[STATE_X1102]= x1102; arr[STATE_X1012]= x1012
            arr[STATE_X0112]= x0112; arr[STATE_X1112]= x1112
            return arr

        for i in range(N):
            bigY0.extend(init_single_cell())
        bigY0= np.array(bigY0, dtype=float)
    else:
        # use provided initial state (from a 'steady-state finder')
        bigY0= initial_Y.copy()

    t_span= (0.0, p['t_final'])
    t_eval= np.linspace(0.0, p['t_final'], 601)

    def rhs_wrap(t, Y):
        return multicell_rhs(t, Y, p)

    sol= solve_ivp(rhs_wrap, t_span, bigY0,
                   t_eval=t_eval,
                   method='RK45',
                   rtol=1e-6,
                   atol=1e-9)
    return sol

###############################################################################
# 5) Plotting function that SAVES to disk
###############################################################################
def plot_multicell(sol, p, label_suffix="", save_prefix="plot"):
    """
    Instead of showing the plots, we save them to disk.
    """
    import os
    t= sol.t
    Y= sol.y
    N= p['N_cells']
    nvar= 23

    # For filenames, we'll incorporate I_app_val and a cleaned-up suffix
    iapp_val_str = f"{p['I_app_val']:.0f}"
    suffix_sanitized = label_suffix.replace(" ", "_").replace("(", "").replace(")", "")
    base_fname = f"{save_prefix}_Iapp{iapp_val_str}_{suffix_sanitized}"

    # 1) Voltage
    plt.figure(figsize=(10,6))
    for i in range(N):
        idxV= i*nvar + STATE_V
        V_i= Y[idxV, :]
        plt.plot(t, V_i, label=f"Cell {i}")
    plt.title(f"Voltage, N={N}, Gap={p['g_gap']}, I_app={p['I_app_val']} {label_suffix}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    outname_v = base_fname + "_Voltage.png"
    plt.savefig(outname_v, dpi=150)
    plt.close()
    print(f"Saved voltage plot to {outname_v}")

    # 2) Ca
    plt.figure(figsize=(10,6))
    for i in range(N):
        idxCai= i*nvar + STATE_CAI
        Cai= Y[idxCai, :]
        plt.plot(t, Cai, label=f"Cell {i} Ca_i")
    plt.title(f"Cytosolic Ca, N={N}, Gap={p['g_gap']}, I_app={p['I_app_val']} {label_suffix}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Ca_i (mM)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    outname_ca = base_fname + "_Calcium.png"
    plt.savefig(outname_ca, dpi=150)
    plt.close()
    print(f"Saved calcium plot to {outname_ca}")

###############################################################################
# 6) Optional “steady-state finder”
###############################################################################
def find_steady_state(p, t_pre=300.0):
    """
    Let the PDE system 'relax' with I_app=0 for t_pre ms,
    then return the final states as a new initial condition.
    """
    # Temporarily store the original I_app params
    saved_app = p['I_app_val']
    saved_start = p['stim_start']
    saved_end   = p['stim_end']

    # Force no patch clamp current
    p['I_app_val']= 0.0
    p['stim_start']= 9999999.
    p['stim_end']=   -9999999.

    # do a run
    p['t_final'] = t_pre
    sol_pre= run_multicell_sim(p)
    finalY= sol_pre.y[:,-1]  # last column

    # restore param
    p['I_app_val']= saved_app
    p['stim_start']= saved_start
    p['stim_end']= saved_end
    # restore final time
    p['t_final']= 600.0
    return finalY


###############################################################################
# 7) MAIN
###############################################################################
if __name__=="__main__":

    # Parameter dictionary
    p= {
        # Buffers / cell
        'Bscyt':225.0, 'aKscyt':0.1,
        'Bser':2000.0, 'aKser':1.0,
        'Bm':111.0,    'aKm':0.123,

        # IP3R gating
        'a1':400.0, 'a2':0.2, 'a3':400.0, 'a4':0.2, 'a5':20.0,
        'd1':0.13,  'd2':1.049,'d3':0.9434,'d4':0.1445,'d5':0.08234,

        'prodip3':0.01, 'V2ip3':12.5, 'ak2ip3':6.0,
        'V3ip3':0.9,   'ak3ip3':0.1,   'ak4ip3':1.0,

        # Mito
        'psi_volts':0.160,
        'Pmito':2.776e-20,
        'Vmito':2e-12*0.08,
        'alphm':0.2,
        'alphi':1.0,
        'Vnc':1.836,
        'aNa':5000.0,
        'akna':8000.0,
        'akca':8.0,

        # PMCA
        'vu':1540000.0,
        'vm':2200000.0,
        'aku':0.303,
        'akmp':0.14,
        'aru':1.8,
        'arm':2.1,
        'p1':0.1,
        'p2':0.01,

        # SERCA
        'k_serca':0.1,
        'Km_serca':0.5,
        'leak_rate_er':0.05,
        'k_ncx':0.001,

        # Ion channel / gating
        'calcium_activation_threshold_CaCC':0.0005,
        'reference_K':5.4,
        'voltage_slope_Kir61':6.0,
        'voltage_shift_Kir61':15.0,

        'conductance_Kir61':0.025,
        'conductance_TRPC1':0.001,
        'conductance_CaCC':0.001,
        'conductance_CaL':0.0005,
        'conductance_leak':0.01,
        'conductance_IP3R1':0.1,
        'conductance_IP3R2':0.05,
        'conductance_RyR':0.01,

        # External / internal
        'K_out':6.26,
        'Ca_out':2.0,
        'Na_in':15.38,

        # Reversal potentials
        'reversal_potential_K': -80.0,
        'reversal_potential_Ca': 60.0,
        'reversal_potential_Cl': -33.0,

        # Physical constants
        'z':2.0,
        'faraday_constant':96485.0,
        'gas_constant':8314.0,
        'temperature':310.0,

        # Volume, capacitance
        'cell_volume':2e-12,
        'Vcyto':2e-12,
        'ER_volume':2e-12*0.2,
        'membrane_capacitance':0.94,

        # CaL gating
        'activation_midpoint_CaL':-40.0,
        'activation_slope_CaL':4.0,
        'inactivation_midpoint_CaL':-45.0,
        'inactivation_slope_CaL':5.0,
        'voltage_shift_CaL':50.0,
        'slope_factor_CaL':20.0,
        'amplitude_factor_CaL':0.6,

        # PDE / multi-cell
        'N_cells':5,
        'g_gap':0.02,
        'dx':1.0,
        't_final':600.0,

        # Patch clamp
        'stim_cell':0,
        'I_app_val': +70.0,  # We'll override below
        'stim_start':100.0,
        'stim_end':400.0
    }

    # 1) Run with I_app = +70
    print("=== Running with I_app_val = +70 pA ===")
    p['I_app_val'] = +70.0
    sol1= run_multicell_sim(p)
    plot_multicell(sol1, p, label_suffix="(I_app=+70)", save_prefix="run1")

    # 2) Run with I_app = -70
    print("=== Running with I_app_val = -70 pA ===")
    p['I_app_val'] = -70.0
    sol2= run_multicell_sim(p)
    plot_multicell(sol2, p, label_suffix="(I_app=-70)", save_prefix="run2")

    ############################################################################
    # EXTRA SECTION A: "STEADY-STATE" approach
    ############################################################################
    print("\n=== Attempting STEADY-STATE approach, I_app=0 for 300 ms, then +70 pA ===")
    # restore normal T_final for the final runs
    p['t_final']=600.0

    # 1) find SS
    Yss= find_steady_state(p, t_pre=300.0)
    # now do a run from that Yss with +70 pA
    p['I_app_val']   = +70.0
    p['stim_start']  = 100.0
    p['stim_end']    = 400.0
    sol_ss_plus= run_multicell_sim(p, initial_Y=Yss)
    plot_multicell(sol_ss_plus, p, label_suffix="(SteadyState_init_+70)", save_prefix="run3")

    # also see mirror
    p['I_app_val']   = -70.0
    sol_ss_minus= run_multicell_sim(p, initial_Y=Yss)
    plot_multicell(sol_ss_minus, p, label_suffix="(SteadyState_init_-70)", save_prefix="run4")

    ############################################################################
    # EXTRA SECTION B: SIMPLE PARAM SWEEPS
    ############################################################################
    print("\n=== Simple Param Sweep Example: vary CaL by factor of [0.5, 1.0, 2.0] ===")

    original_CaL= p['conductance_CaL']
    for factor in [0.5, 1.0, 2.0]:
        p['conductance_CaL']= original_CaL * factor
        # keep i_app = +70
        p['I_app_val'] = +70.0
        # optional: revert to standard initial condition or from SS
        # for demonstration, let's do from standard IC:
        sol_sweep= run_multicell_sim(p)
        plot_multicell(sol_sweep, p,
                       label_suffix=f"(CaL_x{factor}_Iapp=+70)",
                       save_prefix=f"run_sweep_CaL_{factor}")

    # restore CaL
    p['conductance_CaL']= original_CaL
    print("DONE.")
