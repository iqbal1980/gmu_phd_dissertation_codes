"""
new_multi.py (REFINED)

Multi-cell pericyte network with the refined single-cell channel set:
  - KATP, Kir2.2, TRPC3, Cav1.2, CaCC, (Nav optional), ER/mito Ca2+ 
  - Minor leaks
  - Gap junction coupling for membrane potential
  - Optional patch clamp injection
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

def safe_exp(x, cmin=-100, cmax=100):
    return np.exp(np.clip(x, cmin, cmax))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REUSE single-cell channel definitions from new_single.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def kir6_1_current(V, ATP, p):
    atp_factor = 1.0/(1.0+(ATP/p['ATP_half'])**p['ATP_hill'])
    return p['g_kir6_1']*atp_factor*(V - p['E_K'])

def kir2_2_current(V, p):
    arg = (V - p['V_half_Kir2_2'])/p['slope_Kir2_2']
    act = 1.0/(1.0+safe_exp(arg))
    return p['g_kir2_2']*act*(V - p['E_K'])

def trpc3_current(V, frac_open, p):
    return p['g_trpc3']*frac_open*(V - p['E_TRPC3'])

def cav1_2_current(V, Ca_i, p):
    arg_d= -(V - p['cav_d_half'])/p['cav_d_slope']
    d_inf= 1.0/(1.0+safe_exp(arg_d))
    arg_f= (V - p['cav_f_half'])/p['cav_f_slope']
    f_inf= 1.0/(1.0+safe_exp(arg_f))
    return p['g_cav1_2']*d_inf*f_inf*(V - p['E_Ca'])

def cacc_current(V, Ca_i, p):
    top= Ca_i**p['ca_hill_cacc']
    bot= top + p['ca_half_cacc']**p['ca_hill_cacc'] +1e-30
    act= top/bot
    return p['g_cacc']*act*(V - p['E_Cl'])

def nav12_steadystates(V, p):
    arg_m= -(V - p['nav_m_half'])/p['nav_m_slope']
    m_inf= 1.0/(1.0+safe_exp(arg_m))
    arg_h= (V - p['nav_h_half'])/p['nav_h_slope']
    h_inf= 1.0/(1.0+safe_exp(arg_h))
    return m_inf, h_inf

def nav12_current(V, m_nav, h_nav, p):
    return p['g_nav1_2']*(m_nav**3)*h_nav*(V - p['E_Na'])

def leak_current(V, p):
    Ik_leak = p['g_leak_k']*(V - p['E_K'])
    Ins_leak= p['g_leak_ns']*(V - p['E_leak_ns'])
    return Ik_leak + Ins_leak

# IP3R etc
def ip3r_flux(Ca_i, Ca_er, IP3, h_ip3, p):
    d1= p['d1']
    d5= p['d5']
    m_inf= IP3/(IP3+d1+1e-12)
    n_inf= Ca_i/(Ca_i+d5+1e-12)
    p_open=(m_inf**3)*(n_inf**3)*(h_ip3**3)
    return p['v_ip3r']*p_open*(Ca_er - Ca_i)

def dh_ip3_dt(Ca_i, h_ip3, p):
    d2= p['d2']
    tau= p['tau_h']
    h_inf= d2/(d2+Ca_i+1e-12)
    return (h_inf - h_ip3)/tau

def pmca_flux(Ca_i, p):
    top= Ca_i
    bot= Ca_i + p['K_PMCA']+1e-30
    return p['v_PMCA']*(top/bot)

def serca_flux(Ca_i, p):
    top= Ca_i**2
    bot= Ca_i**2 + p['K_SERCA']**2 +1e-30
    return p['v_SERCA']*(top/bot)

def mito_fluxes(Ca_i, Ca_mito, p):
    Jup = p['v_mito']*(Ca_i/(Ca_i+p['K_mito']+1e-12))
    Jrel= p['v_mito_rel']*Ca_mito
    return Jup, Jrel

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SINGLE-CELL ODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def single_cell_ode(t, y, p):
    """
    y => [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3].
    We'll keep TRPC3_frac_open as param for now.
    """
    V, Ca_i, ATP, Ca_er, IP3, Ca_mito, h_ip3= y

    m_nav, h_nav = nav12_steadystates(V, p)
    frac_trp= p.get('TRPC3_frac_open',0.0)

    Ikatp= kir6_1_current(V, ATP, p)
    Ikir2= kir2_2_current(V, p)
    Itrp3= trpc3_current(V, frac_trp, p)
    Icav = cav1_2_current(V, Ca_i, p)
    Icacc= cacc_current(V, Ca_i, p)
    Ina  = nav12_current(V, m_nav, h_nav, p)
    Ileak= leak_current(V, p)

    I_tot= Ikatp+Ikir2+Itrp3+Icav+Icacc+Ina+Ileak
    dVdt= -I_tot/p['Cm']

    # Ca fluxes
    Jip3= ip3r_flux(Ca_i, Ca_er, IP3, h_ip3, p)
    Jpmca= pmca_flux(Ca_i, p)
    Jserc= serca_flux(Ca_i, p)
    Jleak= p['v_ER_leak']*(Ca_er - Ca_i)
    Jup, Jrel= mito_fluxes(Ca_i, Ca_mito, p)

    f_trp_ca= p.get('f_trpc3_ca', 0.1)
    Jcav_in= -Icav*p['ca_current_factor']
    Jtrp_in= -Itrp3*p['ca_current_factor']*f_trp_ca

    dCa_i= (Jcav_in + Jtrp_in
            + Jip3
            - Jpmca
            - Jserc
            + Jleak
            + (Jrel-Jup))

    dATP= 0.0
    dCaER= Jserc - Jip3 - Jleak
    dIP3 = 0.0  # or dynamic if desired
    dCaM= Jup - Jrel
    dh_ip3_= dh_ip3_dt(Ca_i, h_ip3, p)

    return [dVdt, dCa_i, dATP, dCaER, dIP3, dCaM, dh_ip3_]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NETWORK ODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def network_ode(t, Y, p, N, g_gap, inj):
    dYdt= np.zeros_like(Y)
    nv=7
    for i in range(N):
        idx0= i*nv
        idx1= idx0+nv
        yi= Y[idx0:idx1]

        # injection
        if (inj['start_index']<=i<=inj['end_index']
            and inj['t_start']<=t<=inj['t_end']):
            I_inj= inj['I_inj']
        else:
            I_inj=0.0

        dyi= single_cell_ode(t, yi, p)
        # gap junction
        V_i= yi[0]
        if i==0:
            V_left= V_i
        else:
            V_left= Y[(i-1)*nv]

        if i==N-1:
            V_right= V_i
        else:
            V_right= Y[(i+1)*nv]

        I_coup= g_gap*((V_left - V_i)+(V_right - V_i))
        dyi[0]+= I_coup/p['Cm']
        # add injection
        dyi[0]+= I_inj/p['Cm']

        dYdt[idx0:idx1]= dyi

    return dYdt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=="__main__":
    N=30
    g_gap= 0.05
    inj_dict= {
        'start_index': 5,
        'end_index': 17,
        'I_inj': 0.05,
        't_start': 400,
        't_end':   900
    }

    # Similar parameters as revised new_single
    params= {
        'Cm': 1.0,
        # minor leaks
        'g_leak_k': 0.005,
        'g_leak_ns':0.01,
        'E_leak_ns':-20.0,
        # KATP
        'g_kir6_1':0.2,
        'ATP_half':0.2,
        'ATP_hill':2.0,
        # Kir2.2
        'g_kir2_2':0.01,
        'V_half_Kir2_2':-70,
        'slope_Kir2_2':10,
        # TRPC3
        'g_trpc3':0.05,
        'E_TRPC3':0.0,
        'TRPC3_frac_open':0.0,
        'f_trpc3_ca':0.1,
        # Cav1.2
        'g_cav1_2':0.15,
        'E_Ca':60.0,
        'cav_d_half':-20.0,
        'cav_d_slope':6.0,
        'cav_f_half':-40.0,
        'cav_f_slope':6.0,
        # CaCC
        'g_cacc':0.1,
        'E_Cl':-30.0,
        'ca_half_cacc':0.0005,
        'ca_hill_cacc':2.0,
        # Nav
        'g_nav1_2':0.0,
        'E_Na':60.0,
        'nav_m_half':-25.0,
        'nav_m_slope':5.0,
        'nav_h_half':-65.0,
        'nav_h_slope':7.0,
        # IP3R
        'd1':0.13,
        'd2':1.049,
        'd5':0.082,
        'v_ip3r':0.3,
        'tau_h':2.0,
        # pumps
        'v_PMCA':0.1,
        'K_PMCA':0.0003,
        'v_SERCA':0.2,
        'K_SERCA':0.0003,
        'v_ER_leak':0.01,
        # Mito
        'v_mito':0.02,
        'K_mito':0.01,
        'v_mito_rel':0.01,
        # Reversal
        'E_K':-85.0,
        'ca_current_factor':0.01
    }

    # Each cell: [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3]
    # Start near –55 mV, 0.1 µM Ca
    y0_cell= [-55.0, 0.0001, 1.0, 0.3, 0.0, 0.0001, 0.8]
    Y0= np.tile(y0_cell, N)

    t_span= (0,1500)
    t_eval= np.linspace(0,1500,1501)
    sol= solve_ivp(lambda t, Y: network_ode(t, Y, params, N, g_gap, inj_dict),
                   t_span, Y0, t_eval=t_eval, method='LSODA')

    time= sol.t
    Y_sol= sol.y

    # Extract V
    nvar=7
    V_all= []
    for i in range(N):
        V_all.append( Y_sol[i*nvar,:] )

    # log
    with open("new_multi_voltage_log.csv","w",newline="") as f:
        w= csv.writer(f)
        w.writerow(["time_ms","Cell0_V"])
        for i in range(len(time)):
            if i%10==0:
                w.writerow([time[i], V_all[0][i]])

    # Plot
    plt.figure()
    for i in range(N):
        plt.plot(time, V_all[i], label=f'Cell {i}')
    plt.title('Revised Multi-Cell Pericyte Model: Vm')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # If you want to see Ca_i for each cell similarly:
    # ...
