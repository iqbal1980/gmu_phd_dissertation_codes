"""
new_single.py (REFINED)

Single-cell pericyte model, incorporating:
  - Kir6.1 (KATP) with ATP-dependent gating
  - Kir2.2 (strong inward rectifier)
  - TRPC3 (receptor-operated channel) with fraction_of_open param
  - Cav1.2 (L-type) 
  - CaCC (TMEM16A)
  - Optional Nav1.2 (TTX-sensitive) - set g_nav1_2 to 0 if not needed
  - IP3R-based ER Ca2+ release, SERCA/PMCA, Mito Ca2+ flux
  - Consolidated leak for other minor channels

Adjustments to better match patch-clamp data:
  - RMP ~ –50 to –60 mV
  - Basal [Ca2+]i ~100 nM
  - Reduced PMCA & SERCA rates so Ca2+ doesn't crash to near zero
  - Lower ER leak
  - Slightly increased L-type conductance
  - Lower KATP open fraction at normal ATP, etc.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Safe exponential
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def safe_exp(x, clip_min=-100, clip_max=100):
    return np.exp(np.clip(x, clip_min, clip_max))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Optional IP3 production/degradation (commented by default)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ip3_production_rate(Ca_i, params):
    baseline = params.get('ip3_prod_base', 0.0)
    ca_factor= params.get('ip3_prod_ca_factor', 0.0)
    k_ca    = params.get('ip3_prod_ca_k', 0.3)
    hill_ca = params.get('ip3_prod_ca_hill', 2.0)
    frac_ca = (Ca_i**hill_ca)/(Ca_i**hill_ca + k_ca**hill_ca + 1e-30)
    return baseline + ca_factor*frac_ca

def ip3_degradation_rate(IP3, params):
    k_deg = params.get('ip3_deg_k', 0.0)
    return k_deg * IP3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3) Ion channel current definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def kir6_1_current(V, ATP, params):
    """
    KATP (Kir6.1+SUR2B).
    We'll reduce open fraction at normal ATP=1mM by setting 'ATP_half' < 1.0.
    e.g. if ATP_half=0.2 => about 1/(1+(1/0.2)^2)= ~1/26=0.038 open
    """
    g_kir = params['g_kir6_1']
    E_k   = params['E_K']
    ATP_half= params['ATP_half']
    ATP_hill= params['ATP_hill']
    atp_factor = 1.0/(1.0 + (ATP/ATP_half)**ATP_hill)
    return g_kir*atp_factor*(V - E_k)

def kir2_2_current(V, params):
    """
    Kir2.2: strong inward rectifier, moderate expression => modest conductance.
    Using a Boltzmann to represent rectification near negative potentials.
    """
    g_kir2= params['g_kir2_2']
    E_k   = params['E_K']
    V_half= params['V_half_Kir2_2']
    slope = params['slope_Kir2_2']
    arg   = (V - V_half)/slope
    act   = 1.0/(1.0 + safe_exp(arg))
    return g_kir2*act*(V - E_k)

def trpc3_current(V, frac_open, params):
    """
    TRPC3: nonselective cation, E_rev ~ 0 mV
    'frac_open' is typically 0 at rest, >0 if Gq/PLC is stimulated
    """
    return params['g_trpc3']*frac_open*(V - params['E_TRPC3'])

def cav1_2_current(V, Ca_i, params):
    """
    L-type Ca channel; threshold ~ –40 mV; half-activation ~ –20 mV
    """
    g_ca= params['g_cav1_2']
    E_ca= params['E_Ca']
    v0_d = params['cav_d_half']
    k_d  = params['cav_d_slope']
    v0_f = params['cav_f_half']
    k_f  = params['cav_f_slope']

    arg_d = -(V - v0_d)/k_d
    d_inf = 1.0/(1.0 + safe_exp(arg_d))
    arg_f = (V - v0_f)/k_f
    f_inf = 1.0/(1.0 + safe_exp(arg_f))

    return g_ca*d_inf*f_inf*(V - E_ca)

def cacc_current(V, Ca_i, params):
    """
    TMEM16A: Ca2+-activated Cl- channel
    E_rev ~ –30 mV
    K_half ~ ~0.5–1.0 µM => we do 0.0005–0.001 mM
    """
    g_cl = params['g_cacc']
    E_cl = params['E_Cl']
    K_half= params['ca_half_cacc']
    hill = params['ca_hill_cacc']

    top = Ca_i**hill
    bot = top + K_half**hill + 1e-30
    act = top/bot
    return g_cl*act*(V - E_cl)

def nav1_2_steadystates(V, params):
    """
    Quasi-steady gating for Nav1.2
    If not needed, set g_nav1_2=0.
    """
    vm_m = params['nav_m_half']
    km_m = params['nav_m_slope']
    vm_h = params['nav_h_half']
    km_h = params['nav_h_slope']

    arg_m = -(V - vm_m)/km_m
    m_inf = 1.0/(1.0+safe_exp(arg_m))
    arg_h = (V - vm_h)/km_h
    h_inf = 1.0/(1.0+safe_exp(arg_h))

    return m_inf,h_inf

def nav1_2_current(V, m_nav, h_nav, params):
    return params['g_nav1_2']*(m_nav**3)*h_nav*(V - params['E_Na'])

def leak_current(V, params):
    """
    A small K+ leak + a small cation leak => set final RMP ~ –50 to –60 mV
    """
    Ik_leak= params['g_leak_k']*(V - params['E_K'])
    Ins_leak=params['g_leak_ns']*(V - params['E_leak_ns'])
    return Ik_leak+Ins_leak

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4) IP3R, SERCA, PMCA, Mito
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params):
    d1 = params['d1']
    d5 = params['d5']
    m_inf = IP3/(IP3 + d1 + 1e-12)
    n_inf = Ca_i/(Ca_i + d5 + 1e-12)
    p_open= (m_inf**3)*(n_inf**3)*(h_ip3**3)
    return params['v_ip3r']*p_open*(Ca_ER - Ca_i)

def dh_ip3_dt(Ca_i, h_ip3, params):
    d2    = params['d2']
    tau_h = params['tau_h']
    h_inf = d2/(d2 + Ca_i +1e-12)
    return (h_inf - h_ip3)/tau_h

def pmca_flux(Ca_i, params):
    top = Ca_i
    bot = Ca_i + params['K_PMCA'] + 1e-30
    return params['v_PMCA']*(top/bot)

def serca_flux(Ca_i, params):
    top = Ca_i**2
    bot = Ca_i**2 + params['K_SERCA']**2 +1e-30
    return params['v_SERCA']*(top/bot)

def mito_fluxes(Ca_i, Ca_mito, params):
    J_up = params['v_mito']*(Ca_i/(Ca_i + params['K_mito'] +1e-12))
    J_rel= params['v_mito_rel']*Ca_mito
    return J_up, J_rel

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5) ODE System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model(t, y, params):
    """
    Single pericyte ODE system with recommended refinements to match typical
    –50 to –60 mV rest and ~0.1 µM Ca2+ basal.
    y => [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3]
    """

    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3 = y

    # Nav gating
    m_nav, h_nav = nav1_2_steadystates(V, params)
    # TRPC3 fraction
    trpc3_frac   = params.get('TRPC3_frac_open', 0.0)

    # Currents
    Ikatp = kir6_1_current(V, ATP, params)
    Ikir2 = kir2_2_current(V, params)
    Itrpc = trpc3_current(V, trpc3_frac, params)
    Icav  = cav1_2_current(V, Ca_i, params)
    Icacc = cacc_current(V, Ca_i, params)
    Ina   = nav1_2_current(V, m_nav, h_nav, params)
    Ileak_= leak_current(V, params)

    I_total= Ikatp + Ikir2 + Itrpc + Icav + Icacc + Ina + Ileak_

    dVdt= -I_total/params['Cm']

    # Ca fluxes
    J_ip3r_ = ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params)
    J_pmca_ = pmca_flux(Ca_i, params)
    J_ser_  = serca_flux(Ca_i, params)
    J_leak_ = params['v_ER_leak']*(Ca_ER - Ca_i)
    J_mup, J_mrel = mito_fluxes(Ca_i, Ca_mito, params)

    # Ca from channels => negative if inward currents
    f_trp_ca = params.get('f_trpc3_ca', 0.1)
    J_cav_in = -Icav*params['ca_current_factor']
    J_trp_in = -Itrpc*params['ca_current_factor']*f_trp_ca

    dCa_i= (J_cav_in + J_trp_in
            + J_ip3r_
            - J_pmca_
            - J_ser_
            + J_leak_
            + (J_mrel - J_mup))

    dATP= 0.0
    dCaER= J_ser_ - J_ip3r_ - J_leak_
    # (comment out if you want IP3 to remain constant)
    dIP3= 0.0
    dCaM= J_mup - J_mrel
    dhdt= dh_ip3_dt(Ca_i, h_ip3, params)

    return [dVdt, dCa_i, dATP, dCaER, dIP3, dCaM, dhdt]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6) Main Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=="__main__":
    params = {
        # Membrane & Leak
        'Cm': 1.0,         # ~1 µF/cm^2
        'g_leak_k': 0.005, # tuned for ~–55 mV rest
        'g_leak_ns': 0.01, # additional cation leak => helps avoid too negative
        'E_leak_ns': -20.0,
        # KATP
        'g_kir6_1': 0.2,   # higher max, but mostly closed at ATP=1 => small effect
        'ATP_half': 0.2,   # if [ATP]=1 => open fraction ~ 3.8%
        'ATP_hill': 2.0,
        # Kir2.2
        'g_kir2_2': 0.01,
        'V_half_Kir2_2': -70,
        'slope_Kir2_2': 10,
        # TRPC3
        'g_trpc3': 0.05,   # smaller => reduce big depolarizing current at rest
        'E_TRPC3': 0.0,
        'f_trpc3_ca': 0.1,
        'TRPC3_frac_open': 0.0, # 0 => closed at rest
        # Cav1.2
        'g_cav1_2': 0.15,  # slightly bigger to maintain mild Ca2+ influx
        'E_Ca': 60.0,
        'cav_d_half': -20.0,
        'cav_d_slope': 6.0,
        'cav_f_half': -40.0,
        'cav_f_slope': 6.0,
        # CaCC
        'g_cacc': 0.1,
        'E_Cl': -30.0,
        'ca_half_cacc': 0.0005, # 0.5 µM
        'ca_hill_cacc': 2.0,
        # Nav1.2
        'g_nav1_2': 0.0,   # disable by default
        'E_Na': 60.0,
        'nav_m_half': -25.0,
        'nav_m_slope': 5.0,
        'nav_h_half': -65.0,
        'nav_h_slope': 7.0,
        # IP3R
        'd1': 0.13,
        'd2': 1.049,
        'd5': 0.082,
        'v_ip3r': 0.3,    # slightly lower => avoid quick store emptying
        'tau_h': 2.0,
        # Pumps & leak
        'v_PMCA': 0.1,    # reduced => avoid [Ca2+] dropping too low
        'K_PMCA': 0.0003,
        'v_SERCA': 0.2,   # also reduced
        'K_SERCA': 0.0003,
        'v_ER_leak': 0.01,# smaller leak => keep ER from draining
        # Mito
        'v_mito': 0.02,
        'K_mito': 0.01,
        'v_mito_rel': 0.01,
        # Reversal potentials
        'E_K': -85.0,
        # Factor to scale membrane Ca2+ current => concentration
        'ca_current_factor': 0.01
    }

    # Initial Conditions: [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3]
    # Start near –55 mV, Ca_i=0.1 µM, ER=0.3 mM, IP3=0 => minimal release
    y0 = [
        -55.0,
        0.0001,   # 0.1 µM
        1.0,
        0.3,
        0.0,
        0.0001,
        0.8
    ]

    t_span = (0, 2000)
    t_eval = np.linspace(0, 2000, 2001)
    sol = solve_ivp(lambda t, y: model(t, y, params),
                    t_span, y0, t_eval=t_eval, method='LSODA')
    time = sol.t
    V_arr      = sol.y[0]
    Ca_i_arr   = sol.y[1]
    ATP_arr    = sol.y[2]
    Ca_ER_arr  = sol.y[3]
    IP3_arr    = sol.y[4]
    Ca_mito_arr= sol.y[5]
    h_ip3_arr  = sol.y[6]

    # Recompute currents & fluxes for plotting
    Ikir6_list, Ikir2_list = [], []
    Itrpc3_list, Icav_list = [], []
    Icacc_list, Ileak_list = [], []
    Ina_list               = []
    J_ip3r_list, J_pmca_list, J_serca_list= [],[],[]
    J_er_leak_list         = []
    J_mito_up_list, J_mito_rel_list = [],[]

    from math import isclose

    for i in range(len(time)):
        V_  = V_arr[i]
        c_  = Ca_i_arr[i]
        er_ = Ca_ER_arr[i]
        ip3_= IP3_arr[i]
        mit_= Ca_mito_arr[i]
        h_  = h_ip3_arr[i]
        # Nav gating
        m_, h_ = nav1_2_steadystates(V_, params)
        Ik6_ = kir6_1_current(V_, ATP_arr[i], params)
        Ik2_ = kir2_2_current(V_, params)
        Itr_= trpc3_current(V_, params['TRPC3_frac_open'], params)
        Ica_= cav1_2_current(V_, c_, params)
        Icl_= cacc_current(V_, c_, params)
        Ina_= nav1_2_current(V_, m_, h_, params)
        Il_ = leak_current(V_, params)

        Ikir6_list.append(Ik6_)
        Ikir2_list.append(Ik2_)
        Itrpc3_list.append(Itr_)
        Icav_list.append(Ica_)
        Icacc_list.append(Icl_)
        Ileak_list.append(Il_)
        Ina_list.append(Ina_)

        # fluxes
        jip3_ = ip3r_flux(c_, er_, ip3_, h_ip3_arr[i], params)
        J_ip3r_list.append(jip3_)
        jpmca_ = pmca_flux(c_, params)
        J_pmca_list.append(jpmca_)
        jser_  = serca_flux(c_, params)
        J_serca_list.append(jser_)
        jelk_  = params['v_ER_leak']*(er_ - c_)
        J_er_leak_list.append(jelk_)
        jup_, jrel_ = mito_fluxes(c_, mit_, params)
        J_mito_up_list.append(jup_)
        J_mito_rel_list.append(jrel_)

    # Convert to np arrays
    Ikir6_1_array= np.array(Ikir6_list)
    Ikir2_2_array= np.array(Ikir2_list)
    Itrpc3_array = np.array(Itrpc3_list)
    Icav1_2_array= np.array(Icav_list)
    Icacc_array  = np.array(Icacc_list)
    Ileak_array  = np.array(Ileak_list)
    Ina_array    = np.array(Ina_list)
    J_ip3r_array = np.array(J_ip3r_list)
    J_pmca_array = np.array(J_pmca_list)
    J_serca_array= np.array(J_serca_list)
    J_er_leak_arr= np.array(J_er_leak_list)
    J_mito_up_arr= np.array(J_mito_up_list)
    J_mito_rel_arr= np.array(J_mito_rel_list)

    # Logging voltage
    with open("new_single_voltage_log.csv","w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["Time_ms","Vm_mV"])
        for i in range(len(time)):
            if i%10==0:
                writer.writerow([time[i], V_arr[i]])

    # Plot
    plt.figure()
    plt.plot(time, V_arr, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel('Vm (mV)')
    plt.title('Membrane Potential (revised single-cell)')
    plt.tight_layout()

    plt.figure()
    plt.plot(time, Ca_i_arr*1e6, 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel('[Ca²⁺]_i (nM)')
    plt.title('Cytosolic Ca²⁺ (nM)')
    plt.tight_layout()

    plt.figure()
    plt.plot(time, Ca_ER_arr, label='ER Ca (mM)')
    plt.plot(time, Ca_mito_arr, label='Mito Ca (mM)')
    plt.legend()
    plt.title('ER & Mito [Ca²⁺]')
    plt.xlabel('Time (ms)')
    plt.ylabel('Concentration')
    plt.tight_layout()

    # Currents
    plt.figure(figsize=(14,8))
    plt.subplot(2,4,1); plt.plot(time, Ikir6_1_array); plt.title('I_Kir6.1')
    plt.subplot(2,4,2); plt.plot(time, Ikir2_2_array); plt.title('I_Kir2.2')
    plt.subplot(2,4,3); plt.plot(time, Itrpc3_array); plt.title('I_TRPC3')
    plt.subplot(2,4,4); plt.plot(time, Icav1_2_array); plt.title('I_Cav1.2')
    plt.subplot(2,4,5); plt.plot(time, Icacc_array); plt.title('I_CaCC')
    plt.subplot(2,4,6); plt.plot(time, Ileak_array); plt.title('I_leak')
    plt.subplot(2,4,7); plt.plot(time, Ina_array); plt.title('I_Na')
    plt.tight_layout()

    # Fluxes
    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1); plt.plot(time, J_ip3r_array); plt.title('J_IP3R')
    plt.subplot(2,3,2); plt.plot(time, J_pmca_array); plt.title('J_PMCA')
    plt.subplot(2,3,3); plt.plot(time, J_serca_array); plt.title('J_SERCA')
    plt.subplot(2,3,4); plt.plot(time, J_er_leak_arr); plt.title('ER leak')
    plt.subplot(2,3,5); plt.plot(time, J_mito_up_arr); plt.title('Mito Uptake')
    plt.subplot(2,3,6); plt.plot(time, J_mito_rel_arr); plt.title('Mito Release')
    plt.tight_layout()

    plt.show()
