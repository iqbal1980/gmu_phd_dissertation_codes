"""
new_single.py

Single-cell pericyte model incorporating:
  - Kir6.1 (KATP) with ATP-dependent gating (SUR2B subunit implied)
  - Kir2.2 (inward rectifier)
  - TRPC3 (receptor-operated, DAG-activated nonselective cation)
  - Cav1.2 (L-type voltage-gated Ca2+ channel)
  - CaCC (TMEM16A) – Ca2+-activated Cl- channel
  - Nav1.2 (TTX-sensitive voltage-gated Na+ channel)
  - IP3R (De Young–Keizer style) for ER Ca2+ release
  - Mitochondrial Ca2+ fluxes & SERCA/PMCA
  - A minor leak current representing other channels (TRPM7, Kv, etc.)

Translation Efficiency & Subunits:
  - Kir6.1 channels form hetero-octamers with SUR2B. We assume high expression => strong KATP presence.
  - TRPC3 is assumed mostly homomeric (since TRPC6 is much lower). We represent it as DAG-activated.
  - Cav1.2 includes implicit beta subunits for typical smooth muscle gating.
  - CaCC is TMEM16A (Ano1) homodimer, with ~1 µM Ca2+ EC50.
  - Nav1.2 is co-expressed with beta subunits, giving fast neuronal-like gating but low expression => small gNa.

All parameters and gating are annotated with references in the code comments.

References for in-line bracketed citations:
[1] Quayle 2007; [2] Sakagami 1999; [3] Li & Rinzel 1994; etc.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

# =============================================================================
# HELPER FUNCTIONS & SAFE EXP
# =============================================================================

def safe_exp(x, clip_min=-100, clip_max=100):
    """
    Compute exp(x) safely, clipping x to prevent overflow.
    For rapid gating (e.g. Na channel) we want numerical stability.
    """
    return np.exp(np.clip(x, clip_min, clip_max))

# =============================================================================
# OPTIONAL: IP3 Production/Degradation
# =============================================================================

def ip3_production_rate(Ca_i, params):
    """
    Production of IP3, e.g. by PLC -> DAG + IP3.
    Could be Ca2+-dependent or baseline.
    """
    baseline = params.get('ip3_prod_baseline', 0.0)
    ca_factor = params.get('ip3_prod_ca_factor', 0.0)
    k_ca = params.get('ip3_prod_ca_k', 0.3)
    hill_ca = params.get('ip3_prod_ca_hill', 2.0)
    # Simple Hill form:
    frac_ca = (Ca_i**hill_ca) / (Ca_i**hill_ca + k_ca**hill_ca + 1e-30)
    return baseline + ca_factor * frac_ca

def ip3_degradation_rate(IP3, params):
    """
    First-order IP3 degradation with rate constant ip3_deg_k.
    """
    k_deg = params.get('ip3_deg_k', 0.0)
    return k_deg * IP3


# =============================================================================
# MEMBRANE CURRENTS
# =============================================================================
def kir6_1_current(V, ATP, params):
    """
    Kir6.1 (KATP) channel – subunits: 4 x Kir6.1 + 4 x SUR2B.
    Weak inward rectification, *not* strongly voltage-gated.
    Gate depends on [ATP]: 
      Po = 1 / [1 + (ATP/ATP_half)^hill].
    
    Single-channel g ~35 pS [Ashcroft, 2000], but we scale up to 
    a macroscopic conductance as param['g_kir6_1'].
    E_rev = E_K ~ –80 mV typical.
    
    [Ref: Kir6.1 expression is high in pericytes [Kcnj8=1670 counts].
     Often near silent at normal ATP ~1–5 mM, but can open 
     significantly if ATP drops => hyperpolarize ~–80 mV.]
    """
    g_kir = params['g_kir6_1']
    E_k   = params['E_K']
    ATP_half = params.get('ATP_half', 0.5)  # ~0.5 mM
    ATP_hill = params.get('ATP_hill', 2.0)

    # Probability channel is open given cytosolic [ATP]
    atp_factor = 1.0 / (1.0 + (ATP/ATP_half)**ATP_hill)

    return g_kir * atp_factor * (V - E_k)

def kir2_2_current(V, params):
    """
    Kir2.2 – strongly rectifying K+ channel, 
    but expression in pericytes is relatively modest (~31 counts).
    We'll keep it small but extant.
    We apply a standard Boltzmann for rectification near negative V:
      Po = 1 / [1 + exp((V - V_half)/slope)].
    E_rev = E_K.
    """
    g_kir2 = params['g_kir2_2']
    E_k = params['E_K']
    V_half = params['V_half_Kir2_2']
    slope  = params['slope_Kir2_2']

    arg = (V - V_half)/slope
    activation = 1.0/(1.0 + safe_exp(arg))
    return g_kir2 * activation * (V - E_k)

def trpc3_current(V, TRPC3_act, params):
    """
    TRPC3 – receptor-operated, DAG-activated, nonselective cation.
    E_rev ~ 0 mV. 
    We'll treat the open fraction as TRPC3_act (0->1), 
    an input that depends on Gq/PLC => DAG.
    Single-channel ~66 pS, partial Ca2+ permeability [pCa/pNa~1.5].
    Macroscopic conductance => param['g_trpc3'].
    """
    g_trpc3 = params['g_trpc3']
    E_trpc3 = params['E_TRPC3']
    return g_trpc3 * TRPC3_act * (V - E_trpc3)

def cav1_2_current(V, Ca_i, params):
    """
    Cav1.2 (L-type) – typical smooth muscle gating.
    Activation: half ~–20 mV, slope ~6 mV
    Inactivation: half ~–40 mV, slope ~6 mV
    E_rev ~ +60 mV for Ca2+.

    Could add Ca2+-dependent inactivation if desired. 
    We'll do a simpler scheme, referencing [Sakagami 1999].
    """
    g_ca = params['g_cav1_2']
    E_ca = params['E_Ca']

    # Activation
    v0_d = params['cav_d_half']
    k_d  = params['cav_d_slope']
    arg_d = -(V - v0_d)/k_d
    d_inf = 1.0/(1.0 + safe_exp(arg_d))

    # Inactivation
    v0_f = params['cav_f_half']
    k_f  = params['cav_f_slope']
    arg_f = (V - v0_f)/k_f
    f_inf = 1.0/(1.0 + safe_exp(arg_f))

    # If you want some Ca2+-dependent inactivation factor, 
    # define here or keep it at 1.0 for simplicity.
    f_ca = 1.0

    return g_ca * d_inf * f_inf * f_ca * (V - E_ca)

def cacc_current(V, Ca_i, params):
    """
    CaCC (TMEM16A) – Ca2+-activated Cl-.
    E_rev ~ –30 mV. 
    Use a Hill eq for Ca gating: Po ~ (Ca^n / (Ca^n + K_cacc^n)).
    Single-channel ~2-8 pS => sum to a big macroscopic G if many channels. 
    """
    g_cacc = params['g_cacc']
    E_cl   = params['E_Cl']
    K_half = params['ca_half_cacc']
    hill_n = params['ca_hill_cacc']

    # Ca gating
    top = Ca_i**hill_n
    bot = top + K_half**hill_n + 1e-30
    act = top/bot

    return g_cacc * act * (V - E_cl)

def nav1_2_current(V, m_nav, h_nav, params):
    """
    Nav1.2 – TTX-sensitive neuronal-like Na+ channel. 
    We'll do a "quasi-steady" approach (m_inf, h_inf each time step),
    or define separate ODE if you want full HH kinetics. 
    For brevity, we do the steady-state approach:
       I_Na = g_nav * m_inf^3 * h_inf * (V - E_Na).
    Activation half ~–25, inact half ~–65, from [Zhang et al 2005].
    """
    g_na = params['g_nav1_2']
    E_na = params['E_Na']

    return g_na * (m_nav**3)*h_nav * (V - E_na)

def nav1_2_steadystates(V, params):
    """
    For each time step, approximate m_inf, h_inf from voltage. 
    True in neurons, you'd do separate ODE or look-up table. 
    This quick approach is fine if dt is small. 
    Activation half ~–25, slope ~5; inact half ~–65, slope ~7 or so.
    """
    vm_m_half = params['nav_m_half']
    km_m      = params['nav_m_slope']
    vm_h_half = params['nav_h_half']
    km_h      = params['nav_h_slope']

    # Activation
    arg_m = -(V - vm_m_half)/km_m
    m_inf = 1.0/(1.0 + safe_exp(arg_m))
    # Inactivation
    arg_h = (V - vm_h_half)/km_h
    h_inf = 1.0/(1.0 + safe_exp(arg_h))

    return m_inf, h_inf

def leak_current(V, params):
    """
    Consolidated leak representing smaller channels 
    (TRPM7, leftover Kir, Kv, background cation, etc.).
    We combine a K+-specific leak and a nonselective cation leak for realism.
    g_leak_K => E_k
    g_leak_NS => E_ns ~ –20 mV or so, or we pick an intermediate.

    For demonstration, let's do:
       I_leak = g_leak_K*(V - E_K) + g_leak_ns*(V - E_leak_ns).
    Adjust as needed to get correct resting potential with all gating closed.
    """
    I_k_leak = params['g_leak_k']*(V - params['E_K'])
    I_ns_leak= params['g_leak_ns']*(V - params['E_leak_ns'])
    return I_k_leak + I_ns_leak


# =============================================================================
# IP3R FLUX (DE YOUNG–KEIZER)
# =============================================================================
def ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params):
    """
    De Young–Keizer style or Li-Rinzel style biphasic gating:
    p_open ~ m_inf^3 * n_inf^3 * h^3
    where
      m_inf = IP3/(IP3 + d1)
      n_inf = Ca_i/(Ca_i + d5)
    h_ip3 -> inactivation gate with dh/dt eq.
    v_ip3r => maximum flux factor
    """
    d1 = params['d1']
    d5 = params['d5']
    m_inf = IP3/(IP3 + d1 + 1e-12)
    n_inf = Ca_i/(Ca_i + d5 + 1e-12)
    p_open = (m_inf**3)*(n_inf**3)*(h_ip3**3)
    J_ip3r = params['v_ip3r'] * p_open * (Ca_ER - Ca_i)
    return J_ip3r

def dh_ip3_dt(Ca_i, h_ip3, params):
    """
    Inactivation gate eq:
      dh/dt = (h_inf - h)/tau_h
      h_inf = d2/(d2 + Ca_i)
    """
    d2 = params['d2']
    tau_h = params['tau_h']
    h_inf = d2/(d2 + Ca_i + 1e-12)
    return (h_inf - h_ip3)/tau_h

# =============================================================================
# PMCA, SERCA, Mito fluxes
# =============================================================================
def pmca_flux(Ca_i, params):
    """ PMCA – plasma membrane Ca2+ ATPase. """
    top = Ca_i
    bot = Ca_i + params['K_PMCA'] + 1e-30
    return params['v_PMCA']*(top/bot)

def serca_flux(Ca_i, params):
    """ SERCA – pumps Ca2+ into ER. Hill n=2 used. """
    top = Ca_i**2
    bot = Ca_i**2 + params['K_SERCA']**2 + 1e-30
    return params['v_SERCA']*(top/bot)

def mito_fluxes(Ca_i, Ca_mito, params):
    """
    Mitochondrial Ca2+ uptake (via MCU) and release (via NCLX).
    J_in = v_mito*(Ca_i/(Ca_i + K_mito)), 
    J_out= v_mito_rel*Ca_mito.
    """
    J_uptake = params['v_mito']*(Ca_i/(Ca_i + params['K_mito'] + 1e-12))
    J_release= params['v_mito_rel']*Ca_mito
    return J_uptake, J_release

# =============================================================================
# ODE SYSTEM
# =============================================================================
def model(t, y, params):
    """
    Single pericyte ODE system.

    States:
      y[0]: V       (mV)
      y[1]: Ca_i    (cytosolic Ca, mM)
      y[2]: ATP     (mM)
      y[3]: Ca_ER   (mM)
      y[4]: IP3     (mM)
      y[5]: Ca_mito (mM)
      y[6]: h_ip3   (dimensionless IP3R inactivation)
      y[...] optional if you added time-dependent Nav gating, etc.

    We'll keep Nav gating as quasi-steady in code => no state for that.
    We'll do the same for TRPC3 activation if you want a slow variable 
    for DAG, but for demonstration, we treat it as param or a fixed fraction.
    """
    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3 = y

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1) Quasi-steady gating for Nav1.2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_nav, h_nav = nav1_2_steadystates(V, params)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2) TRPC3 activation assumption
    # e.g. if you want a dynamic 'TRPC3_frac' = param or a function of time
    # For now, read from param or set 0 => closed, 1 => open:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TRPC3_frac = params.get('TRPC3_frac_open', 0.0) 
    # e.g. set to 0.5 to partially open. 
    # In a real model, you'd define a separate ODE for DAG or Gq, or 
    # let IP3 production also produce DAG => TRPC3. This is simplified.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3) Membrane Currents
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ikir6 = kir6_1_current(V, ATP, params)          # KATP
    Ikir2 = kir2_2_current(V, params)               # Kir2.2
    Itrpc3= trpc3_current(V, TRPC3_frac, params)    # TRPC3
    Icav  = cav1_2_current(V, Ca_i, params)         # L-type
    Icacc = cacc_current(V, Ca_i, params)           # CaCC
    Ina   = nav1_2_current(V, m_nav, h_nav, params) # Nav1.2
    Ileak = leak_current(V, params)                 # minor channels

    # Summation
    I_total = Ikir6 + Ikir2 + Itrpc3 + Icav + Icacc + Ina + Ileak

    # dV/dt
    dVdt = -I_total / params['Cm']  # current / capacitance => dV/dt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4) Ca2+ fluxes: IP3R, SERCA, PMCA, ER leak, mito
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    J_ip3r  = ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params)
    J_pmca  = pmca_flux(Ca_i, params)
    J_serca = serca_flux(Ca_i, params)
    J_er_leak = params['v_ER_leak']*(Ca_ER - Ca_i)
    J_mito_uptake, J_mito_release = mito_fluxes(Ca_i, Ca_mito, params)

    # Ca2+ influx from membrane channels
    # e.g. fraction of Icav or Itrpc3 carried by Ca2+. 
    # We'll do a simpler version: 
    # J_Ca_in = -Icav - (f_trpc3_ca * Itrpc3), scaled by a factor => concentration 
    # For brevity, keep consistent with your earlier approach:
    f_trpc3_ca = params.get('f_trpc3_ca', 0.1)
    # We'll define a param => 'Ca_to_current_factor' for pA->uM/s if you want. 
    # For demonstration, we do the simpler approach:
    J_cav_in = -Icav*params['ca_current_factor']
    J_trpc_in= -Itrpc3*params['ca_current_factor']*f_trpc3_ca

    dCa_i_dt = (J_cav_in + J_trpc_in
                + J_ip3r 
                - J_pmca 
                - J_serca
                + J_er_leak
                + (J_mito_release - J_mito_uptake))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5) ATP dynamics => keep constant or define your own eq
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dATP_dt = 0.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6) ER Ca2+
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dCa_ER_dt = J_serca - J_ip3r - J_er_leak

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 7) IP3 => optional dynamic or constant
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If you want dynamic:
    #dIP3_dt = ip3_production_rate(Ca_i, params) - ip3_degradation_rate(IP3, params)
    dIP3_dt = 0.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 8) Mito Ca2+
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dCa_mito_dt = J_mito_uptake - J_mito_release

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 9) IP3R Inactivation gate
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dhdt = dh_ip3_dt(Ca_i, h_ip3, params)

    return [dVdt, dCa_i_dt, dATP_dt, dCa_ER_dt, dIP3_dt, dCa_mito_dt, dhdt]

# =============================================================================
# MAIN SIMULATION
# =============================================================================
if __name__ == "__main__":
    # -----------------------------
    # Parameter Dictionary
    # -----------------------------
    params = {
        # Membrane & Leak
        'Cm': 1.0,   # membrane capacitance (uF/cm^2)
        'g_leak_k': 0.01,   # small K+ leak
        'g_leak_ns': 0.005, # small nonselective cation leak
        'E_leak_ns': -20.0, # approximate reversal for NS cation
        # KATP
        'g_kir6_1': 0.1, 
        'ATP_half': 0.5,
        'ATP_hill': 2.0,
        # Kir2.2
        'g_kir2_2': 0.02,  # might be smaller than originally, because Kir6.1 is main
        'V_half_Kir2_2': -70,
        'slope_Kir2_2': 10,
        # TRPC3
        'g_trpc3': 0.1,
        'E_TRPC3': 0.0,
        'f_trpc3_ca': 0.1,  # fraction carried by Ca2+
        # Cav1.2
        'g_cav1_2': 0.1,
        'E_Ca': 60.0,    # typical Ca2+ rev
        'cav_d_half': -20.0,
        'cav_d_slope': 6.0,
        'cav_f_half': -40.0,
        'cav_f_slope': 6.0,
        # CaCC
        'g_cacc': 0.15,
        'E_Cl': -30.0,
        'ca_half_cacc': 0.001,  # 1 µM => 0.001 mM
        'ca_hill_cacc': 2.0,
        # Nav1.2
        'g_nav1_2': 0.05,
        'E_Na': 60.0,
        'nav_m_half': -25.0,
        'nav_m_slope': 5.0,
        'nav_h_half': -65.0,
        'nav_h_slope': 7.0,
        # IP3R (DYK)
        'd1': 0.13,
        'd2': 1.049,
        'd5': 0.082,
        'v_ip3r': 0.5,
        'tau_h': 2.0,
        # SERCA/PMCA
        'v_PMCA': 0.2,
        'K_PMCA': 0.0003,
        'v_SERCA': 0.5,
        'K_SERCA': 0.0003,
        # ER leak
        'v_ER_leak': 0.05,
        # Mito
        'v_mito': 0.05,
        'K_mito': 0.01,
        'v_mito_rel': 0.02,
        # Reversal potentials
        'E_K': -85.0,   # typical for pericyte
        # Extra factors
        'ca_current_factor': 0.01,  # scale to convert pA -> (mM/s), etc.
        # TRPC3 dynamic?
        'TRPC3_frac_open': 0.0,    # set to 0 for baseline, or >0 if stimulated
    }

    # -----------------------------
    # Initial Conditions
    # -----------------------------
    # [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3]
    y0 = [
        -60.0,  # V (mV)
        0.0001, # Ca_i ~100 nM
        1.0,    # ATP (mM)
        0.1,    # Ca_ER (mM)
        0.1,    # IP3 (mM), constant if desired
        0.0001, # Ca_mito (mM)
        0.8     # h_ip3
    ]

    # Simulation
    t_span = (0, 2000)
    t_eval = np.linspace(0, 2000, 2000)
    sol = solve_ivp(lambda t, y: model(t, y, params),
                    t_span, y0, t_eval=t_eval, method='LSODA')
    time = sol.t
    V_arr     = sol.y[0]
    Ca_i_arr  = sol.y[1]
    ATP_arr   = sol.y[2]
    Ca_ER_arr = sol.y[3]
    IP3_arr   = sol.y[4]
    Ca_mito_arr = sol.y[5]
    h_ip3_arr = sol.y[6]

    # ----------------------------------------------------------------
    # EXAMPLE: Recompute Currents & Fluxes for plotting 
    # (like your original approach)
    # ----------------------------------------------------------------
    # We'll store each current in a list
    Ikir6_list, Ikir2_list = [], []
    Itrpc3_list, Icav_list = [], []
    Icacc_list, Ina_list   = [], []
    Ileak_list            = []
    J_ip3r_list, J_pmca_list, J_serca_list = [], [], []
    J_er_leak_list        = []
    J_mito_up_list, J_mito_rel_list = [], []

    for i in range(len(time)):
        V_    = V_arr[i]
        Ca_   = Ca_i_arr[i]
        ATP_  = ATP_arr[i]
        ER_   = Ca_ER_arr[i]
        IP3_  = IP3_arr[i]
        M_    = Ca_mito_arr[i]
        h_    = h_ip3_arr[i]

        # Nav gating
        m_nav_, h_nav_ = nav1_2_steadystates(V_, params)

        Ik6_  = kir6_1_current(V_, ATP_, params)
        Ik2_  = kir2_2_current(V_, params)
        Itrp_ = trpc3_current(V_, params['TRPC3_frac_open'], params)
        Ica_  = cav1_2_current(V_, Ca_, params)
        Icl_  = cacc_current(V_, Ca_, params)
        Ina_  = nav1_2_current(V_, m_nav_, h_nav_, params)
        Il_   = leak_current(V_, params)

        Ikir6_list.append(Ik6_)
        Ikir2_list.append(Ik2_)
        Itrpc3_list.append(Itrp_)
        Icav_list.append(Ica_)
        Icacc_list.append(Icl_)
        Ina_list.append(Ina_)
        Ileak_list.append(Il_)

        # IP3R flux
        j_ip3_ = ip3r_flux(Ca_, ER_, IP3_, h_, params)
        J_ip3r_list.append(j_ip3_)
        # PMCA
        jpmca_ = pmca_flux(Ca_, params)
        J_pmca_list.append(jpmca_)
        # SERCA
        jser_  = serca_flux(Ca_, params)
        J_serca_list.append(jser_)
        # ER leak
        jleak_ = params['v_ER_leak']*(ER_ - Ca_)
        J_er_leak_list.append(jleak_)
        # Mito
        jup_, jrel_ = mito_fluxes(Ca_, M_, params)
        J_mito_up_list.append(jup_)
        J_mito_rel_list.append(jrel_)

    # Convert to arrays
    Ikir6_1_array  = np.array(Ikir6_list)
    Ikir2_2_array  = np.array(Ikir2_list)
    Itrpc3_array   = np.array(Itrpc3_list)
    Icav1_2_array  = np.array(Icav_list)
    Icacc_array    = np.array(Icacc_list)
    Ina_array      = np.array(Ina_list)
    Ileak_array    = np.array(Ileak_list)
    J_ip3r_array   = np.array(J_ip3r_list)
    J_pmca_array   = np.array(J_pmca_list)
    J_serca_array  = np.array(J_serca_list)
    J_er_leak_array= np.array(J_er_leak_list)
    J_mito_up_arr  = np.array(J_mito_up_list)
    J_mito_rel_arr = np.array(J_mito_rel_list)

    # ------------------------------------------------
    # LOG voltage every 10 points
    # ------------------------------------------------
    with open("new_single_voltage_log.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time_ms", "Voltage_mV"])
        for i in range(len(time)):
            if i % 10 == 0:
                writer.writerow([time[i], V_arr[i]])

    # ------------------------------------------------
    # PLOTS
    # ------------------------------------------------
    # Figure: Membrane Voltage
    plt.figure()
    plt.plot(time, V_arr, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel('Vm (mV)')
    plt.title('Membrane Potential')
    plt.tight_layout()

    # Cytosolic Ca
    plt.figure()
    plt.plot(time, Ca_i_arr, 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel('[Ca2+]_i (mM)')
    plt.title('Cytosolic Ca2+')
    plt.tight_layout()

    # ER & Mito Ca
    plt.figure()
    plt.plot(time, Ca_ER_arr, label='ER Ca')
    plt.plot(time, Ca_mito_arr, label='Mito Ca')
    plt.xlabel('Time (ms)')
    plt.ylabel('Concentration (mM)')
    plt.title('ER & Mito Ca2+')
    plt.legend()
    plt.tight_layout()

    # Channel Currents
    plt.figure(figsize=(15,10))
    plt.subplot(3,3,1)
    plt.plot(time, Ikir6_1_array); plt.title('I_Kir6.1')
    plt.subplot(3,3,2)
    plt.plot(time, Ikir2_2_array); plt.title('I_Kir2.2')
    plt.subplot(3,3,3)
    plt.plot(time, Itrpc3_array); plt.title('I_TRPC3')
    plt.subplot(3,3,4)
    plt.plot(time, Icav1_2_array); plt.title('I_Cav1.2')
    plt.subplot(3,3,5)
    plt.plot(time, Icacc_array); plt.title('I_CaCC')
    plt.subplot(3,3,6)
    plt.plot(time, Ina_array); plt.title('I_Na (Nav1.2)')
    plt.subplot(3,3,7)
    plt.plot(time, Ileak_array); plt.title('I_leak')
    plt.tight_layout()

    # Fluxes
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.plot(time, J_ip3r_array); plt.title('J_IP3R')
    plt.subplot(2,3,2)
    plt.plot(time, J_pmca_array); plt.title('J_PMCA')
    plt.subplot(2,3,3)
    plt.plot(time, J_serca_array); plt.title('J_SERCA')
    plt.subplot(2,3,4)
    plt.plot(time, J_er_leak_array); plt.title('ER Leak')
    plt.subplot(2,3,5)
    plt.plot(time, J_mito_up_arr); plt.title('Mito Uptake')
    plt.subplot(2,3,6)
    plt.plot(time, J_mito_rel_arr); plt.title('Mito Release')
    plt.tight_layout()

    plt.show()
