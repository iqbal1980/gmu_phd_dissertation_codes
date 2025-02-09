"""
new_multi.py

Multi-cell pericyte network with the same refined channel model:
  - Kir6.1 (KATP), Kir2.2, TRPC3, Cav1.2, CaCC, Nav1.2
  - IP3R-based ER release, SERCA/PMCA, Mito Ca2+ flux
  - Consolidated leak for other channels
  - Gap-junction coupling for membrane potential only
  - Optional patch-clamp injection

We assume each cell has 7 states:
   [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3].

Author: [Your Name]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

def safe_exp(x, minclip=-100, maxclip=100):
    return np.exp(np.clip(x, minclip, maxclip))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reuse or replicate the single-cell channel functions with final parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def kir6_1_current(V, ATP, params):
    atp_factor = 1.0/(1.0 + (ATP/params['ATP_half'])**params['ATP_hill'])
    return params['g_kir6_1']*atp_factor*(V - params['E_K'])

def kir2_2_current(V, params):
    arg = (V - params['V_half_Kir2_2'])/params['slope_Kir2_2']
    act = 1.0/(1.0 + safe_exp(arg))
    return params['g_kir2_2']*act*(V - params['E_K'])

def trpc3_current(V, frac_open, params):
    return params['g_trpc3']*frac_open*(V - params['E_TRPC3'])

def cav1_2_current(V, Ca_i, params):
    arg_d = -(V - params['cav_d_half'])/params['cav_d_slope']
    d_inf = 1.0/(1.0 + safe_exp(arg_d))
    arg_f = (V - params['cav_f_half'])/params['cav_f_slope']
    f_inf = 1.0/(1.0 + safe_exp(arg_f))
    return params['g_cav1_2']*d_inf*f_inf*(V - params['E_Ca'])

def cacc_current(V, Ca_i, params):
    top = Ca_i**params['ca_hill_cacc']
    bot = top + params['ca_half_cacc']**params['ca_hill_cacc'] + 1e-30
    act = top/bot
    return params['g_cacc']*act*(V - params['E_Cl'])

def nav12_steadystates(V, params):
    arg_m = -(V - params['nav_m_half'])/params['nav_m_slope']
    m_inf = 1.0/(1.0 + safe_exp(arg_m))
    arg_h = (V - params['nav_h_half'])/params['nav_h_slope']
    h_inf = 1.0/(1.0 + safe_exp(arg_h))
    return m_inf, h_inf

def nav12_current(V, m_nav, h_nav, params):
    return params['g_nav1_2']*(m_nav**3)*h_nav*(V - params['E_Na'])

def leak_current(V, params):
    Ik_leak = params['g_leak_k']*(V - params['E_K'])
    Ins_leak= params['g_leak_ns']*(V - params['E_leak_ns'])
    return Ik_leak + Ins_leak

# IP3R, SERCA, PMCA, Mito
def ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params):
    d1 = params['d1']
    d5 = params['d5']
    m_inf = IP3/(IP3 + d1 + 1e-12)
    n_inf = Ca_i/(Ca_i + d5 + 1e-12)
    p_open= (m_inf**3)*(n_inf**3)*(h_ip3**3)
    return params['v_ip3r']*p_open*(Ca_ER - Ca_i)

def dh_ip3_dt(Ca_i, h_ip3, params):
    d2 = params['d2']
    tau_h= params['tau_h']
    h_inf= d2/(d2 + Ca_i +1e-12)
    return (h_inf - h_ip3)/tau_h

def pmca_flux(Ca_i, params):
    top = Ca_i
    bot = Ca_i + params['K_PMCA'] + 1e-30
    return params['v_PMCA']*(top/bot)

def serca_flux(Ca_i, params):
    top = Ca_i**2
    bot = Ca_i**2 + params['K_SERCA']**2 + 1e-30
    return params['v_SERCA']*(top/bot)

def mito_fluxes(Ca_i, Ca_mito, params):
    J_up = params['v_mito']*(Ca_i/(Ca_i + params['K_mito'] +1e-12))
    J_rel= params['v_mito_rel']*Ca_mito
    return J_up, J_rel

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SINGLE-CELL ODE for each pericyte
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def single_cell_ode(t, y, params):
    """
    y => [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3].
    We'll keep TRPC3 fraction as a constant param or zero. 
    If you want dynamic, add an ODE for DAG or so.
    """
    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3 = y

    # 1) Nav gating (quasi-steady)
    m_nav, h_nav = nav12_steadystates(V, params)
    # 2) TRPC3 open fraction
    frac_trpc3 = params.get('TRPC3_frac_open', 0.0)

    # Currents
    Ikatp = kir6_1_current(V, ATP, params)
    Ikir2 = kir2_2_current(V, params)
    Itrpc = trpc3_current(V, frac_trpc3, params)
    Icav  = cav1_2_current(V, Ca_i, params)
    Icacc = cacc_current(V, Ca_i, params)
    Ina   = nav12_current(V, m_nav, h_nav, params)
    Ileak_ = leak_current(V, params)

    I_total = Ikatp + Ikir2 + Itrpc + Icav + Icacc + Ina + Ileak_

    dVdt = - I_total / params['Cm']

    # IP3R flux
    J_ip3r_ = ip3r_flux(Ca_i, Ca_ER, IP3, h_ip3, params)
    # PMCA
    J_pmca_ = pmca_flux(Ca_i, params)
    # SERCA
    J_serca_= serca_flux(Ca_i, params)
    # ER leak
    J_leak_ = params['v_ER_leak']*(Ca_ER - Ca_i)
    # Mito
    J_up, J_rel = mito_fluxes(Ca_i, Ca_mito, params)

    # Ca2+ from Cav + partial TRPC3 => we approximate 
    J_cav_in = -Icav*params['ca_current_factor']
    J_trpc_in= -Itrpc*params['ca_current_factor']*params.get('f_trpc3_ca', 0.1)

    dCa_i = (J_cav_in + J_trpc_in
             + J_ip3r_ 
             - J_pmca_
             - J_serca_
             + J_leak_
             + (J_rel - J_up))

    dATP = 0.0  # keep constant unless you'd like an ATP eq
    dCa_ER= J_serca_ - J_ip3r_ - J_leak_
    # If you want dynamic IP3 => define ip3_production_rate, ip3_degradation_rate
    dIP3  = 0.0
    dCa_mito= J_up - J_rel
    dh_ip3_= dh_ip3_dt(Ca_i, h_ip3, params)

    return [dVdt, dCa_i, dATP, dCa_ER, dIP3, dCa_mito, dh_ip3_]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NETWORK ODE with gap junction coupling 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def network_ode(t, Y, params, N, g_gap, inj_dict):
    """
    Y => concatenated states for N cells: 7*N length.
    g_gap => coupling conductance (mS/cm^2 or similar).
    inj_dict => { 'start_index', 'end_index', 'I_inj', 't_start','t_end'}

    We apply gap junction coupling to the membrane potential only:
    dV_i/dt += (g_gap*(V_left - V_i) + g_gap*(V_right - V_i))/Cm 
    """
    dYdt = np.zeros_like(Y)
    nvar = 7
    for i in range(N):
        idx0 = i*nvar
        idx1 = idx0 + nvar
        y_i = Y[idx0:idx1]

        # Decide injection current
        if (inj_dict['start_index'] <= i <= inj_dict['end_index']
            and inj_dict['t_start'] <= t <= inj_dict['t_end']):
            I_inj = inj_dict['I_inj']
        else:
            I_inj = 0.0

        # Single-cell ODE
        dydt_i = single_cell_ode(t, y_i, params)

        # Gap-junction coupling for V
        V_i = y_i[0]
        if i == 0:
            V_left = V_i
        else:
            V_left = Y[(i-1)*nvar]

        if i == N-1:
            V_right = V_i
        else:
            V_right = Y[(i+1)*nvar]

        I_coup = g_gap*((V_left - V_i) + (V_right - V_i))
        # Add to dVdt
        dydt_i[0] += I_coup / params['Cm']
        # Also add any injection current
        dydt_i[0] += I_inj / params['Cm']

        dYdt[idx0:idx1] = dydt_i

    return dYdt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    N = 5
    g_gap = 0.05  # gap junction

    inj_dict = {
        'start_index': 1,
        'end_index': 2,
        'I_inj': 0.05,  # mA/cm^2
        't_start': 200,
        't_end': 400
    }

    params = {
        # Membrane & leaks
        'Cm': 1.0,
        'g_leak_k': 0.01,
        'g_leak_ns': 0.01,
        'E_leak_ns': -20.0,
        # KATP
        'g_kir6_1': 0.1,
        'ATP_half': 0.5,
        'ATP_hill': 2.0,
        # Kir2.2
        'g_kir2_2': 0.02,
        'V_half_Kir2_2': -70,
        'slope_Kir2_2': 10,
        # TRPC3
        'g_trpc3': 0.1,
        'E_TRPC3': 0.0,
        'TRPC3_frac_open': 0.0,
        'f_trpc3_ca': 0.1,
        # Cav1.2
        'g_cav1_2': 0.1,
        'E_Ca': 60.0,
        'cav_d_half': -20.0,
        'cav_d_slope': 6.0,
        'cav_f_half': -40.0,
        'cav_f_slope': 6.0,
        # CaCC
        'g_cacc': 0.15,
        'E_Cl': -30.0,
        'ca_half_cacc': 0.001, 
        'ca_hill_cacc': 2.0,
        # Nav1.2
        'g_nav1_2': 0.05,
        'E_Na': 60.0,
        'nav_m_half': -25.0,
        'nav_m_slope': 5.0,
        'nav_h_half': -65.0,
        'nav_h_slope': 7.0,
        # IP3R
        'd1': 0.13,
        'd2': 1.049,
        'd5': 0.082,
        'v_ip3r': 0.5,
        'tau_h': 2.0,
        # SERCA / PMCA
        'v_PMCA': 0.2,
        'K_PMCA': 0.0003,
        'v_SERCA': 0.5,
        'K_SERCA': 0.0003,
        'v_ER_leak': 0.05,
        # Mito
        'v_mito': 0.05,
        'K_mito': 0.01,
        'v_mito_rel': 0.02,
        # Reversals
        'E_K': -85.0,
        # Factor for Ca2+ from currents
        'ca_current_factor': 0.01
    }

    # Each cell: [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_ip3]
    y0_cell = [-60.0, 0.0001, 1.0, 0.1, 0.1, 0.0001, 0.8]
    Y0 = np.tile(y0_cell, N)

    t_span = (0, 1500)
    t_eval = np.linspace(0, 1500, 3000)

    sol = solve_ivp(lambda t, y: network_ode(t, y, params, N, g_gap, inj_dict),
                    t_span, Y0, t_eval=t_eval, method='LSODA')

    time = sol.t
    Y_sol = sol.y

    # Extract V from each cell
    nvar = 7
    V_cells = []
    for i in range(N):
        V_cells.append( Y_sol[i*nvar, :] )

    # Log cell0 V every 10 steps
    with open("new_multi_voltage_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "Cell0_V"])
        for i in range(len(time)):
            if i % 10 == 0:
                writer.writerow([time[i], V_cells[0][i]])

    # Plot
    plt.figure(figsize=(10,6))
    for i in range(N):
        plt.plot(time, V_cells[i], label=f'Cell {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Pericyte Network: Membrane Potentials')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # If you want to plot e.g. Ca_i for each cell, do similarly:
    # (just be mindful of large overhead if N is big)
