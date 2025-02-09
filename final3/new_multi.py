import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

# =============================================================================
# SAFE EXP FOR STABILITY
# =============================================================================
def safe_exp(x, clip_min=-100, clip_max=100):
    """
    Compute np.exp(x) with clipping to avoid overflow.
    """
    return np.exp(np.clip(x, clip_min, clip_max))

# =============================================================================
# MEMBRANE CURRENTS (KEEPING ALL 6 CHANNELS)
# =============================================================================
def calculate_membrane_currents(V, Ca_i, ATP, IP3, params):
    """
    Returns the 6 channel currents:
      I_Kir6.1, I_Kir2.2, I_TRPC3, I_Cav1.2, I_CaCC, I_Nav1.2
    in the same structure as your single.py
    """
    # --- Kir6.1 ---
    atp_factor = 1 / (1 + (ATP / params['ATP_half_Kir6_1'])**params['ATP_hill_Kir6_1'])
    I_Kir6_1 = params['conductance_Kir6_1'] * atp_factor * (V - params['reversal_potential_K'])

    # --- Kir2.2 ---
    arg_Kir2 = (V - params['V_half_Kir2_2']) / params['slope_Kir2_2']
    Kir2_activation = 1 / (1 + safe_exp(arg_Kir2))
    I_Kir2_2 = params['conductance_Kir2_2'] * Kir2_activation * (V - params['reversal_potential_K'])

    # --- TRPC3 ---
    TRPC3_activation = IP3 / (IP3 + params['IP3_half_TRPC3'] + 1e-12)
    I_TRPC3 = params['conductance_TRPC3'] * TRPC3_activation * (V - params['reversal_potential_TRPC3'])

    # --- Cav1.2 ---
    arg_d = -(V - params['activation_midpoint_Cav1_2']) / params['activation_slope_Cav1_2']
    d_inf = 1 / (1 + safe_exp(arg_d))
    arg_f = (V - params['inactivation_midpoint_Cav1_2']) / params['inactivation_slope_Cav1_2']
    f_inf = 1 / (1 + safe_exp(arg_f))
    I_Cav1_2 = params['conductance_Cav1_2'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])

    # --- CaCC ---
    CaCC_activation = Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'] + 1e-12)
    I_CaCC = params['conductance_CaCC'] * CaCC_activation * (V - params['reversal_potential_Cl'])

    # --- Nav1.2 ---
    arg_m = -(V - params['activation_midpoint_Nav1_2']) / params['activation_slope_Nav1_2']
    m_inf = 1 / (1 + safe_exp(arg_m))
    arg_h = (V - params['inactivation_midpoint_Nav1_2']) / params['inactivation_slope_Nav1_2']
    h_inf = 1 / (1 + safe_exp(arg_h))
    I_Nav1_2 = params['conductance_Nav1_2'] * (m_inf**3) * h_inf * (V - params['reversal_potential_Na'])

    return I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2

# =============================================================================
# IP3R (DE YOUNG–KEIZER), SERCA, PMCA, MITO
# =============================================================================
def calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params):
    d1 = params['d1']
    d5 = params['d5']
    # IP3 & Ca activation
    m_inf = IP3 / (IP3 + d1 + 1e-12)
    n_inf = Ca_i / (Ca_i + d5 + 1e-12)
    p_open = (m_inf**3) * (n_inf**3) * (h_IP3**3)
    J_IP3R = params['v_IP3R'] * p_open * (Ca_ER - Ca_i)
    return J_IP3R

def dh_ip3_dt(Ca_i, h_IP3, params):
    d2 = params['d2']
    tau_h = params['tau_h']
    h_inf = d2 / (d2 + Ca_i + 1e-12)
    return (h_inf - h_IP3) / tau_h

def calculate_PMCA(Ca_i, params):
    top = Ca_i
    bot = Ca_i + params['K_PMCA'] + 1e-12
    return params['v_PMCA'] * (top / bot)

def calculate_SERCA(Ca_i, params):
    top = Ca_i**2
    bot = Ca_i**2 + params['K_SERCA']**2 + 1e-30
    return params['v_SERCA']*(top/bot)

def calculate_mito_fluxes(Ca_i, Ca_mito, params):
    J_in  = params['v_mito'] * Ca_i/(Ca_i + params['K_mito'] + 1e-12)
    J_out = params['v_mito_rel'] * Ca_mito
    return J_in, J_out

# =============================================================================
# SINGLE-CELL ODE
# =============================================================================
def single_cell_ode(t, y, params, I_inj):
    """
    Single pericyte ODE for 7 states:
      [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3].
    I_inj: externally injected current for that cell.
    """
    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3 = y

    # 1) Membrane currents
    I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2 = calculate_membrane_currents(
        V, Ca_i, ATP, IP3, params
    )
    I_total = I_Kir6_1 + I_Kir2_2 + I_TRPC3 + I_Cav1_2 + I_CaCC + I_Nav1_2
    dVdt = ( - I_total + I_inj ) / params['membrane_capacitance']

    # 2) IP3R flux, SERCA, PMCA, ER leak, Mito
    J_IP3R = calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params)
    J_PMCA = calculate_PMCA(Ca_i, params)
    J_SERCA = calculate_SERCA(Ca_i, params)
    J_ER_leak = params['v_ER_leak']*(Ca_ER - Ca_i)
    J_mito_uptake, J_mito_release = calculate_mito_fluxes(Ca_i, Ca_mito, params)

    # 3) Ca influx from channels
    #   same logic as single.py: J_Ca_in = -(I_Cav1_2 + f_TRPC3_Ca*I_TRPC3)
    f_trpc3_ca = params.get('f_TRPC3_Ca', 0.1)
    J_Ca_in = - I_Cav1_2 - f_trpc3_ca*I_TRPC3

    # 4) dCa_i/dt
    dCa_i_dt = ( J_Ca_in
                + J_IP3R
                - J_PMCA
                - J_SERCA
                + J_ER_leak
                + (J_mito_release - J_mito_uptake) )

    # 5) keep ATP constant
    dATP_dt = 0.0

    # 6) dCa_ER
    dCa_ER_dt = J_SERCA - J_IP3R - J_ER_leak

    # 7) dIP3 (assumed constant => 0; you can enable dynamic IP3 if desired)
    dIP3_dt = 0.0

    # 8) Mito
    dCa_mito_dt = J_mito_uptake - J_mito_release

    # 9) IP3R inactivation gate
    dhdt = dh_ip3_dt(Ca_i, h_IP3, params)

    return [dVdt, dCa_i_dt, dATP_dt, dCa_ER_dt, dIP3_dt, dCa_mito_dt, dhdt]

# =============================================================================
# NETWORK ODE
# =============================================================================
def network_ode(t, Y, params, N, g_c, inj_params):
    """
    Y has length 7*N: for each of N cells,
      [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3].
    g_c => gap junction coupling (mS/cm^2) for membrane potential only.

    inj_params => dict with 
      'start_index', 'end_index', 'I_inj', 't_start', 't_end'
    for patch clamp injection in a range of cells.
    """
    dYdt = np.zeros_like(Y)
    n_vars = 7

    for i in range(N):
        idx_start = i*n_vars
        idx_end   = (i+1)*n_vars
        y_i = Y[idx_start:idx_end]

        # Decide I_inj for this cell, based on time & inj_params
        if (inj_params['start_index'] <= i <= inj_params['end_index']
            and inj_params['t_start'] <= t <= inj_params['t_end']):
            I_inj = inj_params['I_inj']
        else:
            I_inj = 0.0

        # Single-cell ODE
        dY_i = single_cell_ode(t, y_i, params, I_inj)

        # Gap junction coupling for V
        V_i = y_i[0]
        if i == 0:
            V_left = V_i  # or consider no boundary cell
        else:
            V_left = Y[(i-1)*n_vars]

        if i == N-1:
            V_right = V_i
        else:
            V_right = Y[(i+1)*n_vars]

        I_coup = g_c*((V_left - V_i) + (V_right - V_i))
        # Add to dVdt
        dY_i[0] += I_coup/params['membrane_capacitance']

        # store
        dYdt[idx_start:idx_end] = dY_i

    return dYdt

# =============================================================================
# MAIN SIMULATION
# =============================================================================
if __name__ == "__main__":
    # -- Number of cells
    N = 10
    g_c = 0.05  # gap junction conductance

    # -- Injection parameters (like your original)
    inj_params = {
        'start_index': 2,   # e.g. stimulate cells 2..5
        'end_index': 5,
        'I_inj': 0.05,      # (mA/cm^2)
        't_start': 200,     # ms
        't_end': 400,       # ms
    }

    # -- Model parameters
    params = {
        # Membrane
        'membrane_capacitance': 1.0,

        # Ion channels
        'conductance_Kir6_1': 0.1,
        'conductance_Kir2_2': 0.2,
        'conductance_TRPC3':  0.1,
        'conductance_Cav1_2': 0.1,
        'conductance_CaCC':   0.15,
        'conductance_Nav1_2': 0.1,

        # Reversals
        'reversal_potential_K':  -85,
        'reversal_potential_TRPC3': 0,
        'reversal_potential_Ca': 120,
        'reversal_potential_Cl': -30,
        'reversal_potential_Na': 60,

        # KATP
        'ATP_half_Kir6_1': 0.5,
        'ATP_hill_Kir6_1': 2,
        # Kir2.2
        'V_half_Kir2_2': -70,
        'slope_Kir2_2': 10,
        # TRPC3
        'IP3_half_TRPC3': 0.2,
        # Cav1.2 gating
        'activation_midpoint_Cav1_2': -20,
        'activation_slope_Cav1_2': 6,
        'inactivation_midpoint_Cav1_2': -40,
        'inactivation_slope_Cav1_2': 6,
        # CaCC
        'calcium_activation_threshold_CaCC': 0.0005,
        # Nav1.2 gating
        'activation_midpoint_Nav1_2': -35,
        'activation_slope_Nav1_2': 7,
        'inactivation_midpoint_Nav1_2': -60,
        'inactivation_slope_Nav1_2': 5,

        # IP3R
        'd1': 0.13,
        'd2': 1.049,
        'd5': 0.082,
        'v_IP3R': 0.5,
        'tau_h': 2.0,

        # Pumps & leaks
        'v_PMCA': 0.2,
        'K_PMCA': 0.0003,
        'v_SERCA': 0.5,
        'K_SERCA': 0.0003,
        'v_ER_leak': 0.05,

        # Mito
        'v_mito': 0.05,
        'K_mito': 0.01,
        'v_mito_rel': 0.02,

        # Fraction TRPC3 -> Ca
        'f_TRPC3_Ca': 0.1,
    }

    # -- Initial conditions per cell
    # [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3]
    # e.g. start near -30 mV, Ca_i=100 nM, etc.
    y0_cell = [-30.0, 0.0001, 1.0, 0.1, 0.1, 0.0001, 0.8]
    Y0 = np.tile(y0_cell, N)

    # -- Time span
    t_span = (0, 1000)
    t_eval = np.linspace(0, 1000, 2000)

    # -- Solve
    sol = solve_ivp(
        lambda t, Y: network_ode(t, Y, params, N, g_c, inj_params),
        t_span, Y0, t_eval=t_eval, method='BDF'
    )

    time = sol.t
    Y_sol = sol.y  # shape (7*N, len(time))

    # ---------------------------------------------------------------------
    # Extract membrane potential for each cell
    # ---------------------------------------------------------------------
    n_vars = 7
    V_traces = []
    for i in range(N):
        V_traces.append(Y_sol[i*n_vars, :])  # V for cell i

    # ---------------------------------------------------------------------
    # Logging: store time, V of cell 0 every 10 steps
    # ---------------------------------------------------------------------
    with open("new_multi_voltage_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "Cell0_V_mV"])
        for idx in range(len(time)):
            if idx % 10 == 0:
                writer.writerow([time[idx], V_traces[0][idx]])

    # ---------------------------------------------------------------------
    # Plot the network membrane potentials
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(time, V_traces[i], label=f'Cell {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Network Pericyte Model: Membrane Voltages')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # If you want additional plots (like Ca_i for each cell),
    # you can replicate the approach above. For example:
    # Ca_traces = []
    # for i in range(N):
    #     Ca_traces.append(Y_sol[i*n_vars+1, :])  # Ca_i for cell i
    #
    # plt.figure(figsize=(10, 6))
    # for i in range(N):
    #     plt.plot(time, Ca_traces[i], label=f'Cell {i} Ca_i')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Cytosolic [Ca2+] (mM)')
    # plt.title('Cytosolic Ca2+ in Each Pericyte')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
