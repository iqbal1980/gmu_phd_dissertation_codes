import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv

# =============================================================================
# HELPER FUNCTIONS FOR IP3 DYNAMICS (OPTIONAL)
# =============================================================================
def ip3_production_rate(Ca_i, params):
    """
    Example: IP3 production = ip3_prod_base + ip3_prod_ca_factor * f(Ca)
    If you want purely constant IP3, set ip3_prod_ca_factor=0 or comment out.
    """
    base = params.get('ip3_prod_base', 0.0)  # constant baseline
    ca_factor = params.get('ip3_prod_ca_factor', 0.0)
    Kca = params.get('ip3_prod_ca_K', 0.5)    # example half-activation
    hill = params.get('ip3_prod_ca_hill', 2.0)

    # Simple Hill function for Ca feedback:
    f_ca = (Ca_i**hill) / (Ca_i**hill + Kca**hill + 1e-30)
    return base + ca_factor*f_ca

def ip3_degradation_rate(IP3, params):
    """
    First-order IP3 degradation with rate k_deg:
      dIP3/dt (degradation) = k_deg * IP3
    """
    k_deg = params.get('ip3_deg_k', 0.0)
    return k_deg * IP3


# =============================================================================
# MEMBRANE CURRENT FUNCTIONS (FROM ORIGINAL + KEEPS ALL CHANNELS)
# =============================================================================
def calculate_membrane_currents(V, Ca_i, ATP, IP3, params):
    """
    Same as in your original single.py, preserving Kir6.1, Kir2.2, TRPC3,
    Cav1.2, CaCC, and Nav1.2.
    """
    # Kir6.1
    atp_factor = 1 / (1 + (ATP / params['ATP_half_Kir6_1'])**params['ATP_hill_Kir6_1'])
    I_Kir6_1 = params['conductance_Kir6_1'] * atp_factor * (V - params['reversal_potential_K'])
    
    # Kir2.2
    Kir2_activation = 1 / (1 + np.exp((V - params['V_half_Kir2_2']) / params['slope_Kir2_2']))
    I_Kir2_2 = params['conductance_Kir2_2'] * Kir2_activation * (V - params['reversal_potential_K'])
    
    # TRPC3
    TRPC3_activation = IP3 / (IP3 + params['IP3_half_TRPC3'] + 1e-12)
    I_TRPC3 = params['conductance_TRPC3'] * TRPC3_activation * (V - params['reversal_potential_TRPC3'])
    
    # Cav1.2
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_Cav1_2']) / params['activation_slope_Cav1_2']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_Cav1_2']) / params['inactivation_slope_Cav1_2']))
    I_Cav1_2 = params['conductance_Cav1_2'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    
    # CaCC
    CaCC_activation = Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'] + 1e-12)
    I_CaCC = params['conductance_CaCC'] * CaCC_activation * (V - params['reversal_potential_Cl'])
    
    # Nav1.2
    m_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_Nav1_2']) / params['activation_slope_Nav1_2']))
    h_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_Nav1_2']) / params['inactivation_slope_Nav1_2']))
    I_Nav1_2 = params['conductance_Nav1_2'] * (m_inf**3) * h_inf * (V - params['reversal_potential_Na'])
    
    return I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2

# =============================================================================
# IP3R1 FLUX AND INACTIVATION DYNAMICS (De Young–Keizer)
# =============================================================================
def calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params):
    """
    Same as your original single.py, but carefully handle small denominators.
    """
    d1 = params['d1']
    d5 = params['d5']
    m_inf = IP3 / (IP3 + d1 + 1e-12)
    n_inf = Ca_i / (Ca_i + d5 + 1e-12)
    p_open = (m_inf**3) * (n_inf**3) * (h_IP3**3)
    J_IP3R = params['v_IP3R'] * p_open * (Ca_ER - Ca_i)
    return J_IP3R

def dh_ip3_dt(Ca_i, h_IP3, params):
    """
    Same as original: De Young–Keizer or Li-Rinzel style.
    """
    d2    = params['d2']
    tau_h = params['tau_h']
    # Avoid dividing by zero
    h_inf = d2 / (d2 + Ca_i + 1e-12)
    return (h_inf - h_IP3) / tau_h

# =============================================================================
# PMCA, SERCA, and Mitochondrial Fluxes (Same as original or improved)
# =============================================================================
def calculate_PMCA(Ca_i, params):
    """Plasma membrane Ca2+ ATPase (PMCA)."""
    J_PMCA = params['v_PMCA'] * (Ca_i / (Ca_i + params['K_PMCA'] + 1e-12))
    return J_PMCA

def calculate_SERCA(Ca_i, params):
    """SERCA flux."""
    top = (Ca_i**2)
    bot = (Ca_i**2 + params['K_SERCA']**2 + 1e-30)
    return params['v_SERCA'] * (top / bot)

def calculate_mito_fluxes(Ca_i, Ca_mito, params):
    """
    Mito uptake & release.
    """
    # uptake
    J_mito_uptake = params['v_mito'] * Ca_i / (Ca_i + params['K_mito'] + 1e-12)
    # release
    J_mito_release = params['v_mito_rel'] * Ca_mito
    return J_mito_uptake, J_mito_release

# =============================================================================
# ODE SYSTEM: FULL PERICYTE MODEL
# =============================================================================
def model(t, y, params):
    """
    ODE system for the single-cell pericyte model with full channels + IP3R/ER/mito.
    
    State variables:
      y[0] : V        (mV)
      y[1] : Ca_i     (mM)
      y[2] : ATP      (mM)
      y[3] : Ca_ER    (mM)
      y[4] : IP3      (mM)
      y[5] : Ca_mito  (mM)
      y[6] : h_IP3    (dimensionless)
    """
    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3 = y

    # 1) Membrane Currents
    I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2 = calculate_membrane_currents(
        V, Ca_i, ATP, IP3, params
    )
    I_total = I_Kir6_1 + I_Kir2_2 + I_TRPC3 + I_Cav1_2 + I_CaCC + I_Nav1_2

    dV_dt = -I_total / params['membrane_capacitance']

    # 2) Ca2+ fluxes: IP3R, PMCA, SERCA, ER leak, Mito uptake/release
    J_IP3R = calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params)
    J_PMCA = calculate_PMCA(Ca_i, params)
    J_SERCA = calculate_SERCA(Ca_i, params)
    J_ER_leak = params['v_ER_leak'] * (Ca_ER - Ca_i)
    J_mito_uptake, J_mito_release = calculate_mito_fluxes(Ca_i, Ca_mito, params)

    # Ca2+ entry via channels: (Cav1.2 + fraction of TRPC3)
    f_TRPC3_Ca = params.get('f_TRPC3_Ca', 0.1)
    # Negative sign because inward current => increase Ca_i
    # But in your original code, you had: J_Ca_in = -I_Cav1_2 - f_TRPC3_Ca * I_TRPC3
    # We'll keep that logic:
    J_Ca_in = - I_Cav1_2 - f_TRPC3_Ca * I_TRPC3

    dCa_i_dt = (J_Ca_in
                + J_IP3R
                - J_SERCA
                - J_PMCA
                + J_ER_leak
                + (J_mito_release - J_mito_uptake))

    # 3) ATP dynamics (kept constant unless you want to add an eq)
    dATP_dt = 0.0

    # 4) ER Ca2+ 
    dCa_ER_dt = J_SERCA - J_IP3R - J_ER_leak

    # 5) IP3 dynamics: If you want IP3 = constant, set dIP3_dt = 0
    # If you want dynamic IP3, uncomment:
    # dIP3_dt = ip3_production_rate(Ca_i, params) - ip3_degradation_rate(IP3, params)
    dIP3_dt = 0.0   # default to constant, same as your original

    # 6) Mito Ca2+
    dCa_mito_dt = J_mito_uptake - J_mito_release

    # 7) IP3R inactivation gating
    dh_IP3_dt_val = dh_ip3_dt(Ca_i, h_IP3, params)

    return [dV_dt, dCa_i_dt, dATP_dt, dCa_ER_dt, dIP3_dt, dCa_mito_dt, dh_IP3_dt_val]


# =============================================================================
# MAIN SIMULATION (Keeping your original style)
# =============================================================================
if __name__ == "__main__":
    # -- Parameter Dictionary (SAME + new optional IP3 params) --
    params = {
        # Membrane
        'membrane_capacitance': 1.0,

        # Ion Channel Conductances
        'conductance_Kir6_1': 0.1,
        'conductance_Kir2_2': 0.2,
        'conductance_TRPC3': 0.1,
        'conductance_Cav1_2': 0.1,
        'conductance_CaCC': 0.15,
        'conductance_Nav1_2': 0.1,

        # Reversal potentials
        'reversal_potential_K':  -85,
        'reversal_potential_TRPC3': 0.0,
        'reversal_potential_Ca': 120,
        'reversal_potential_Cl': -30,
        'reversal_potential_Na': 60,

        # Kir6.1
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
        'calcium_activation_threshold_CaCC': 0.0005,  # 0.5 µM
        # Nav1.2 gating
        'activation_midpoint_Nav1_2': -35,
        'activation_slope_Nav1_2': 7,
        'inactivation_midpoint_Nav1_2': -60,
        'inactivation_slope_Nav1_2': 5,

        # De Young–Keizer
        'd1': 0.13,
        'd2': 1.049,
        'd5': 0.082,
        'v_IP3R': 0.5,
        'tau_h': 2.0,

        # PMCA
        'v_PMCA': 0.2,
        'K_PMCA': 0.0003,

        # SERCA
        'v_SERCA': 0.5,
        'K_SERCA': 0.0003,

        # ER leak
        'v_ER_leak': 0.05,

        # Mito
        'v_mito': 0.05,
        'K_mito': 0.01,
        'v_mito_rel': 0.02,

        # Fraction TRPC3 -> Ca
        'f_TRPC3_Ca': 0.1,

        # Optional IP3 dynamic parameters (currently not used by default)
        'ip3_prod_base': 0.01,
        'ip3_prod_ca_factor': 0.0,
        'ip3_prod_ca_K': 0.5,
        'ip3_prod_ca_hill': 2.0,
        'ip3_deg_k': 0.01,
    }

    # -- Initial Conditions: [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3]
    y0 = [
        -70,      # V (mV)
        0.0001,   # Ca_i (mM)
        1.0,      # ATP (mM)
        0.1,      # Ca_ER (mM)
        0.1,      # IP3 (mM)
        0.0001,   # Ca_mito (mM)
        0.8       # h_IP3
    ]

    # -- Time Span (ms) & Solve
    t_span = (0, 1000)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(lambda t, y: model(t, y, params), t_span, y0,
                    t_eval=t_eval, method='RK45')

    time = sol.t
    V_array = sol.y[0]
    Ca_i_array = sol.y[1]
    ATP_array = sol.y[2]
    Ca_ER_array = sol.y[3]
    IP3_array = sol.y[4]
    Ca_mito_array = sol.y[5]
    h_IP3_array = sol.y[6]

    # -------------------------------------------------------------------------
    # Post-Processing: Compute Membrane Currents & Intracellular Fluxes
    # (Same structure as your original)
    # -------------------------------------------------------------------------
    I_Kir6_1_list, I_Kir2_2_list = [], []
    I_TRPC3_list, I_Cav1_2_list = [], []
    I_CaCC_list, I_Nav1_2_list = [], []

    J_IP3R_list, J_PMCA_list = [], []
    J_SERCA_list, J_ER_leak_list = [], []
    J_mito_uptake_list, J_mito_release_list = [], []

    for i in range(len(time)):
        V_ = V_array[i]
        Ca_ = Ca_i_array[i]
        ATP_ = ATP_array[i]
        ER_ = Ca_ER_array[i]
        IP3_ = IP3_array[i]
        M_ = Ca_mito_array[i]
        h_ = h_IP3_array[i]

        # Currents
        Ik6, Ik2, Itrpc3, Icav, Icacc, Inav = calculate_membrane_currents(V_, Ca_, ATP_, IP3_, params)
        I_Kir6_1_list.append(Ik6)
        I_Kir2_2_list.append(Ik2)
        I_TRPC3_list.append(Itrpc3)
        I_Cav1_2_list.append(Icav)
        I_CaCC_list.append(Icacc)
        I_Nav1_2_list.append(Inav)

        # Fluxes
        # IP3R
        jip3r = calculate_ip3r_flux(Ca_, ER_, IP3_, h_, params)
        J_IP3R_list.append(jip3r)
        # PMCA
        jpmca = calculate_PMCA(Ca_, params)
        J_PMCA_list.append(jpmca)
        # SERCA
        jser = calculate_SERCA(Ca_, params)
        J_SERCA_list.append(jser)
        # ER leak
        jleak = params['v_ER_leak']*(ER_ - Ca_)
        J_ER_leak_list.append(jleak)
        # Mito
        jup, jrel = calculate_mito_fluxes(Ca_, M_, params)
        J_mito_uptake_list.append(jup)
        J_mito_release_list.append(jrel)

    # Convert to np arrays for plotting
    I_Kir6_1_array = np.array(I_Kir6_1_list)
    I_Kir2_2_array = np.array(I_Kir2_2_list)
    I_TRPC3_array  = np.array(I_TRPC3_list)
    I_Cav1_2_array = np.array(I_Cav1_2_list)
    I_CaCC_array   = np.array(I_CaCC_list)
    I_Nav1_2_array = np.array(I_Nav1_2_list)

    J_IP3R_array = np.array(J_IP3R_list)
    J_PMCA_array = np.array(J_PMCA_list)
    J_SERCA_array= np.array(J_SERCA_list)
    J_ER_leak_array = np.array(J_ER_leak_list)
    J_mito_uptake_array  = np.array(J_mito_uptake_list)
    J_mito_release_array = np.array(J_mito_release_list)

    # -------------------------------------------------------------------------
    # LOGGING: e.g. log every 10th voltage
    # -------------------------------------------------------------------------
    with open("new_single_voltage_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "V_mV"])
        for i in range(len(time)):
            if i % 10 == 0:
                writer.writerow([time[i], V_array[i]])

    # -------------------------------------------------------------------------
    # PLOTTING (Same as your original with multiple figs)
    # -------------------------------------------------------------------------
    # Figure 1: Membrane Voltage
    plt.figure(figsize=(8, 4))
    plt.plot(time, V_array, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    plt.title('Membrane Voltage')
    plt.tight_layout()

    # Figure 2: Cytosolic Calcium
    plt.figure(figsize=(8, 4))
    plt.plot(time, Ca_i_array, 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel('Cytosolic [Ca²⁺] (mM)')
    plt.title('Cytosolic Calcium')
    plt.tight_layout()

    # Figure 3: ER Calcium
    plt.figure(figsize=(8, 4))
    plt.plot(time, Ca_ER_array, 'g')
    plt.xlabel('Time (ms)')
    plt.ylabel('ER [Ca²⁺] (mM)')
    plt.title('ER Calcium')
    plt.tight_layout()

    # Figure 4: Mitochondrial Calcium
    plt.figure(figsize=(8, 4))
    plt.plot(time, Ca_mito_array, 'm')
    plt.xlabel('Time (ms)')
    plt.ylabel('Mitochondrial [Ca²⁺] (mM)')
    plt.title('Mitochondrial Calcium')
    plt.tight_layout()

    # Figure 5: Membrane Channel Currents (6 channels)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(time, I_Kir6_1_array, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_Kir6.1')
    plt.title('Kir6.1 Current')

    plt.subplot(2, 3, 2)
    plt.plot(time, I_Kir2_2_array, 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_Kir2.2')
    plt.title('Kir2.2 Current')

    plt.subplot(2, 3, 3)
    plt.plot(time, I_TRPC3_array, 'g')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_TRPC3')
    plt.title('TRPC3 Current')

    plt.subplot(2, 3, 4)
    plt.plot(time, I_Cav1_2_array, 'c')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_Cav1.2')
    plt.title('Cav1.2 Current')

    plt.subplot(2, 3, 5)
    plt.plot(time, I_CaCC_array, 'm')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_CaCC')
    plt.title('CaCC Current')

    plt.subplot(2, 3, 6)
    plt.plot(time, I_Nav1_2_array, 'k')
    plt.xlabel('Time (ms)')
    plt.ylabel('I_Nav1.2')
    plt.title('Nav1.2 Current')
    plt.tight_layout()

    # Figure 6: Intracellular Fluxes
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(time, J_IP3R_array, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_IP3R')
    plt.title('IP₃R Flux')

    plt.subplot(2, 3, 2)
    plt.plot(time, J_PMCA_array, 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_PMCA')
    plt.title('PMCA Flux')

    plt.subplot(2, 3, 3)
    plt.plot(time, J_SERCA_array, 'g')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_SERCA')
    plt.title('SERCA Flux')

    plt.subplot(2, 3, 4)
    plt.plot(time, J_ER_leak_array, 'c')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_ER_leak')
    plt.title('ER Leak Flux')

    plt.subplot(2, 3, 5)
    plt.plot(time, J_mito_uptake_array, 'm')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_mito_uptake')
    plt.title('Mitochondrial Uptake')

    plt.subplot(2, 3, 6)
    plt.plot(time, J_mito_release_array, 'k')
    plt.xlabel('Time (ms)')
    plt.ylabel('J_mito_release')
    plt.title('Mitochondrial Release')
    plt.tight_layout()

    plt.show()
