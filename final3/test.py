import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTION TO COMPUTE EFFECTIVE CONDUCTANCE FROM RNA DATA
# =============================================================================
def compute_effective_conductance(mrna, base_pS, efficiency, stoichiometry, scaling_factor):
    """
    Compute the effective conductance (in mS/cm²) for a channel based on:
      - mrna: Average mRNA counts per cell (from CSV)
      - base_pS: Single-channel conductance in picoSiemens (pS; use the average if a range is given)
      - efficiency: Translation efficiency (fraction of mRNA that becomes functional protein)
      - stoichiometry: Number of subunits per channel (divide by this number)
      - scaling_factor: A conversion factor to scale the computed value into mS/cm²
    Conversion:
      1 pS = 1e-12 S, and 1 S = 1e3 mS, so 1 pS = 1e-9 mS.
    """
    base_conductance_mS = base_pS * 1e-9   # convert pS to mS
    effective_channels = mrna * efficiency / stoichiometry
    return base_conductance_mS * effective_channels * scaling_factor

# =============================================================================
# USER–DEFINED CONSTANTS FOR RNA–BASED CONDUCTANCE CALCULATIONS
# =============================================================================
# (These values can be adjusted.)
translation_efficiency = 0.1   # e.g., 10% of mRNA is translated into functional protein
scaling_factor = 6.4e4         # Global scaling factor to bring effective conductances into proper range

# Stoichiometry (number of subunits per channel) for our channels:
stoich = {
    'Kir6.1': 4,
    'Kir2.2': 4,
    'TRPC3': 4,
    'Cav1.2': 1,  # We assume the alpha subunit determines conduction
    'CaCC': 2,    # TMEM16A is thought to function as a dimer (or dimer-of-dimers; adjust as needed)
    'Nav1.2': 1
}

# RNA expression (mRNA counts) and single-channel conductances (in pS; average if range is given)
# Values extracted from the CSV (use average where a range is provided):
rna_data = {
    'Kir6.1':   {'mrna': 1670.21, 'base_pS': (35 + 40) / 2},   # 37.5 pS
    'Kir2.2':   {'mrna': 31.01,   'base_pS': (30 + 35) / 2},   # 32.5 pS
    'TRPC3':    {'mrna': 266.99,  'base_pS': 80},               # ~80 pS
    'Cav1.2':   {'mrna': 99.46,   'base_pS': 25},               # ~25 pS
    'CaCC':     {'mrna': 329.91,  'base_pS': 8},                # ~8 pS
    'Nav1.2':   {'mrna': 3.02,    'base_pS': 15}                # ~15 pS
}

# Compute effective conductances (in mS/cm²) for each channel:
conductances = {}
for channel in ['Kir6.1', 'Kir2.2', 'TRPC3', 'Cav1.2', 'CaCC', 'Nav1.2']:
    data = rna_data[channel]
    conductances[channel] = compute_effective_conductance(
        mrna = data['mrna'],
        base_pS = data['base_pS'],
        efficiency = translation_efficiency,
        stoichiometry = stoich[channel],
        scaling_factor = scaling_factor
    )

# For debugging, print the computed effective conductances:
print("Computed effective conductances (mS/cm²):")
for ch, g in conductances.items():
    print(f"  {ch}: {g:.5f}")

# =============================================================================
# MEMBRANE CURRENT FUNCTIONS (USING THE COMPUTED CONDUCTANCES)
# =============================================================================
def calculate_membrane_currents(V, Ca_i, ATP, IP3, params):
    """
    Calculate the membrane currents from the selected channels:
      - Kir6.1 (ATP-sensitive inward rectifier)
      - Kir2.2 (strong inward rectifier)
      - TRPC3 (receptor-operated non-selective cation channel)
      - Cav1.2 (L-type voltage-gated Ca²⁺ channel)
      - CaCC (Ca²⁺-activated Cl⁻ channel)
      - Nav1.2 (voltage-gated Na⁺ channel)
    """
    # --- Kir6.1 ---
    atp_factor = 1 / (1 + (ATP / params['ATP_half_Kir6_1'])**params['ATP_hill_Kir6_1'])
    I_Kir6_1 = params['conductance_Kir6_1'] * atp_factor * (V - params['reversal_potential_K'])
    
    # --- Kir2.2 ---
    Kir2_activation = 1 / (1 + np.exp((V - params['V_half_Kir2_2']) / params['slope_Kir2_2']))
    I_Kir2_2 = params['conductance_Kir2_2'] * Kir2_activation * (V - params['reversal_potential_K'])
    
    # --- TRPC3 ---
    TRPC3_activation = IP3 / (IP3 + params['IP3_half_TRPC3'])
    I_TRPC3 = params['conductance_TRPC3'] * TRPC3_activation * (V - params['reversal_potential_TRPC3'])
    
    # --- Cav1.2 ---
    d_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_Cav1_2']) / params['activation_slope_Cav1_2']))
    f_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_Cav1_2']) / params['inactivation_slope_Cav1_2']))
    I_Cav1_2 = params['conductance_Cav1_2'] * d_inf * f_inf * (V - params['reversal_potential_Ca'])
    
    # --- CaCC ---
    CaCC_activation = Ca_i / (Ca_i + params['calcium_activation_threshold_CaCC'])
    I_CaCC = params['conductance_CaCC'] * CaCC_activation * (V - params['reversal_potential_Cl'])
    
    # --- Nav1.2 ---
    m_inf = 1 / (1 + np.exp(-(V - params['activation_midpoint_Nav1_2']) / params['activation_slope_Nav1_2']))
    h_inf = 1 / (1 + np.exp((V - params['inactivation_midpoint_Nav1_2']) / params['inactivation_slope_Nav1_2']))
    I_Nav1_2 = params['conductance_Nav1_2'] * (m_inf**3) * h_inf * (V - params['reversal_potential_Na'])
    
    return I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2

# =============================================================================
# IP3R1 FLUX AND INACTIVATION DYNAMICS (De Young–Keizer)
# =============================================================================
def calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params):
    """
    Calculate the Ca²⁺ flux through the IP₃ receptor using a
    simplified De Young–Keizer formulation.
    """
    m_inf = IP3 / (IP3 + params['d1'])
    n_inf = Ca_i / (Ca_i + params['d5'])
    p_open = (m_inf**3) * (n_inf**3) * (h_IP3**3)
    J_IP3R = params['v_IP3R'] * p_open * (Ca_ER - Ca_i)
    return J_IP3R

def dh_ip3_dt(Ca_i, h_IP3, params):
    """
    Calculate the time derivative of the IP₃ receptor inactivation variable h.
    """
    h_inf = params['d2'] / (params['d2'] + Ca_i)
    dh_dt = (h_inf - h_IP3) / params['tau_h']
    return dh_dt

# =============================================================================
# PMCA, SERCA, and Mitochondrial Fluxes
# =============================================================================
def calculate_PMCA(Ca_i, params):
    """Plasma membrane Ca²⁺ ATPase (PMCA) flux: Ca removal from the cytosol."""
    J_PMCA = params['v_PMCA'] * Ca_i / (Ca_i + params['K_PMCA'])
    return J_PMCA

def calculate_SERCA(Ca_i, params):
    """SERCA pump flux: Ca uptake into the ER."""
    J_SERCA = params['v_SERCA'] * (Ca_i**2) / (Ca_i**2 + params['K_SERCA']**2)
    return J_SERCA

def calculate_mito_fluxes(Ca_i, Ca_mito, params):
    """
    Mitochondrial Ca²⁺ handling: uptake from the cytosol and release back.
    """
    J_mito_uptake = params['v_mito'] * Ca_i / (Ca_i + params['K_mito'])
    J_mito_release = params['v_mito_rel'] * Ca_mito
    return J_mito_uptake, J_mito_release

# =============================================================================
# ODE SYSTEM: FULL PERICYTE MODEL
# =============================================================================
def model(t, y, params):
    """
    ODE system for the pericyte model.
    State variables (y):
      y[0] : V       - Membrane potential (mV)
      y[1] : Ca_i    - Cytosolic Ca²⁺ concentration (mM)
      y[2] : ATP     - Cytosolic ATP concentration (mM)
      y[3] : Ca_ER   - ER Ca²⁺ concentration (mM)
      y[4] : IP3     - IP₃ concentration (mM)
      y[5] : Ca_mito - Mitochondrial Ca²⁺ concentration (mM)
      y[6] : h_IP3   - IP₃R inactivation variable (dimensionless)
    """
    V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3 = y

    # --- Membrane Currents ---
    I_Kir6_1, I_Kir2_2, I_TRPC3, I_Cav1_2, I_CaCC, I_Nav1_2 = calculate_membrane_currents(V, Ca_i, ATP, IP3, params)
    I_total = I_Kir6_1 + I_Kir2_2 + I_TRPC3 + I_Cav1_2 + I_CaCC + I_Nav1_2
    dV_dt = -I_total / params['membrane_capacitance']
    
    # --- Intracellular Ca²⁺ Dynamics ---
    J_IP3R = calculate_ip3r_flux(Ca_i, Ca_ER, IP3, h_IP3, params)   # ER release via IP₃R
    J_PMCA = calculate_PMCA(Ca_i, params)                            # Extrusion via PMCA
    J_SERCA = calculate_SERCA(Ca_i, params)                          # Uptake into ER via SERCA
    J_ER_leak = params['v_ER_leak'] * (Ca_ER - Ca_i)                  # Passive leak from ER
    
    # Mitochondrial fluxes:
    J_mito_uptake, J_mito_release = calculate_mito_fluxes(Ca_i, Ca_mito, params)
    
    # Ca²⁺ entry via membrane channels (assume inward currents increase Ca_i):
    f_TRPC3_Ca = params.get('f_TRPC3_Ca', 0.1)
    J_Ca_in = -I_Cav1_2 - f_TRPC3_Ca * I_TRPC3
    
    dCa_i_dt = J_Ca_in + J_IP3R - J_SERCA - J_PMCA + J_ER_leak + J_mito_release - J_mito_uptake
    
    # --- ATP Dynamics (assumed constant for now) ---
    dATP_dt = 0
    
    # --- ER Ca²⁺ Dynamics ---
    dCa_ER_dt = J_SERCA - J_IP3R - J_ER_leak
    
    # --- IP₃ Dynamics (assumed constant) ---
    dIP3_dt = 0
    
    # --- Mitochondrial Ca²⁺ Dynamics ---
    dCa_mito_dt = J_mito_uptake - J_mito_release
    
    # --- IP₃R Inactivation Variable Dynamics ---
    dh_IP3_dt_val = dh_ip3_dt(Ca_i, h_IP3, params)
    
    return [dV_dt, dCa_i_dt, dATP_dt, dCa_ER_dt, dIP3_dt, dCa_mito_dt, dh_IP3_dt_val]

# =============================================================================
# MAIN SIMULATION AND PLOTTING
# =============================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Build the parameter dictionary – using our computed effective conductances
    # -----------------------------------------------------------------------------
    params = {
        # Membrane capacitance (µF/cm²)
        'membrane_capacitance': 1.0,
        
        # --- Channel Parameters (using computed effective conductances) ---
        'conductance_Kir6_1': conductances['Kir6.1'],
        'conductance_Kir2_2': conductances['Kir2.2'],
        'conductance_TRPC3':  conductances['TRPC3'],
        'conductance_Cav1_2': conductances['Cav1.2'],
        'conductance_CaCC':   conductances['CaCC'],
        'conductance_Nav1_2': conductances['Nav1.2'],
        
        # Reversal potentials (mV)
        'reversal_potential_K': -90,
        'reversal_potential_TRPC3': -10,  # non-selective; near 0 mV
        'reversal_potential_Ca': 120,
        'reversal_potential_Cl': -40,
        'reversal_potential_Na': 55,
        
        # --- Kir6.1 gating ---
        'ATP_half_Kir6_1': 0.5,    # mM
        'ATP_hill_Kir6_1': 2,
        
        # --- Kir2.2 gating ---
        'V_half_Kir2_2': -60,      # mV
        'slope_Kir2_2': 10,        # mV
        
        # --- TRPC3 gating ---
        'IP3_half_TRPC3': 0.2,     # mM
        
        # --- Cav1.2 gating (L-type Ca²⁺ channel) ---
        'activation_midpoint_Cav1_2': -20,  # mV
        'activation_slope_Cav1_2': 6,       # mV
        'inactivation_midpoint_Cav1_2': -40,  # mV
        'inactivation_slope_Cav1_2': 6,       # mV
        
        # --- CaCC gating (Ca²⁺-activated Cl⁻ channel) ---
        'calcium_activation_threshold_CaCC': 0.0005,  # mM
        
        # --- Nav1.2 gating (voltage-gated Na⁺ channel) ---
        'activation_midpoint_Nav1_2': -35,  # mV
        'activation_slope_Nav1_2': 7,       # mV
        'inactivation_midpoint_Nav1_2': -60,  # mV
        'inactivation_slope_Nav1_2': 5,       # mV,
        
        # --- De Young–Keizer IP₃R1 parameters ---
        'd1': 0.13,     # mM
        'd2': 1.049,    # mM
        'd5': 0.082,    # mM
        'v_IP3R': 0.5,  # Maximum IP₃R flux (arbitrary units)
        'tau_h': 2.0,   # Time constant for h dynamics
        
        # --- PMCA (Plasma Membrane Ca²⁺ ATPase) ---
        'v_PMCA': 0.2,      # Maximal flux
        'K_PMCA': 0.0003,   # mM
        
        # --- SERCA (ER Ca²⁺ ATPase) ---
        'v_SERCA': 0.3,     # Maximal flux
        'K_SERCA': 0.0003,  # mM
        
        # --- ER Leak ---
        'v_ER_leak': 0.1,   # Leak rate
        
        # --- Mitochondrial Parameters ---
        'v_mito': 0.05,     # Uptake rate
        'K_mito': 0.0003,   # mM
        'v_mito_rel': 0.02, # Release rate
        
        # --- Fraction of TRPC3 current that carries Ca²⁺ ---
        'f_TRPC3_Ca': 0.1,
    }
    
    # -----------------------------------------------------------------------------
    # Initial Conditions: [V, Ca_i, ATP, Ca_ER, IP3, Ca_mito, h_IP3]
    # -----------------------------------------------------------------------------
    y0 = [
        -30,      # Membrane potential (mV)
        0.0001,   # Cytosolic [Ca²⁺] (mM)
        1.0,      # ATP (mM)
        0.1,      # ER [Ca²⁺] (mM)
        0.1,      # IP₃ (mM)
        0.0001,   # Mitochondrial [Ca²⁺] (mM)
        0.8       # IP₃R inactivation variable h
    ]
    
    # -----------------------------------------------------------------------------
    # Time Span (milliseconds) and ODE Integration
    # -----------------------------------------------------------------------------
    t_span = (0, 1000)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    sol = solve_ivp(lambda t, y: model(t, y, params),
                    t_span, y0, t_eval=t_eval, method='LSODA')
    
    # Extract solution arrays:
    time         = sol.t
    V_array      = sol.y[0]
    Ca_i_array   = sol.y[1]
    ATP_array    = sol.y[2]
    Ca_ER_array  = sol.y[3]
    IP3_array    = sol.y[4]
    Ca_mito_array= sol.y[5]
    h_IP3_array  = sol.y[6]
    
    # -----------------------------------------------------------------------------
    # Post-Processing: Compute Membrane Currents and Intracellular Fluxes
    # -----------------------------------------------------------------------------
    I_Kir6_1_list, I_Kir2_2_list = [], []
    I_TRPC3_list, I_Cav1_2_list = [], []
    I_CaCC_list, I_Nav1_2_list = [], []
    
    J_IP3R_list    = []
    J_PMCA_list    = []
    J_SERCA_list   = []
    J_ER_leak_list = []
    J_mito_uptake_list, J_mito_release_list = [], []
    
    for i in range(len(time)):
        # Compute membrane currents:
        currents = calculate_membrane_currents(V_array[i], Ca_i_array[i], ATP_array[i], IP3_array[i], params)
        I_Kir6_1_list.append(currents[0])
        I_Kir2_2_list.append(currents[1])
        I_TRPC3_list.append(currents[2])
        I_Cav1_2_list.append(currents[3])
        I_CaCC_list.append(currents[4])
        I_Nav1_2_list.append(currents[5])
        
        # Compute intracellular fluxes:
        J_IP3R_list.append(calculate_ip3r_flux(Ca_i_array[i], Ca_ER_array[i], IP3_array[i], h_IP3_array[i], params))
        J_PMCA_list.append(calculate_PMCA(Ca_i_array[i], params))
        J_SERCA_list.append(calculate_SERCA(Ca_i_array[i], params))
        J_ER_leak_list.append(params['v_ER_leak'] * (Ca_ER_array[i] - Ca_i_array[i]))
        mito_up, mito_rel = calculate_mito_fluxes(Ca_i_array[i], Ca_mito_array[i], params)
        J_mito_uptake_list.append(mito_up)
        J_mito_release_list.append(mito_rel)
    
    # Convert lists to arrays:
    I_Kir6_1_array = np.array(I_Kir6_1_list)
    I_Kir2_2_array = np.array(I_Kir2_2_list)
    I_TRPC3_array  = np.array(I_TRPC3_list)
    I_Cav1_2_array = np.array(I_Cav1_2_list)
    I_CaCC_array   = np.array(I_CaCC_list)
    I_Nav1_2_array = np.array(I_Nav1_2_list)
    
    J_IP3R_array         = np.array(J_IP3R_list)
    J_PMCA_array         = np.array(J_PMCA_list)
    J_SERCA_array        = np.array(J_SERCA_list)
    J_ER_leak_array      = np.array(J_ER_leak_list)
    J_mito_uptake_array  = np.array(J_mito_uptake_list)
    J_mito_release_array = np.array(J_mito_release_list)
    
    # -----------------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------------
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
