import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##############################################################################
#                            Single-Cell Subroutines
##############################################################################
# Physical constants
faraday_constant = 96485.0  # C/mol
gas_constant = 8314.0       # J/(mol*K)
temperature = 310.0         # K

# Global volume settings (example)
cell_volume = 2e-12         # L
Vcyto = cell_volume
ER_volume = cell_volume * 0.2

# Buffers
Bscyt = 225.0
aKscyt = 0.1
Bser = 2000.0
aKser = 1.0
Bm = 111.0
aKm = 0.123

# IP3R (De Young-Keizer style) parameters
a1 = 400.0
a2 = 0.2
a3 = 400.0
a4 = 0.2
a5 = 20.0
d1 = 0.13
d2 = 1.049
d3 = 0.9434
d4 = 0.1445
d5 = 82.34e-3
v1 = 90.0
b1 = a1*d1
b2 = a2*d2
b3 = a3*d3
b4 = a4*d4
b5 = a5*d5

# PMCA parameters
vu = 1540000.0
vm = 2200000.0
aku = 0.303
akmp = 0.14
aru = 1.8
arm = 2.1
p1 = 0.1
p2 = 0.01

# Mito
psi_mV = 160.0
psi_volts = psi_mV/1000.0
z = 2.0  # valence Ca2+

##############################################################################
#                          Patch Clamp Injection Helper
##############################################################################
def patch_clamp_current(t, cell_idx, params):
    """
    Returns the injection current (pA) for cell cell_idx at time t.
    A simple step from t_start to t_end in just one cell.
    """
    t_start = params['patch_tstart']
    t_end   = params['patch_tend']
    amp     = params['patch_amplitude']  # in pA

    # Only inject into the cell == params['patch_cell'] during [t_start, t_end]
    if (cell_idx == params['patch_cell']) and (t >= t_start) and (t <= t_end):
        return amp
    else:
        return 0.0

##############################################################################
#                          Helper Functions
##############################################################################
def safe_log(x):
    """Avoid log(0) by clamping x to a minimum of 1e-12."""
    return np.log(max(x, 1e-12))

def calculate_buffering_factors(Ca_i, Ca_ER):
    """ Return cytosolic and ER buffering factors for given Ca levels. """
    Ca_i  = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    beta_cyt = 1.0/(1.0 + (Bscyt*aKscyt)/((aKscyt+Ca_i)**2) + (Bm*aKm)/((aKm+Ca_i)**2))
    beta_er  = 1.0/(1.0 + (Bser*aKser)/((aKser+Ca_ER)**2) + (Bm*aKm)/((aKm+Ca_ER)**2))
    return beta_cyt, beta_er

def calculate_ip3r_states(Ca_i, IP3,
                          x000, x100, x010, x001, x110, x101, x011, x111):
    """ Compute derivatives of the 8 IP3R states for one IP3 receptor species. """
    Ca_i = max(Ca_i, 1e-12)
    IP3  = max(IP3, 1e-12)

    # Reactions
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
    dx000 = f1 + f2 + f3
    dx100 = f4 + f5 - f2
    dx010 = -f1 + f6 + f7
    dx001 = f8 - f3 + f9
    dx110 = -f4 - f6 + f10
    dx101 = f11 - f9 - f5
    dx011 = -f8 - f7 + f12
    dx111 = -f11 - f12 - f10
    return dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    """ Single-channel IP3R flux term for a single IP3R 'active' state. """
    Ca_ER = max(Ca_ER, 1e-12)
    Ca_i  = max(Ca_i,  1e-12)
    return v1*(x110**3)*(Ca_ER - Ca_i)

def calculate_pmca_flux(Ca_i, dpmca):
    """ PMCA flux. """
    Ca_i = max(Ca_i,1e-12)
    u4 = (vu*(Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5 = (vm*(Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1 - dpmca)*u5)/(6.6253e5)
    return cJpmca, u4, u5

def calculate_mito_fluxes(Ca_i, Ca_m, Pmito, Vmito, alphm, alphi, Vnc, aNa, akna, akca):
    """ Uniporter (J_uni) and Na+/Ca2+ exchanger (J_nc) fluxes into mitochondria. """
    Ca_i = max(Ca_i,1e-12)
    Ca_m = max(Ca_m,1e-12)
    bb = (z*psi_volts*faraday_constant)/(gas_constant*temperature)

    # prevent overflow in exp(-bb)
    if bb < 50:
        exp_neg_bb = np.exp(-bb)
    else:
        exp_neg_bb = 1e-22  # artificially small

    # Mito uniporter
    # J_uni = Pmito * bb * (...) / Vmito
    # (We handle Ca_i, Ca_m with exponent factor)
    J_uni = (Pmito / Vmito)*bb*((alphm*Ca_i*np.exp(-bb) - alphi*Ca_m)/(exp_neg_bb - 1))

    # Mito Na+/Ca2+ exchanger
    som = (aNa**3)*Ca_m/(akna**3 * akca)
    soe = (aNa**3)*Ca_i/(akna**3 * akca)
    B   = np.exp(0.5*psi_volts*z*faraday_constant/(gas_constant*temperature))
    denom = (1+(aNa**3/(akna**3))+Ca_m/akca + som +
             (aNa**3/(akna**3))+Ca_i/akca + soe)
    J_nc = Vnc*(B*som - (1/B)*soe)/denom
    return J_uni, J_nc

def ip3_ode(Ca_i, IP3, prodip3, V2ip3, ak2ip3, V3ip3, ak3ip3, ak4ip3):
    """ Simple IP3 production/consumption ODE. """
    Ca_i = max(Ca_i,1e-12)
    IP3  = max(IP3,1e-12)
    term1 = prodip3
    term2 = V2ip3/(1 + (ak2ip3/IP3))
    term3 = (V3ip3/(1 + (ak3ip3/IP3))) * (1/(1 + (ak4ip3/(Ca_i))))
    dIP3dt = term1 - term2 - term3

    # Soft clamp IP3 from going above or below certain range
    new_IP3 = IP3 + dIP3dt
    if new_IP3 > 0.01:
        dIP3dt = 0.01 - IP3
    if new_IP3 < 1e-12:
        dIP3dt = 1e-12 - IP3
    return dIP3dt

def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER,
                       x110, x1102, IP3, params, i):
    """
    For cell i, compute:
      - Ion currents (Kir, TRPC, CaCC, CaL, leak, PMCA, NCX)
      - SERCA, leak fluxes, IP3R flux, RyR flux
    Returns a tuple of:
      I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA,
      J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR
    """

    # Grab array-based parameters for cell i
    gKir = params['conductance_Kir61_array'][i]
    gTRPC = params['conductance_TRPC1_array'][i]
    gCaCC = params['conductance_CaCC_array'][i]
    gCaL  = params['conductance_CaL_array'][i]
    gLeak = params['conductance_leak_array'][i]
    gIP3R1 = params['conductance_IP3R1_array'][i]
    gIP3R2 = params['conductance_IP3R2_array'][i]
    gRyR = params['conductance_RyR_array'][i]
    kSERCA = params['k_serca_array'][i]
    KmSERCA = params['Km_serca_array'][i]
    leakER = params['leak_rate_er_array'][i]
    kNCX = params['k_ncx_array'][i]
    Ca_out = params['Ca_out_array'][i]
    Na_in  = params['Na_in_array'][i]
    Ca_act_CaCC = params['calcium_activation_threshold_CaCC_array'][i]
    K_out = params['K_out_array'][i]
    K_in  = params['K_in_array'][i]
    Cl_out = params['Cl_out_array'][i]
    Cl_in  = params['Cl_in_array'][i]

    # Clamp Ca_i to avoid log(0)
    Ca_i_clamped = max(Ca_i, 1e-12)
    Ca_out_clamped = max(Ca_out, 1e-12)

    # Reversal potentials (Nernst) - assume everything in mV
    E_K  = (gas_constant*temperature/(1*faraday_constant))*safe_log(K_out/K_in)
    E_Ca = (gas_constant*temperature/(2*faraday_constant))*safe_log(Ca_out_clamped / Ca_i_clamped)
    E_Cl = (gas_constant*temperature/(-1*faraday_constant))*safe_log(Cl_out/Cl_in)

    # Convert from Joule/C to mV
    # (gas_constant*temperature / faraday_constant) ~ 26.7 mV at 310 K
    # So it's okay to interpret directly as mV
    # Alternatively, you might do *1000 if needed, but typically these are ~ 26-27 mV logs.
    # Ca_L gating
    vmid_act    = params['activation_midpoint_CaL']
    vslope_act  = params['activation_slope_CaL']
    vmid_inact  = params['inactivation_midpoint_CaL']
    vslope_inact= params['inactivation_slope_CaL']
    amp_factor  = params['amplitude_factor_CaL']
    voltage_shift = params['voltage_shift_CaL']
    slope_factor  = params['slope_factor_CaL']
    d_inf = 1/(1 + np.exp(-(V - vmid_act)/vslope_act))
    f_inf = 1/(1 + np.exp((V - vmid_inact)/vslope_inact)) \
            + amp_factor/(1 + np.exp((voltage_shift - V)/slope_factor))

    # Kir (example)
    I_Kir61 = gKir * atp * np.sqrt(K_out/5.4) * (V - E_K) \
              / (1 + np.exp((V - E_K - 15)/6))
    I_TRPC1 = gTRPC*(V - E_Ca)
    I_CaCC  = gCaCC*(Ca_i_clamped/(Ca_i_clamped + Ca_act_CaCC))*(V - E_Cl)
    I_CaL   = gCaL*d_inf*f_inf*(V - E_Ca)
    I_leak  = gLeak*(V + 40) #gLeak*(V - E_K)  # or (V - E_Leak)
    
     

    # PMCA
    cJpmca, _, _ = calculate_pmca_flux(Ca_i_clamped, dpmca)

    # Convert flux to current
    I_PMCA = cJpmca*z*faraday_constant*Vcyto*1e6

    # SERCA / leak
    J_SERCA   = kSERCA*(Ca_i_clamped**2)/(Ca_i_clamped**2 + KmSERCA**2)
    J_ER_leak = leakER*(Ca_ER - Ca_i_clamped)

    # NCX (very simplified example)
    I_NCX = kNCX*(Na_in**3/(Na_in**3+87.5**3))*(Ca_out/(Ca_out+1))

    # IP3R flux: from both IP3R1 + IP3R2 states x110
    J_IP3R1 = calculate_ip3r_flux(x110, Ca_ER, Ca_i_clamped)*gIP3R1
    J_IP3R2 = calculate_ip3r_flux(x1102, Ca_ER, Ca_i_clamped)*gIP3R2
    J_IP3R  = J_IP3R1 + J_IP3R2

    # RyR
    J_RyR = gRyR*(Ca_i_clamped/(Ca_i_clamped+0.3))*(Ca_ER - Ca_i_clamped)
    return (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak,
            d_inf, f_inf, I_PMCA,
            J_SERCA, J_ER_leak, I_NCX,
            J_IP3R, J_RyR)

##############################################################################
#                          Multi-Cell ODE System
##############################################################################
def multicell_model(t, y, params):
    """
    ODE for N cells, each with 23 states, plus optional gap-junction coupling on V,
    plus a patch-clamp injection current.
    """
    N = params['N']
    dYdt = np.zeros_like(y)
    global p1, p2  # PMCA gating, if needed
    for i in range(N):
        offset = 23 * i
        V       = y[offset +  0]
        Ca_i    = y[offset +  1]
        atp     = y[offset +  2]
        dpmca   = y[offset +  3]
        Ca_ER   = y[offset +  4]
        x000    = y[offset +  5]
        x100    = y[offset +  6]
        x010    = y[offset +  7]
        x001    = y[offset +  8]
        x110    = y[offset +  9]
        x101    = y[offset + 10]
        x011    = y[offset + 11]
        x111    = y[offset + 12]
        IP3     = y[offset + 13]
        Ca_m    = y[offset + 14]
        x0002   = y[offset + 15]
        x1002   = y[offset + 16]
        x0102   = y[offset + 17]
        x0012   = y[offset + 18]
        x1102   = y[offset + 19]
        x1012   = y[offset + 20]
        x0112   = y[offset + 21]
        x1112   = y[offset + 22]

        # Buffering
        beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)

        # IP3R states (IP3R1)
        dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
            calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001,
                                  x110, x101, x011, x111)

        # IP3R states (IP3R2)
        dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, dx1112 = \
            calculate_ip3r_states(Ca_i, IP3, x0002, x1002, x0102, x0012,
                                  x1102, x1012, x0112, x1112)

        # Currents and fluxes
        (I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak,
         d_inf, f_inf, I_PMCA,
         J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR) = \
            calculate_currents(V, Ca_i, atp, dpmca, Ca_ER,
                               x110, x1102, IP3, params, i)

        # Membrane eqn
        cm = params['membrane_capacitance_array'][i]
        I_net_single = (I_Kir61 + I_TRPC1 + I_CaCC + I_CaL + I_leak + I_PMCA + I_NCX)

        # Gap-junction coupling on voltage
        g_gap = params['g_gap']
        if i > 0:
            V_left  = y[offset - 23]
        else:
            V_left  = V
        if i < N - 1:
            V_right = y[offset + 23]
        else:
            V_right = V
        I_gap = g_gap * ((V_left - V) + (V_right - V))

        # Patch-clamp injection current (pA)
        I_inject = patch_clamp_current(t, i, params)
        dV_dt = -(I_net_single + I_inject)/cm + I_gap/cm

        # Ca dynamics
        # from current (I_CaL is inward, so negative means Ca entering cell)
        Ca_influx = -I_CaL/(2.0*faraday_constant*cell_volume)
        Ca_efflux = -(I_PMCA + I_NCX)/(2.0*faraday_constant*cell_volume)
        dCa_dt = beta_cyt*(Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)

        # ER Ca
        dCa_ER_dt = beta_er*(Vcyto/ER_volume)*(J_SERCA - (J_ER_leak + J_IP3R + J_RyR))

        # PMCA gating
        w1 = p1*Ca_i
        w2 = p2
        taom = 1/(w1 + w2)
        dpmcainf = w2/(w1 + w2)
        ddpmca_dt = (dpmcainf - dpmca)/taom

        # IP3
        dIP3_dt = ip3_ode(Ca_i, IP3,
                          params['prodip3_array'][i],
                          params['V2ip3_array'][i],
                          params['ak2ip3_array'][i],
                          params['V3ip3_array'][i],
                          params['ak3ip3_array'][i],
                          params['ak4ip3_array'][i])

        # Mito fluxes
        (J_uni, J_nc) = calculate_mito_fluxes(Ca_i, Ca_m,
                                              params['Pmito_array'][i],
                                              params['Vmito_array'][i],
                                              params['alphm_array'][i],
                                              params['alphi_array'][i],
                                              params['Vnc_array'][i],
                                              params['aNa_array'][i],
                                              params['akna_array'][i],
                                              params['akca_array'][i])
        dCa_m_dt = J_uni - J_nc

        # Store derivatives
        dYdt[offset +  0] = dV_dt
        dYdt[offset +  1] = dCa_dt
        dYdt[offset +  2] = 0.0  # atp not changing here
        dYdt[offset +  3] = ddpmca_dt
        dYdt[offset +  4] = dCa_ER_dt
        dYdt[offset +  5] = dx000
        dYdt[offset +  6] = dx100
        dYdt[offset +  7] = dx010
        dYdt[offset +  8] = dx001
        dYdt[offset +  9] = dx110
        dYdt[offset + 10] = dx101
        dYdt[offset + 11] = dx011
        dYdt[offset + 12] = dx111
        dYdt[offset + 13] = dIP3_dt
        dYdt[offset + 14] = dCa_m_dt
        dYdt[offset + 15] = dx0002
        dYdt[offset + 16] = dx1002
        dYdt[offset + 17] = dx0102
        dYdt[offset + 18] = dx0012
        dYdt[offset + 19] = dx1102
        dYdt[offset + 20] = dx1012
        dYdt[offset + 21] = dx0112
        dYdt[offset + 22] = dx1112
    return dYdt

##############################################################################
#                      Run + Plot Multicell Simulation
##############################################################################
def run_multicell_simulation(params):
    """
    Creates an initial condition array, runs solve_ivp on the multicell_model,
    and returns the solution object.
    """
    N = params['N']

    # 23 states per cell
    y0 = np.zeros(23*N)

    # Example initial conditions for each cell
    for i in range(N):
        offset = 23*i
        y0[offset + 0] = -70.0      # V (mV)
        y0[offset + 1] = 0.0001     # Ca_i (mM)
        y0[offset + 2] = 4.4        # ATP
        y0[offset + 3] = 1.0        # dpmca
        y0[offset + 4] = 0.5        # Ca_ER (mM)

        # IP3R1 states
        x000 = 0.27
        x100 = 0.039
        x010 = 0.29
        x001 = 0.17
        x110 = 0.042
        x101 = 0.0033
        x011 = 0.18
        x111 = 0.0035
        total = (x000 + x100 + x010 + x001 + x110 + x101 + x011 + x111)
        x000 /= total
        x100 /= total
        x010 /= total
        x001 /= total
        x110 /= total
        x101 /= total
        x011 /= total
        x111 /= total
        y0[offset + 5]  = x000
        y0[offset + 6]  = x100
        y0[offset + 7]  = x010
        y0[offset + 8]  = x001
        y0[offset + 9]  = x110
        y0[offset + 10] = x101
        y0[offset + 11] = x011
        y0[offset + 12] = x111

        # IP3
        y0[offset + 13] = 0.1

        # Mito Ca
        y0[offset + 14] = 0.0001

        # IP3R2 states (same approach)
        x0002 = 0.27
        x1002 = 0.039
        x0102 = 0.29
        x0012 = 0.17
        x1102 = 0.042
        x1012 = 0.0033
        x0112 = 0.18
        x1112 = 0.0035
        total2 = (x0002 + x1002 + x0102 + x0012 + x1102 + x1012 + x0112 + x1112)
        x0002 /= total2
        x1002 /= total2
        x0102 /= total2
        x0012 /= total2
        x1102 /= total2
        x1012 /= total2
        x0112 /= total2
        x1112 /= total2
        y0[offset + 15] = x0002
        y0[offset + 16] = x1002
        y0[offset + 17] = x0102
        y0[offset + 18] = x0012
        y0[offset + 19] = x1102
        y0[offset + 20] = x1012
        y0[offset + 21] = x0112
        y0[offset + 22] = x1112
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    sol = solve_ivp(
        fun=lambda t, Y: multicell_model(t, Y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='BDF',           # or 'Radau'
        rtol=1e-8,
        atol=1e-11
    )
    return sol

def plot_multicell_results(sol, params):
    """
    Example plotting function:
      - Plots membrane potential for each cell
      - Plots cytosolic Ca for each cell
    """
    t = sol.t
    N = params['N']
    V = []
    Ca_i = []
    for i in range(N):
        offset = 23*i
        V.append(sol.y[offset + 0, :])    # shape: (time points,)
        Ca_i.append(sol.y[offset + 1, :]) # shape: (time points,)

    # Plot Membrane Potential
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot(t, V[i], label=f"Cell {i}")
    plt.title("Membrane Potential (multi-cell) with Patch-Clamp Injection")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Plot Cytosolic Ca
    #plt.figure(figsize=(10, 5))
    #for i in range(N):
    #    plt.plot(t, Ca_i[i], label=f"Cell {i}")
    #plt.title("Cytosolic Ca (multi-cell)")
    #plt.xlabel("Time (ms)")
    #plt.ylabel("Ca_i (mM)")
    #plt.legend(loc='best')
    #plt.grid(True)
    #plt.show()
##############################################################################
#                        Example Usage / Main
##############################################################################
if __name__ == "__main__":

    # We might have a global p1, p2 (PMCA gating), define them here:
    p1 = 0.1
    p2 = 0.01

    # Number of cells
    N = 5

    # Make array-based parameters for each cell (here identical).
    conductance_Kir61_array = np.full(N, 0.025)
    conductance_TRPC1_array = np.full(N, 0.001)
    conductance_CaCC_array  = np.full(N, 0.001)
    conductance_CaL_array   = np.full(N, 0.0005)
    conductance_leak_array  = np.full(N, 10.0)
    conductance_IP3R1_array = np.full(N, 0.1)
    conductance_IP3R2_array = np.full(N, 0.05)
    conductance_RyR_array   = np.full(N, 0.01)
    k_serca_array      = np.full(N, 0.1)
    Km_serca_array     = np.full(N, 0.5)
    leak_rate_er_array = np.full(N, 0.05)
    k_ncx_array        = np.full(N, 0.001)

    # External / internal [Ca], [Na], [K], [Cl], etc.
    Ca_out_array   = np.full(N, 2.0)
    Na_in_array    = np.full(N, 15.38)
    calcium_activation_threshold_CaCC_array = np.full(N, 0.0005)
    K_out_array    = np.full(N, 6.26)
    K_in_array     = np.full(N, 140.0)
    Cl_out_array   = np.full(N, 110.0)
    Cl_in_array    = np.full(N, 9.65)

    # Mito parameters
    Vmito_array = np.full(N, Vcyto*0.08)
    Pmito_array = np.full(N, 2.776e-20)
    alphm_array = np.full(N, 0.2)
    alphi_array = np.full(N, 1.0)
    Vnc_array   = np.full(N, 1.836)
    aNa_array   = np.full(N, 5000.0)
    akna_array  = np.full(N, 8000.0)
    akca_array  = np.full(N, 8.0)

    # IP3 production / consumption
    prodip3_array = np.full(N, 0.01)
    V2ip3_array   = np.full(N, 12.5)
    ak2ip3_array  = np.full(N, 6.0)
    V3ip3_array   = np.full(N, 0.9)
    ak3ip3_array  = np.full(N, 0.1)
    ak4ip3_array  = np.full(N, 1.0)

    # Membrane Capacitance
    membrane_capacitance_array = np.full(N,0.94)  # pF source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5783500/

    # Gap junction conductance for V
    g_gap = 0.49  # pA/mV ? == pS

    # Patch clamp injection setup
    # We'll inject +0.1 pA from t=20 ms to t=60 ms in cell 0 (smaller amplitude)
    patch_amplitude = -5      # pA  (reduced from 5)
    patch_cell      = 0
    patch_tstart    = 20.0     # ms
    patch_tend      = 60.0     # ms

    # Package into a dictionary
    params = {
        'N': 3,
        'g_gap': g_gap,
        'conductance_Kir61_array': conductance_Kir61_array,
        'conductance_TRPC1_array': conductance_TRPC1_array,
        'conductance_CaCC_array': conductance_CaCC_array,
        'conductance_CaL_array': conductance_CaL_array,
        'conductance_leak_array': conductance_leak_array,
        'conductance_IP3R1_array': conductance_IP3R1_array,
        'conductance_IP3R2_array': conductance_IP3R2_array,
        'conductance_RyR_array': conductance_RyR_array,
        'k_serca_array': k_serca_array,
        'Km_serca_array': Km_serca_array,
        'leak_rate_er_array': leak_rate_er_array,
        'k_ncx_array': k_ncx_array,
        'Ca_out_array': Ca_out_array,
        'Na_in_array': Na_in_array,
        'calcium_activation_threshold_CaCC_array': calcium_activation_threshold_CaCC_array,
        'K_out_array': K_out_array,
        'K_in_array': K_in_array,
        'Cl_out_array': Cl_out_array,
        'Cl_in_array': Cl_in_array,
        'Vmito_array': Vmito_array,
        'Pmito_array': Pmito_array,
        'alphm_array': alphm_array,
        'alphi_array': alphi_array,
        'Vnc_array': Vnc_array,
        'aNa_array': aNa_array,
        'akna_array': akna_array,
        'akca_array': akca_array,
        'prodip3_array': prodip3_array,
        'V2ip3_array': V2ip3_array,
        'ak2ip3_array': ak2ip3_array,
        'V3ip3_array': V3ip3_array,
        'ak3ip3_array': ak3ip3_array,
        'ak4ip3_array': ak4ip3_array,
        'membrane_capacitance_array': membrane_capacitance_array,

        # CaL gating
        'activation_midpoint_CaL': -40,
        'activation_slope_CaL': 4,
        'inactivation_midpoint_CaL': -45,
        'inactivation_slope_CaL': 5,
        'amplitude_factor_CaL': 0.6,
        'voltage_shift_CaL': 50,
        'slope_factor_CaL': 20,

        # Simulation times
        'simulation_duration': 100.0,  # ms
        'time_points': 1000,          # for t_eval

        # Patch clamp injection parameters
        'patch_amplitude': patch_amplitude,
        'patch_cell': patch_cell,
        'patch_tstart': patch_tstart,
        'patch_tend': patch_tend,
    }

    # Run the multicell simulation
    sol = run_multicell_simulation(params)

    # Plot the results
    plot_multicell_results(sol, params)