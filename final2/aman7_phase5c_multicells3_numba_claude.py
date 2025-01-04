import time
import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

###############################################################################
#                            CONSTANTS
###############################################################################
faraday_constant = 96485   # C/mol
gas_constant = 8314        # J/(mol*K)
temperature = 310          # K

membrane_capacitance = 0.94  # pF
cell_volume = 2e-12          # L
Vcyto = cell_volume
ER_volume = cell_volume*0.2

valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2  # Ca2+

K_out = 6.26
K_in = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

a1, a2, a3, a4, a5 = 400.0, 0.2, 400.0, 0.2, 20.0
d1, d2, d3, d4, d5 = 0.13, 1.049, 0.9434, 0.1445, 82.34e-3
v1 = 90.0
b1 = a1*d1
b2 = a2*d2
b3 = a3*d3
b4 = a4*d4
b5 = a5*d5

vu, vm = 1540000.0, 2200000.0
aku, akmp = 0.303, 0.14
aru, arm = 1.8, 2.1
p1, p2 = 0.1, 0.01

Bscyt, aKscyt = 225.0, 0.1
Bser, aKser = 2000.0, 1.0
Bm, aKm = 111.0, 0.123

# Channel conductances
conductance_Kir61 = 0.025
conductance_TRPC1 = 0.001
conductance_CaL = 0.0005
conductance_CaCC = 0.001
conductance_leak = 0.01
conductance_IP3R1 = 0.1
conductance_IP3R2 = 0.05
conductance_RyR = 0.01

# Other parameters
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
calcium_activation_threshold_CaCC = 0.0005
k_serca = 0.1
Km_serca = 0.5
leak_rate_er = 0.05
k_ncx = 0.001

# IP3 parameters
prodip3 = 0.01
V2ip3 = 12.5
ak2ip3 = 6.0
V3ip3 = 0.9
ak3ip3 = 0.1
ak4ip3 = 1.0

# Mitochondrial parameters
Vmito = 2e-12 * 0.08
Pmito = 2.776e-20
psi_mV = 160.0
psi_volts = psi_mV/1000.0
alphm = 0.2
alphi = 1.0
Vnc = 1.836
aNa = 5000.0
akna = 8000.0
akca = 8.0

# Simulation parameters
simulation_duration = 100
time_points = 1000

# Initial conditions
initial_voltage = -70
initial_calcium = Ca_in
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.1
Ca_m_initial = 0.0001

# IP3R states initialization
x000_initial = 0.27
x100_initial = 0.039
x010_initial = 0.29
x001_initial = 0.17
x110_initial = 0.042
x101_initial = 0.0033
x011_initial = 0.18
x111_initial = 0.0035
total1 = (x000_initial + x100_initial + x010_initial + x001_initial +
          x110_initial + x101_initial + x011_initial + x111_initial)
x000_initial /= total1
x100_initial /= total1
x010_initial /= total1
x001_initial /= total1
x110_initial /= total1
x101_initial /= total1
x011_initial /= total1
x111_initial /= total1

x0002_initial = 0.27
x1002_initial = 0.039
x0102_initial = 0.29
x0012_initial = 0.17
x1102_initial = 0.042
x1012_initial = 0.0033
x0112_initial = 0.18
x1112_initial = 0.0035
total2 = (x0002_initial + x1002_initial + x0102_initial + x0012_initial +
          x1102_initial + x1012_initial + x0112_initial + x1112_initial)
x0002_initial /= total2
x1002_initial /= total2
x0102_initial /= total2
x0012_initial /= total2
x1102_initial /= total2
x1012_initial /= total2
x0112_initial /= total2
x1112_initial /= total2

@njit
def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    """Calculate reversal potentials for all ion types."""
    revK = (gas_constant*temperature/(valence_K*faraday_constant))*np.log(K_out/K_in)
    revCa = (gas_constant*temperature/(valence_Ca*faraday_constant))*np.log(Ca_out/Ca_i)
    revNa = (gas_constant*temperature/(valence_Na*faraday_constant))*np.log(Na_out/Na_in)
    revCl = (gas_constant*temperature/(valence_Cl*faraday_constant))*np.log(Cl_out/Cl_in)
    return revK, revCa, revNa, revCl

@njit
def calculate_buffering_factors(Ca_i, Ca_ER):
    """Calculate buffering factors for cytosol and ER."""
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    beta_cyt = 1.0/(1.0 + (Bscyt*aKscyt)/((aKscyt+Ca_i)**2) + (Bm*aKm)/((aKm+Ca_i)**2))
    beta_er = 1.0/(1.0 + (Bser*aKser)/((aKser+Ca_ER)**2) + (Bm*aKm)/((aKm+Ca_ER)**2))
    return beta_cyt, beta_er

@njit
def calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111):
    """Calculate IP3R state transitions."""
    Ca_i = max(Ca_i, 1e-12)
    IP3 = max(IP3, 1e-12)
    
    f1 = b5*x010 - a5*Ca_i*x000
    f2 = b1*x100 - a1*IP3*x000
    f3 = b4*x001 - a4*Ca_i*x000
    f4 = b5*x110 - a5*Ca_i*x100
    f5 = b2*x101 - a2*Ca_i*x100
    f6 = b1*x110 - a1*IP3*x010
    f7 = b4*x011 - a4*Ca_i*x010
    f8 = b5*x011 - a5*Ca_i*x001
    f9 = b3*x101 - a3*IP3*x001
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

@njit
def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    """Calculate IP3R flux."""
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    return v1*(x110**3)*(Ca_ER - Ca_i)

@njit
def calculate_pmca_flux(Ca_i, dpmca):
    """Calculate PMCA flux."""
    Ca_i = max(Ca_i, 1e-12)
    u4 = (vu*(Ca_i**aru)) / (Ca_i**aru + aku**aru)
    u5 = (vm*(Ca_i**arm)) / (Ca_i**arm + akmp**arm)
    cJpmca = (dpmca*u4 + (1.0 - dpmca)*u5) / 6.6253e5
    return cJpmca, u4, u5

@njit
def calculate_mito_fluxes(Ca_i, Ca_m):
    """Calculate mitochondrial calcium fluxes."""
    Ca_i = max(Ca_i, 1e-12)
    Ca_m = max(Ca_m, 1e-12)
    bb = (z*psi_volts*faraday_constant)/(gas_constant*temperature)
    exp_neg_bb = np.exp(-bb)
    if np.isinf(exp_neg_bb) or np.isnan(exp_neg_bb):
        exp_neg_bb = 0.0
    J_uni = (Pmito/Vmito)*bb*((alphm*Ca_i*np.exp(-bb) - alphi*Ca_m)/(exp_neg_bb - 1))
    
    som = (aNa**3)*Ca_m/(akna**3*akca)
    soe = (aNa**3)*Ca_i/(akna**3*akca)
    B = np.exp(0.5*psi_volts*z*faraday_constant/(gas_constant*temperature))
    denominator = (1 + (aNa**3/(akna**3)) + Ca_m/akca + som +
                  (aNa**3/(akna**3)) + Ca_i/akca + soe)
    J_nc = Vnc*(B*som - (1.0/B)*soe)/denominator
    return J_uni, J_nc

@njit
def ip3_ode(Ca_i, IP3):
    """Calculate IP3 dynamics."""
    Ca_i = max(Ca_i, 1e-12)
    IP3 = max(IP3, 1e-12)
    term1 = prodip3
    term2 = V2ip3 / (1.0 + (ak2ip3/IP3))
    term3 = (V3ip3 / (1.0 + (ak3ip3/IP3))) * (1.0/(1.0 + (ak4ip3/(Ca_i))))
    dIP3dt = term1 - term2 - term3
    new_IP3 = IP3 + dIP3dt
    if new_IP3 > 0.01:
        dIP3dt = (0.01 - IP3)
    if new_IP3 < 1e-12:
        dIP3dt = (1e-12 - IP3)
    return dIP3dt

@njit
def model_core(t, y):
    """Core model function for single cell calculations."""
    # Unpack state variables
    (V, Ca_i, atp, dpmca, Ca_ER,
     x000, x100, x010, x001, x110, x101, x011, x111,
     IP3, Ca_m,
     x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112) = y
    
    # Calculate reversal potentials
    revK, revCa, revNa, revCl = calculate_reversal_potentials(
        K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in)
    
    # Calculate buffering factors
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    
    # Calculate IP3R states
    dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111 = \
        calculate_ip3r_states(Ca_i, IP3, x000, x100, x010, x001, x110, x101, x011, x111)
    dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, dx1112 = \
        calculate_ip3r_states(Ca_i, IP3, x0002, x1002, x0102, x0012, x1102, x1012, x0112, x1112)
    
    # Calculate currents
    Ca_i = max(Ca_i, 1e-12)
    Ca_ER = max(Ca_ER, 1e-12)
    
    # CaL channel dynamics
    d_inf = 1.0/(1.0 + np.exp(-(V+40.0)/4.0))
    f_inf = (1.0/(1.0 + np.exp((V + 45.0)/5.0)) +
             0.6/(1.0 + np.exp((50.0 - V)/20.0)))
    
    I_Kir61 = (conductance_Kir61*atp*np.sqrt(K_out/5.4) *
               (V - revK)/(1.0 + np.exp((V - revK - 15.0)/6.0)))
    I_TRPC1 = conductance_TRPC1*(V - revCa)
    I_CaCC = (conductance_CaCC*(Ca_i/(Ca_i + calcium_activation_threshold_CaCC))*(V - revCl))
    I_CaL = conductance_CaL*d_inf*f_inf*(V - revCa)
    I_leak = conductance_leak*(V - revK)
    
    # Calculate PMCA flux
    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca*z*faraday_constant*Vcyto*1e6
    
    # Calculate SERCA and ER leak
    J_SERCA = k_serca*(Ca_i**2)/(Ca_i**2 + Km_serca**2)
    J_ER_leak = leak_rate_er*(Ca_ER - Ca_i)
    I_NCX = (k_ncx * (Na_in**3/(Na_in**3+87.5**3)) * (Ca_out/(Ca_out+1.0)))
    
    # Calculate IP3R and RyR fluxes
    J_IP3R1 = calculate_ip3r_flux(x110, Ca_ER, Ca_i)*conductance_IP3R1
    J_IP3R2 = calculate_ip3r_flux(x1102, Ca_ER, Ca_i)*conductance_IP3R2
    J_IP3R = J_IP3R1 + J_IP3R2
    J_RyR = conductance_RyR*(Ca_i/(Ca_i+0.3))*(Ca_ER - Ca_i)
    
    # Calculate voltage derivative
    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX)/membrane_capacitance
    
    # Calculate calcium fluxes
    Ca_influx = -I_CaL/(2.0*z*faraday_constant*cell_volume)
    Ca_efflux = (-I_PMCA - I_NCX)/(2.0*z*faraday_constant*cell_volume)
    
    # Calculate calcium derivatives
    dCa_dt = beta_cyt*(Ca_influx + Ca_efflux + J_ER_leak + J_IP3R + J_RyR - J_SERCA)
    dCa_ER_dt = beta_er*((Vcyto/ER_volume)*(J_SERCA - J_ER_leak - J_IP3R - J_RyR))
    
    # Calculate PMCA dynamics
    w1 = p1*Ca_i
    w2 = p2
    taom = 1.0/(w1 + w2)
    dpmcainf = w2/(w1 + w2)
    ddpmca_dt = (dpmcainf - dpmca)/taom
    
    # Calculate IP3 dynamics
    dIP3_dt = ip3_ode(Ca_i, IP3)
    
    # Calculate mitochondrial dynamics
    J_uni, J_nc = calculate_mito_fluxes(Ca_i, Ca_m)
    dCa_m_dt = J_uni - J_nc
    
    # Return array of derivatives
    return np.array([
        dV_dt, dCa_dt, 0.0, ddpmca_dt, dCa_ER_dt,
        dx000, dx100, dx010, dx001, dx110, dx101, dx011, dx111,
        dIP3_dt, dCa_m_dt,
        dx0002, dx1002, dx0102, dx0012, dx1102, dx1012, dx0112, dx1112
    ])

@njit
def multi_cell_model(t, Y, Ng, g_gap, dx, cm, stim_params):
    """Multi-cell model with gap junctions and stimulus."""
    nvar = 23  # number of variables per cell
    dYdt = np.zeros_like(Y)
    
    # Unpack stimulus parameters
    I_app_val, stim_cell, stim_start, stim_end = stim_params
    
    for i in range(Ng):
        idx_start = i*nvar
        idx_end = idx_start + nvar
        
        # Get current cell state
        y_i = Y[idx_start:idx_end]
        
        # Calculate single cell dynamics
        dyi = model_core(t, y_i)
        
        # Get voltage derivatives and values
        dVdt_single = dyi[0]
        V_i = y_i[0]
        
        # Calculate neighboring voltages
        V_left = Y[max(0, (i-1)*nvar)] if i > 0 else V_i
        V_right = Y[min((Ng-1)*nvar, (i+1)*nvar)] if i < Ng-1 else V_i
        
        # Calculate gap junction current
        I_gj = (g_gap/(dx**2))*(V_left + V_right - 2.0*V_i)
        dVdt_multi = dVdt_single + (I_gj/cm)
        
        # Add stimulus if applicable
        if (stim_start <= t <= stim_end) and (i == stim_cell):
            dVdt_multi += (I_app_val/cm)
        
        # Update voltage derivative
        dyi[0] = dVdt_multi
        dYdt[idx_start:idx_end] = dyi
    
    return dYdt

def run_multi_cell_simulation(Ng=5, g_gap=0.02, t_span=(0,300), dt=1.0,
                            stim_params=(50.0, 2, 100.0, 400.0)):
    """Run multi-cell simulation with given parameters."""
    print("[DEBUG] Starting multi-cell simulation...")
    start_time = time.perf_counter()
    
    # Setup time points
    t_eval = np.arange(t_span[0], t_span[1]+dt, dt)
    
    # Initial conditions for all cells
    Y0 = np.zeros(23 * Ng)
    for i in range(Ng):
        idx = i * 23
        Y0[idx] = initial_voltage
        Y0[idx + 1] = initial_calcium
        Y0[idx + 2] = initial_atp
        Y0[idx + 3] = initial_dpmca
        Y0[idx + 4] = Ca_ER_initial
        # Set IP3R states
        Y0[idx + 5:idx + 13] = [x000_initial, x100_initial, x010_initial,
                               x001_initial, x110_initial, x101_initial,
                               x011_initial, x111_initial]
        Y0[idx + 13] = IP3_initial
        Y0[idx + 14] = Ca_m_initial
        Y0[idx + 15:idx + 23] = [x0002_initial, x1002_initial, x0102_initial,
                                x0012_initial, x1102_initial, x1012_initial,
                                x0112_initial, x1112_initial]
    
    # Solve system
    sol = solve_ivp(
        lambda t, Y: multi_cell_model(t, Y, Ng, g_gap, 1.0, membrane_capacitance, stim_params),
        t_span, Y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    
    end_time = time.perf_counter()
    print(f"[DEBUG] Multi-cell simulation completed in {end_time - start_time:.4f} s.")
    return sol

def plot_results(sol, Ng=5):
    """Plot simulation results."""
    plt.figure(figsize=(10, 6))
    for i in range(Ng):
        plt.plot(sol.t, sol.y[i*23], label=f'Cell {i+1}')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Multi-cell Pericyte Model with Gap Junctions")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run multi-cell simulation
    sol = run_multi_cell_simulation(
        Ng=5,
        g_gap=0.02,
        t_span=(0, 300),
        dt=1.0,
        stim_params=(50.0, 2, 100.0, 400.0)
    )
    
    # Plot results
    plot_results(sol)