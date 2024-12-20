import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants
faraday_constant = 96485
gas_constant = 8314
temperature = 310

# Cell parameters
membrane_capacitance = 0.94  # pF
cell_volume = 2e-12  # L
Vcyto = cell_volume
ER_volume = cell_volume*0.2

# Ion valences
valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2

# Ion concentrations in mM
K_out = 6.26
K_in = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

# De Young-Keizer IP3R parameters (original T-cell style)
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

# PMCA parameters (from T-cell model)
vu = 1540000.0
vm = 2200000.0
aku = 0.303
akmp = 0.14
aru = 1.8
arm = 2.1
p1 = 0.1
p2 = 0.01

# Buffer parameters (enhanced)
Bscyt = 225.0
aKscyt = 0.1
Bser = 2000.0
aKser = 1.0
Bm = 111.0
aKm = 0.123

# Other parameters
conductance_Kir61 = 0.025
conductance_TRPC1 = 0.001
conductance_CaL = 0.0005
conductance_CaCC = 0.001
conductance_leak = 0.01
conductance_IP3R1 = 0.1
conductance_IP3R2 = 0.05
conductance_RyR = 0.01
calcium_extrusion_rate = 100.0
resting_calcium = 0.001
calcium_activation_threshold_CaCC = 0.0005
k_serca = 0.1
Km_serca = 0.5
leak_rate_er = 0.05
k_ncx = 0.001

# IP3 kinetics (Phase 4)
prodip3 = 5.0
V2ip3 = 12.5
ak2ip3 = 6.0
V3ip3 = 0.9
ak3ip3 = 0.1
ak4ip3 = 1.0

# Time simulation parameters
simulation_duration = 100
time_points = 1000

# Initial conditions
initial_voltage = -70
initial_calcium = Ca_in
initial_atp = 4.4
initial_dpmca = 1.0
Ca_ER_initial = 0.5
IP3_initial = 0.1

# Normalize IP3R states
x000_initial = 0.27
x100_initial = 0.039
x010_initial = 0.29
x001_initial = 0.17
x110_initial = 0.042
x101_initial = 0.0033
x011_initial = 0.18
x111_initial = 0.0035
total = x000_initial + x100_initial + x010_initial + x001_initial + x110_initial + x101_initial + x011_initial + x111_initial
x000_initial /= total
x100_initial /= total
x010_initial /= total
x001_initial /= total
x110_initial /= total
x101_initial /= total
x011_initial /= total
x111_initial /= total

def calculate_reversal_potentials(K_out, K_in, Ca_out, Ca_i, Na_out, Na_in, Cl_out, Cl_in):
    reversal_potential_K = (gas_constant*temperature/(valence_K*faraday_constant))*np.log(K_out/K_in)
    reversal_potential_Ca = (gas_constant*temperature/(valence_Ca*faraday_constant))*np.log(Ca_out/Ca_i)
    reversal_potential_Na = (gas_constant*temperature/(valence_Na*faraday_constant))*np.log(Na_out/Na_in)
    reversal_potential_Cl = (gas_constant*temperature/(valence_Cl*faraday_constant))*np.log(Cl_out/Cl_in)
    return reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl

def calculate_buffering_factors(Ca_i, Ca_ER):
    beta_cyt = 1.0/(1.0+(Bscyt*aKscyt)/((aKscyt+Ca_i)**2)+(Bm*aKm)/((aKm+Ca_i)**2))
    beta_er = 1.0/(1.0+(Bser*aKser)/((aKser+Ca_ER)**2)+(Bm*aKm)/((aKm+Ca_ER)**2))
    return beta_cyt, beta_er

def calculate_ip3r_states(Ca_i, x000, x100, x010, x001, x110, x101, x011, x111, IP3):
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

    dx000_dt = f1+f2+f3
    dx100_dt = f4+f5-f2
    dx010_dt = -f1+f6+f7
    dx001_dt = f8-f3+f9
    dx110_dt = -f4-f6+f10
    dx101_dt = f11-f9-f5
    dx011_dt = -f8-f7+f12
    dx111_dt = -f11-f12-f10

    return dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt

def calculate_ip3r_flux(x110, Ca_ER, Ca_i):
    return v1*(x110**3)*(Ca_ER-Ca_i)

def calculate_pmca_flux(Ca_i, dpmca):
    Ca_i = max(Ca_i,1e-12)
    u4 = (vu*(Ca_i**aru))/(Ca_i**aru+aku**aru)
    u5 = (vm*(Ca_i**arm))/(Ca_i**arm+akmp**arm)
    cJpmca = (dpmca*u4+(1-dpmca)*u5)/(6.6253e5)
    return cJpmca, u4, u5

def calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, IP3, params):
    d_inf = 1/(1+np.exp(-(V-params['activation_midpoint_CaL'])/params['activation_slope_CaL']))
    f_inf = 1/(1+np.exp((V - params['inactivation_midpoint_CaL'])/params['inactivation_slope_CaL'])) + \
            params['amplitude_factor_CaL']/(1+np.exp((params['voltage_shift_CaL'] - V)/params['slope_factor_CaL']))

    I_Kir61 = params['conductance_Kir61']*atp*np.sqrt(params['K_out']/params['reference_K'])*(V-params['reversal_potential_K'])/ \
              (1+np.exp((V-params['reversal_potential_K']-params['voltage_shift_Kir61'])/params['voltage_slope_Kir61']))
    I_TRPC1 = params['conductance_TRPC1']*(V-params['reversal_potential_Ca'])
    I_CaCC = params['conductance_CaCC']*(Ca_i/(Ca_i+params['calcium_activation_threshold_CaCC']))*(V-params['reversal_potential_Cl'])
    I_CaL = params['conductance_CaL']*d_inf*f_inf*(V-params['reversal_potential_Ca'])
    I_leak = params['conductance_leak']*(V-params['reversal_potential_K'])

    cJpmca, _, _ = calculate_pmca_flux(Ca_i, dpmca)
    I_PMCA = cJpmca*z*faraday_constant*Vcyto*1e6

    J_SERCA = params['k_serca']*(Ca_i**2)/(Ca_i**2+params['Km_serca']**2)
    J_ER_leak = params['leak_rate_er']*(Ca_ER - Ca_i)
    I_NCX = params['k_ncx']*(params['Na_in']**3/(params['Na_in']**3+87.5**3))*(params['Ca_out']/(params['Ca_out']+1))
    J_IP3R = calculate_ip3r_flux(x110, Ca_ER, Ca_i)
    J_RyR = params['conductance_RyR']*(Ca_i/(Ca_i+0.3))*(Ca_ER - Ca_i)

    # Return 13 values now (no J_IP3R2)
    return I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR

def ip3_ode(Ca_i, IP3):
    IP3 = max(IP3,1e-12)
    term1 = prodip3
    term2 = V2ip3/(1+(ak2ip3/IP3))
    term3 = (V3ip3/(1+(ak3ip3/IP3)))*(1/(1+(ak4ip3/max(Ca_i,1e-12))))
    dIP3dt = term1 - term2 - term3
    return dIP3dt

def model(t, y, params):
    V, Ca_i, atp, dpmca, Ca_ER, x000, x100, x010, x001, x110, x101, x011, x111, IP3 = y
    beta_cyt, beta_er = calculate_buffering_factors(Ca_i, Ca_ER)
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR = \
        calculate_currents(V, Ca_i, atp, dpmca, Ca_ER, x110, IP3, params)

    dV_dt = (-I_Kir61 - I_TRPC1 - I_CaCC - I_CaL - I_leak - I_PMCA - I_NCX)/membrane_capacitance
    Ca_influx = -I_CaL/(2*faraday_constant*cell_volume)
    Ca_efflux = (-I_PMCA - I_NCX)/(2*faraday_constant*cell_volume)
    dCa_dt = beta_cyt*(Ca_influx+Ca_efflux+J_ER_leak+J_IP3R+J_RyR-J_SERCA)
    dCa_ER_dt = beta_er*((Vcyto/ER_volume)*(J_SERCA - J_ER_leak - J_IP3R - J_RyR))

    datp_dt = 0
    w1 = p1*Ca_i
    w2 = p2
    taom = 1/(w1+w2)
    dpmcainf = w2/(w1+w2)
    ddpmca_dt = (dpmcainf - dpmca)/taom

    dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt = \
        calculate_ip3r_states(Ca_i, x000, x100, x010, x001, x110, x101, x011, x111, IP3)

    dIP3_dt = ip3_ode(Ca_i, IP3)

    if abs(t-round(t))<1e-9 and round(t)%10==0:
        print(f"Time: {t:.2f} ms, V: {V:.2f} mV, Ca_i: {Ca_i:.6f}, Ca_ER: {Ca_ER:.6f}, IP3: {IP3:.6f}")
        print(f"IP3R (x110^3): {(x110**3):.6f}, IP3R flux: {J_IP3R:.6f}, PMCA: {dpmca:.6f}")

    return [dV_dt, dCa_dt, datp_dt, ddpmca_dt, dCa_ER_dt,
            dx000_dt, dx100_dt, dx010_dt, dx001_dt, dx110_dt, dx101_dt, dx011_dt, dx111_dt, dIP3_dt]

def run_simulation(params):
    t_span = (0, params['simulation_duration'])
    t_eval = np.linspace(0, params['simulation_duration'], params['time_points'])
    y0 = [params['initial_voltage'], params['initial_calcium'], params['initial_atp'], params['initial_dpmca'],
          params['Ca_ER_initial'], x000_initial, x100_initial, x010_initial, x001_initial, x110_initial,
          x101_initial, x011_initial, x111_initial, IP3_initial]

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=(params,), dense_output=True)
    return sol

def plot_results_with_two_figures(sol, params):
    # Now we have 13 variables from calculate_currents instead of 14
    I_Kir61, I_TRPC1, I_CaCC, I_CaL, I_leak, d_inf, f_inf, I_PMCA, J_SERCA, J_ER_leak, I_NCX, J_IP3R, J_RyR = [
        np.zeros_like(sol.t) for _ in range(13)
    ]
    beta_cyt = np.zeros_like(sol.t)
    beta_er = np.zeros_like(sol.t)

    V = sol.y[0]
    Ca_i = sol.y[1]
    dpmca = sol.y[3]
    Ca_ER = sol.y[4]
    x110 = sol.y[9]
    IP3 = sol.y[-1]

    for i in range(len(sol.t)):
        beta_cyt[i], beta_er[i] = calculate_buffering_factors(Ca_i[i], Ca_ER[i])
        (I_Kir61[i], I_TRPC1[i], I_CaCC[i], I_CaL[i], I_leak[i], d_inf[i], f_inf[i], I_PMCA[i],
         J_SERCA[i], J_ER_leak[i], I_NCX[i], J_IP3R[i], J_RyR[i]) = \
            calculate_currents(V[i], Ca_i[i], params['initial_atp'], dpmca[i], Ca_ER[i], x110[i], IP3[i], params)

    fig1, axs1 = plt.subplots(7, 1, figsize=(12, 35))
    fig1.suptitle('Cell Dynamics - Part 1 (Phases 1-4)', fontsize=16, fontweight='bold')

    axs1[0].plot(sol.t, V, color='blue')
    axs1[0].set_title('Membrane Potential', fontsize=14, fontweight='bold')
    axs1[0].set_ylabel('Vm (mV)')
    axs1[0].grid(True)

    axs1[1].plot(sol.t, Ca_i, color='orange')
    axs1[1].set_title('Intracellular Ca²⁺ (mM)', fontsize=14, fontweight='bold')
    axs1[1].grid(True)

    axs1[2].plot(sol.t, Ca_ER, color='red')
    axs1[2].set_title('ER Ca²⁺ (mM)', fontsize=14, fontweight='bold')
    axs1[2].grid(True)

    axs1[3].plot(sol.t, x110, label='x110', color='purple')
    axs1[3].set_title('IP3R x110 State', fontsize=14, fontweight='bold')
    axs1[3].legend()
    axs1[3].grid(True)

    axs1[4].plot(sol.t, beta_cyt, label='Cytosolic', color='green')
    axs1[4].plot(sol.t, beta_er, label='ER', color='blue')
    axs1[4].set_title('Buffering Factors', fontsize=14, fontweight='bold')
    axs1[4].legend()
    axs1[4].grid(True)

    axs1[5].plot(sol.t, dpmca, label='dPMCA', color='cyan')
    axs1[5].set_title('PMCA Modulation State', fontsize=14, fontweight='bold')
    axs1[5].grid(True)

    axs1[6].plot(sol.t, IP3, label='IP3', color='magenta')
    axs1[6].set_title('IP3 Concentration (mM)', fontsize=14, fontweight='bold')
    axs1[6].grid(True)

    fig1.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 20))
    fig2.suptitle('Cell Dynamics - Part 2 (Phases 1-4)', fontsize=16, fontweight='bold')

    axs2[0].plot(sol.t, I_Kir61, label='I_Kir61')
    axs2[0].plot(sol.t, I_TRPC1, label='I_TRPC1')
    axs2[0].plot(sol.t, I_CaCC, label='I_CaCC')
    axs2[0].plot(sol.t, I_CaL, label='I_CaL')
    axs2[0].plot(sol.t, I_leak, label='I_leak')
    axs2[0].plot(sol.t, I_PMCA, label='I_PMCA')
    axs2[0].plot(sol.t, I_NCX, label='I_NCX')
    axs2[0].set_title('Membrane Currents Over Time', fontsize=14, fontweight='bold')
    axs2[0].legend(loc='center left', bbox_to_anchor=(1,0.5))
    axs2[0].grid(True)

    axs2[1].plot(sol.t, J_SERCA, label='J_SERCA', color='purple')
    axs2[1].plot(sol.t, J_ER_leak, label='J_ER_leak', color='cyan')
    axs2[1].plot(sol.t, J_IP3R, label='J_IP3R', color='pink')
    axs2[1].plot(sol.t, J_RyR, label='J_RyR', color='brown')
    axs2[1].set_title('Calcium Fluxes Over Time', fontsize=14, fontweight='bold')
    axs2[1].legend(loc='center left', bbox_to_anchor=(1,0.5))
    axs2[1].grid(True)

    axs2[2].plot(sol.t, d_inf, label='d_inf', linestyle='--', color='purple')
    axs2[2].plot(sol.t, f_inf, label='f_inf', linestyle='--', color='brown')
    axs2[2].set_title('Gating Variables (CaL) Over Time', fontsize=14, fontweight='bold')
    axs2[2].legend(loc='center left', bbox_to_anchor=(1,0.5))
    axs2[2].grid(True)

    axs2[3].plot(sol.t, dpmca, label='dPMCA', color='cyan')
    axs2[3].set_title('PMCA Modulation State Over Time (Repeat)', fontsize=14, fontweight='bold')
    axs2[3].legend(loc='center left', bbox_to_anchor=(1,0.5))
    axs2[3].grid(True)

    fig2.tight_layout()
    plt.show()

params = {
    'K_out': K_out, 'K_in': K_in, 'Ca_out': Ca_out, 'Ca_in': Ca_in,
    'Na_out': Na_out, 'Na_in': Na_in, 'Cl_out': Cl_out, 'Cl_in': Cl_in,
    'conductance_Kir61': conductance_Kir61, 'conductance_TRPC1': conductance_TRPC1,
    'conductance_CaCC': conductance_CaCC, 'conductance_CaL': conductance_CaL,
    'conductance_leak': conductance_leak,
    'conductance_IP3R1': conductance_IP3R1,
    'conductance_IP3R2': conductance_IP3R2,
    'conductance_RyR': conductance_RyR,
    'k_serca': k_serca,
    'Km_serca': Km_serca,
    'leak_rate_er': leak_rate_er,
    'k_ncx': k_ncx,
    'calcium_extrusion_rate': calcium_extrusion_rate,
    'resting_calcium': resting_calcium,
    'calcium_activation_threshold_CaCC': calcium_activation_threshold_CaCC,
    'reference_K': 5.4,
    'voltage_slope_Kir61': 6,
    'voltage_shift_Kir61': 15,
    'activation_midpoint_CaL': -40,
    'activation_slope_CaL': 4,
    'inactivation_midpoint_CaL': -45,
    'inactivation_slope_CaL': 5,
    'voltage_shift_CaL': 50,
    'slope_factor_CaL': 20,
    'amplitude_factor_CaL': 0.6,
    'vu': vu,
    'vm': vm,
    'aku': aku,
    'akmp': akmp,
    'aru': aru,
    'arm': arm,
    'initial_dpmca': initial_dpmca,
    'simulation_duration': simulation_duration,
    'time_points': time_points,
    'initial_voltage': initial_voltage,
    'initial_calcium': initial_calcium,
    'initial_atp': initial_atp,
    'Ca_ER_initial': Ca_ER_initial
}

reversal_potential_K, reversal_potential_Ca, reversal_potential_Na, reversal_potential_Cl = calculate_reversal_potentials(
    params['K_out'], params['K_in'], params['Ca_out'], params['Ca_in'],
    params['Na_out'], params['Na_in'], params['Cl_out'], params['Cl_in'])

params.update({
    'reversal_potential_K': reversal_potential_K,
    'reversal_potential_Ca': reversal_potential_Ca,
    'reversal_potential_Na': reversal_potential_Na,
    'reversal_potential_Cl': reversal_potential_Cl
})

sol = run_simulation(params)
plot_results_with_two_figures(sol, params)

