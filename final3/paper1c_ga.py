import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants
faraday_constant = 96485  # C/mol
gas_constant = 8314       # J/(mol*K)
temperature = 310         # K

# Cell parameters
membrane_capacitance = 0.94  # pF
cell_volume = 2e-12          # L
Vcyto = cell_volume
ER_volume = cell_volume*0.2

# Ion valences
valence_K = 1
valence_Ca = 2
valence_Na = 1
valence_Cl = -1
z = 2  # Ca2+

# Concentrations in mM
K_out = 6.26
K_in = 140
Ca_out = 2.0
Ca_in = 0.0001
Na_out = 140
Na_in = 15.38
Cl_out = 110
Cl_in = 9.65

# Noradrenaline and contractility parameters
NA_sensitivity = 0.1      # Sensitivity to noradrenaline
cAMP_basal = 1.0         # Baseline cAMP level
MLCP_sensitivity = 0.5    # MLCP sensitivity to cAMP
baseline_tone = 0.2      # Baseline contractile tone
max_contraction = 0.6    # Maximum contraction response

# Add these parameters to your existing parameter set
params = {
    'K_out': K_out,
    'K_in': K_in,
    'Ca_out': Ca_out,
    'Ca_in': Ca_in,
    'Na_out': Na_out,
    'Na_in': Na_in,
    'Cl_out': Cl_out,
    'Cl_in': Cl_in,
    'NA_sensitivity': NA_sensitivity,
    'cAMP_basal': cAMP_basal,
    'MLCP_sensitivity': MLCP_sensitivity,
    'baseline_tone': baseline_tone,
    'max_contraction': max_contraction,
    'initial_voltage': -70,        # mV
    'initial_calcium': Ca_in,      # mM
    'initial_atp': 4.4,           # mM
    'initial_dpmca': 1.0,         # dimensionless
    'Ca_ER_initial': 0.5,         # mM
}

# Your existing IP3R and other parameters here...
# [Previous parameters remain unchanged]

def calculate_NA_response(NA_concentration, cAMP, tone):
    """Calculate the noradrenaline-induced response"""
    # cAMP dynamics
    cAMP_change = -NA_sensitivity * NA_concentration * cAMP + params['cAMP_basal']

    # MLCP activity
    MLCP_activity = cAMP / (params['MLCP_sensitivity'] + cAMP)

    # Contractile tone
    tone_change = (params['max_contraction'] * (1 - MLCP_activity) - tone)

    return cAMP_change, tone_change

def model_with_NA(t, y, NA_concentration=0):
    """Extended model including noradrenaline effects"""
    # Unpack state variables
    V = y[0]
    Ca_i = y[1]
    atp = y[2]
    dpmca = y[3]
    Ca_ER = y[4]
    cAMP = y[5]    # New state variable
    tone = y[6]    # New state variable

    # Calculate membrane potential dynamics
    dV_dt = -0.1 * (V + 70)  # Simple relaxation to resting potential

    # Calculate calcium dynamics
    dCa_dt = -0.1 * (Ca_i - Ca_in)  # Simple relaxation to baseline

    # Calculate PMCA dynamics
    ddpmca_dt = -0.1 * (dpmca - 1.0)  # Simple relaxation to baseline

    # Calculate ER calcium dynamics
    dCa_ER_dt = -0.1 * (Ca_ER - 0.5)  # Simple relaxation to baseline

    # Calculate NA response
    cAMP_change, tone_change = calculate_NA_response(NA_concentration, cAMP, tone)

    # Combine all derivatives
    derivatives = [
        dV_dt,
        dCa_dt,
        0,  # ATP is constant
        ddpmca_dt,
        dCa_ER_dt,
        cAMP_change,    # Add cAMP dynamics
        tone_change     # Add contractile tone dynamics
    ]

    return derivatives

def run_simulation_with_NA(NA_concentration, simulation_duration=100):
    """Run simulation with specified NA concentration"""
    t_span = (0, simulation_duration)
    t_eval = np.linspace(0, simulation_duration, 1000)

    # Initial conditions including new variables
    y0 = [
        params['initial_voltage'],
        params['initial_calcium'],
        params['initial_atp'],
        params['initial_dpmca'],
        params['Ca_ER_initial'],
        params['cAMP_basal'],    # Initial cAMP
        params['baseline_tone']   # Initial tone
    ]

    sol = solve_ivp(
        lambda t, y: model_with_NA(t, y, NA_concentration),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45'
    )

    return sol

def plot_NA_response(NA_concentrations=[0, 2, 200]):
    """Plot cellular response to different NA concentrations"""
    plt.figure(figsize=(15, 10))

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

    for NA_conc in NA_concentrations:
        sol = run_simulation_with_NA(NA_conc)

        # Plot membrane potential
        ax1.plot(sol.t, sol.y[0], label=f'NA={NA_conc}µM')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.grid(True)

        # Plot calcium
        ax2.plot(sol.t, sol.y[1], label=f'NA={NA_conc}µM')
        ax2.set_ylabel('[Ca²⁺]ᵢ (mM)')
        ax2.grid(True)

        # Plot cAMP
        ax3.plot(sol.t, sol.y[5], label=f'NA={NA_conc}µM')
        ax3.set_ylabel('[cAMP] (a.u.)')
        ax3.grid(True)

        # Plot contractile tone
        ax4.plot(sol.t, sol.y[6], label=f'NA={NA_conc}µM')
        ax4.set_ylabel('Contractile Tone')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)

    # Add legends and titles
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend()

    fig.suptitle('Pericyte Response to Noradrenaline')
    plt.tight_layout()
    plt.show()

# Additional parameters for drug effects
drug_params = {
    'clonidine_efficacy': 0.8,    # Relative to NA
    'xylazine_efficacy': 0.7,     # Relative to NA
    'atipamezole_block': 0.9,     # Degree of α2R blockade
    'drug_onset_time': 10,        # seconds
    'drug_duration': 60           # seconds
}

def simulate_drug_effect(drug_type, NA_concentration=0):
    """
    Simulate the effect of different drugs on the pericyte response
    drug_type: 'clonidine', 'xylazine', or 'atipamezole'
    """
    simulation_duration = 100  # seconds
    t_span = (0, simulation_duration)
    t_eval = np.linspace(0, simulation_duration, 1000)

    def drug_effect(t):
        if drug_type == 'atipamezole':
            # Antagonist effect
            if t < drug_params['drug_onset_time']:
                return 1.0
            else:
                return 1.0 - drug_params['atipamezole_block']
        else:
            # Agonist effects
            if t < drug_params['drug_onset_time']:
                return 0.0
            elif t < drug_params['drug_onset_time'] + drug_params['drug_duration']:
                if drug_type == 'clonidine':
                    return drug_params['clonidine_efficacy']
                else:  # xylazine
                    return drug_params['xylazine_efficacy']
            else:
                return 0.0

    # Modified model function to include drug effects
    def model_with_drug(t, y):
        drug_modifier = drug_effect(t)
        if drug_type == 'atipamezole':
            effective_NA = NA_concentration * drug_modifier
        else:
            effective_NA = NA_concentration + (drug_modifier * NA_concentration)
        return model_with_NA(t, y, effective_NA)

    # Initial conditions (same as before)
    y0 = [
        params['initial_voltage'],
        params['initial_calcium'],
        params['initial_atp'],
        params['initial_dpmca'],
        params['Ca_ER_initial'],
        params['cAMP_basal'],
        params['baseline_tone']
    ]

    sol = solve_ivp(model_with_drug, t_span, y0, t_eval=t_eval, method='RK45')
    return sol

def plot_drug_experiments():
    """Plot comprehensive drug experiment results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Experiment 1: NA alone
    sol_na = run_simulation_with_NA(200)  # 200 µM NA
    ax1.plot(sol_na.t, sol_na.y[6], 'b-', label='NA 200µM')
    ax1.set_title('NA Response')
    ax1.set_ylabel('Contractile Tone')
    ax1.grid(True)
    ax1.legend()

    # Experiment 2: Clonidine
    sol_clon = simulate_drug_effect('clonidine', NA_concentration=0)
    ax2.plot(sol_clon.t, sol_clon.y[6], 'g-', label='Clonidine')
    ax2.set_title('Clonidine Effect')
    ax2.grid(True)
    ax2.legend()

    # Experiment 3: Xylazine
    sol_xyl = simulate_drug_effect('xylazine', NA_concentration=0)
    ax3.plot(sol_xyl.t, sol_xyl.y[6], 'r-', label='Xylazine')
    ax3.set_title('Xylazine Effect')
    ax3.set_ylabel('Contractile Tone')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    ax3.legend()

    # Experiment 4: Atipamezole + NA
    sol_atip = simulate_drug_effect('atipamezole', NA_concentration=200)
    ax4.plot(sol_atip.t, sol_atip.y[6], 'm-', label='Atipamezole + NA')
    ax4.set_title('Atipamezole + NA Effect')
    ax4.set_xlabel('Time (s)')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

def analyze_concentration_response():
    """Analyze and plot concentration-response relationships"""
    NA_concentrations = np.logspace(-2, 3, 20)  # 0.01 to 1000 µM
    max_responses = []

    for conc in NA_concentrations:
        sol = run_simulation_with_NA(conc)
        max_responses.append(np.max(sol.y[6]))

    plt.figure(figsize=(8, 6))
    plt.semilogx(NA_concentrations, max_responses, 'bo-')
    plt.xlabel('NA Concentration (µM)')
    plt.ylabel('Maximum Contractile Response')
    plt.grid(True)
    plt.title('NA Concentration-Response Relationship')
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Run basic NA response simulation
    plot_NA_response()

    # Run drug experiments
    plot_drug_experiments()

    # Run concentration-response analysis
    analyze_concentration_response()