import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE system
def model(y, t, params, drug):
    RL, cAMP, D = y  # Receptor-ligand complex, cAMP, diameter
    kon, koff, P_basal, P_inhibited_max, D_rate, k_constriction, D_baseline = params

    # Drug concentrations
    if 200 <= t <= 600:  # Drug application period
        if drug == "NA":
            L = 2.0  # Noradrenaline concentration
        elif drug == "NA+atipamezole":
            L = 2.0  # NA concentration
            P_inhibited_max = 0.0  # Atipamezole blocks α2 receptor activation
        elif drug == "clonidine":
            L = 2.0  # Clonidine concentration
        elif drug == "xylazine":
            L = 2.0  # Xylazine concentration
        elif drug == "phenylephrine":
            L = 0.0  # Phenylephrine does not activate α2 receptors
    else:
        L = 0.1  # Baseline NA concentration

    # Receptor-ligand binding
    R = 1.0 - RL  # Free receptor (assuming total receptor = 1)
    dRL_dt = kon * R * L - koff * RL

    # cAMP dynamics
    P_inhibited = P_inhibited_max * RL
    dcAMP_dt = P_basal - P_inhibited - D_rate * cAMP

    # Capillary diameter dynamics
    D_target = D_baseline * (1 - 0.65 * (1 - cAMP / (cAMP + 0.5)))
    dD_dt = -k_constriction * (D - D_target)

    return [dRL_dt, dcAMP_dt, dD_dt]

# Function to run the simulation for a specific drug
def run_simulation(drug, params, y0, t):
    results = odeint(model, y0, t, args=(params, drug))
    RL, cAMP, D = results.T
    return RL, cAMP, D

# Initial conditions
y0 = [0.0, 1.0, 4.0]  # Initial [RL], cAMP, and diameter
t = np.linspace(0, 1000, 10000)  # Time points

# Parameters
params = [0.1, 0.01, 1.0, 0.98, 0.1, 0.1, 4.0]  # Example parameter values

# Drugs to simulate
drugs = ["NA", "NA+atipamezole", "clonidine", "xylazine", "phenylephrine"]

# Run simulations and calculate constriction percentages
results = {}
for drug in drugs:
    RL, cAMP, D = run_simulation(drug, params, y0, t)
    baseline_diameter = D[0]  # Diameter before drug application
    min_diameter = np.min(D)  # Minimum diameter during drug application
    constriction_percentage = 100 * (baseline_diameter - min_diameter) / baseline_diameter
    results[drug] = constriction_percentage
    print(f"{drug} constriction: {constriction_percentage:.1f}%")

# Plot the results for visualization
plt.figure(figsize=(12, 6))

# Plot capillary diameter for each drug
for drug in drugs:
    _, _, D = run_simulation(drug, params, y0, t)
    plt.plot(t, D, label=drug)

plt.title("Capillary Diameter Dynamics")
plt.xlabel("Time (s)")
plt.ylabel("Diameter (µm)")
plt.legend()
plt.show()