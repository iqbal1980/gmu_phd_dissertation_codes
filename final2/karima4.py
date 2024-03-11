import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the model parameters
gamma = 0.5  # Choose a gamma value that might be typical for cardiac tissue

# Calculate v1 and v2 based on the given formulas
v1 = 0.12 / (30 + gamma)
v2 = 30.12 / (30 + gamma)

# Define the piecewise functions F(v) and tau(v)
def F(v):
    if v < v1:
        return -30 * v
    if v1 < v < v2:
        return gamma * v - 0.12
    if v > v2:
        return -30 * (v - 1)

def tau(v):
    if v < v1:
        return 2
    if v > v1:
        return 16.6
        
# This example assumes you want to replace tau(v) with a constant value for the sake of the demonstration.
def puschino_model_with_tau(tau_constant):
    def model(t, variables):
        v, w = variables
        dvdt = F(v) - w
        dwdt = (v - w) / tau_constant  # Use tau_constant directly
        return [dvdt, dwdt]
    return model
        

# Define the system of differential equations
def puschino_model(t, variables):
    v, w = variables
    dvdt = F(v) - w
    dwdt = (v - w) / tau(v)
    return [dvdt, dwdt]

# Initial conditions (these may need to be adjusted based on the expected behavior of the model)
v0 = -1.0  # A typical value to start near the resting membrane potential
w0 = 1.0
initial_conditions = [v0, w0]

# Time span for the simulation (this can be adjusted based on the expected duration of action potentials)
t_span = [0, 50]  # simulate for 50 time units
t_eval = np.linspace(t_span[0], t_span[1], 500)  # create a time grid for output

 

tau_values = [ 2,  16.6]  # Example values

solutions = []  # Store tuples of (tau_value, solution)

for tau_constant in tau_values:
    # Create a model function that incorporates the current tau_constant
    model_with_current_tau = puschino_model_with_tau(tau_constant)
    solution = solve_ivp(model_with_current_tau, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    solutions.append((tau_constant, solution))


# Plot the results for different tau values for both v(t) and w(t)
plt.figure(figsize=(12, 6))

for i, (tau_constant, solution) in enumerate(solutions):
    # Create subplots for both v(t) and w(t)
    plt.subplot(1, len(solutions), i + 1)
    plt.plot(solution.t, solution.y[0], 'b-', label=f'v(t) - Membrane potential (τ = {tau_constant})')
    plt.plot(solution.t, solution.y[1], 'r--', label=f'w(t) - Recovery variable (τ = {tau_constant})')
    plt.title(f'Action Potential & Recovery\nfor τ = {tau_constant}')
    plt.xlabel('Time')
    plt.ylabel('Variables')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
