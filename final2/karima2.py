import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the model parameters
gamma = 0.5  # Choose a gamma value that might be more typical for cardiac tissue

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

# Solve the system of differential equations
solution = solve_ivp(puschino_model, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Check for any solver issues
solver_message = solution.message

# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(solution.t, solution.y[0], label='v(t) - Membrane potential')
plt.plot(solution.t, solution.y[1], label='w(t) - Recovery variable')
plt.title('Simulation of Ventricular Action Potential using Puschino model')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.legend()
plt.grid(True)
plt.show()

# Output any message from the solver
print(solver_message)
