import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(x, t, mu):
    return mu * x

# Parameter values
mu_vals = [-0.5, 0, 0.5]
t = np.linspace(0, 10, 200)

# Initial condition
x0 = 1

# Solve the ODE for different mu values
for mu in mu_vals:
    sol = odeint(model, x0, t, args=(mu,))
    plt.plot(t, sol, label=f'mu = {mu}')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()




#--------------------------------------------------------------------------------

def analyze_system(mu):
    fixed_points = [0]  # The system has a fixed point at x = 0 for all mu
    stability = 'stable' if mu < 0 else 'unstable'
    return fixed_points, stability

# Range of mu values to analyze
mu_vals = np.linspace(-2, 2, 100)

# Analyze the system for each mu value
fixed_points = []
stabilities = []
for mu in mu_vals:
    fp, stab = analyze_system(mu)
    fixed_points.extend(fp)
    stabilities.append(stab)

# Create the bifurcation diagram
plt.figure(figsize=(8, 6))
plt.plot(mu_vals, fixed_points, 'b-', linewidth=2)
plt.xlabel('mu')
plt.ylabel('Fixed Points')
plt.title('Bifurcation Diagram')

# Add stability information
for i, mu in enumerate(mu_vals):
    if stabilities[i] == 'stable':
        plt.plot(mu, 0, 'g.', markersize=10)
    else:
        plt.plot(mu, 0, 'r.', markersize=10)

plt.xlim(mu_vals[0], mu_vals[-1])
plt.ylim(-0.1, 0.1)
plt.show()



import PyDSTool as dst

# Define the ODE system
ode = dst.Generator.Vode_ODEsystem({"x'": "mu * x"}, ["x"], ["mu"])

# Set up the continuation
PC = dst.ContClass(ode)
PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars = ['mu']
PCargs.MaxNumPoints = 100
PCargs.MaxStepSize = 0.1
PCargs.MinStepSize = 1e-5
PCargs.StepSize = 0.01
PCargs.LocBifPoints = 'all'
PCargs.verbosity = 2
PCargs.SaveEigen = True

# Run the continuation
PC.newCurve(PCargs)
PC['EQ1'].forward()

# Display the bifurcation diagram
PC.display(('mu','x'))