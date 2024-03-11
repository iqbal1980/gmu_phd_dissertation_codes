import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
Cm = 1.0 # membrane capacitance, in uF/cm^2
gNa = 120.0 # maximum conducances, in mS/cm^2
gK = 36.0
gL = 0.3
ENa = 50.0 # Nernst reversal potentials, in mV
EK = -77.0
EL = -54.387

# Equations describing the gating variables
def alpha_m(V): return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
def beta_m(V): return 4.0*np.exp(-(V+65.0) / 18.0)
def alpha_h(V): return 0.07*np.exp(-(V+65.0) / 20.0)
def beta_h(V): return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
def alpha_n(V): return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))
def beta_n(V): return 0.125*np.exp(-(V+65) / 80.0)

# The HH model differential equations.
def dALLdt(X, t, I):
    V, m, h, n = X
    dVdt = (I - gNa*m**3*h*(V - ENa) - gK*n**4*(V - EK) - gL*(V - EL)) / Cm
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dmdt, dhdt, dndt

# Initial conditions
V0 = -65.0 # initial membrane potential
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
X0 = V0, m0, h0, n0

# Time vector
t = np.linspace(0, 50, 1000) # 50 ms divided into 1000 points

# Solve ODEs for different initial conditions and with/without applied current
solutions = {}
for V_initial in [-65, -60, -57]:
    X0 = V_initial, m0, h0, n0
    solutions[V_initial] = odeint(dALLdt, X0, t, args=(0,))

# Apply a current of 15 uA/cm^2 and solve again
I_app = 15
X0 = -57, m0, h0, n0
solutions['I_app'] = odeint(dALLdt, X0, t, args=(I_app,))

# Plot the results to replicate Figure 2.13
plt.figure(figsize=(12, 6))

# Subplot for part A
plt.subplot(1, 2, 1)
for V_initial, sol in solutions.items():
    if V_initial != 'I_app': # Exclude the I_app case for the first plot
        plt.plot(t, sol[:, 0], label=f'V(0)={V_initial} mV')

plt.title('Figure 2.13A')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.legend()

# Subplot for part B
plt.subplot(1, 2, 2)
plt.plot(t, solutions['I_app'][:, 0], label=f'I_app = {I_app} uA/cm^2')
plt.title('Figure 2.13B')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.legend()

plt.tight_layout()
plt.show()
