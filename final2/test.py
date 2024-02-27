import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
C_m = 1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conducances, in mS/cm^2
g_K = 36.0
g_L = 0.3
E_Na = 50.0  # Nernst reversal potentials, in mV
E_K = -77.0
E_L = -54.387

# Equations describing ion channel kinetics
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

# Hodgkin-Huxley model
def hodgkin_huxley(X, t, I):
    V, m, h, n = X
    dVdt = (I - g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) - g_L * (V - E_L)) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

# Initial conditions
X0 = [-65, 0.05, 0.6, 0.32]

# Time values
t = np.linspace(0, 50, 1000)  # 50 ms in 1000 steps

# Input current
I = 10  # in uA/cm^2

# Solve ODE
X = odeint(hodgkin_huxley, X0, t, args=(I,))

# Plotting
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.title('Hodgkin-Huxley Neuron')
plt.plot(t, X[:, 0], 'k')
plt.ylabel('V (mV)')

plt.subplot(2,1,2)
plt.plot(t, [I if 10 <= tt <= 40 else 0 for tt in t], 'k')
plt.xlabel('t (ms)')
plt.ylabel('I (uA/cm^2)')

plt.show()
