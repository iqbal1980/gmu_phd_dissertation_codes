import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the Hodgkin-Huxley model parameters and initial conditions
params = {
    'i': 0,  # Applied current
    'vna': 50,  # Sodium reversal potential
    'vk': -77,  # Potassium reversal potential
    'vl': -54.4,  # Leak channel reversal potential
    'gna': 120,  # Sodium channel conductance
    'gk': 36,  # Potassium channel conductance
    'gl': 0.3,  # Leak channel conductance
    'c': 1,  # Membrane capacitance
}
initial_conditions = [-65, 0.052, 0.596, 0.317]  # [v, m, h, n]

# Hodgkin-Huxley model equations
def hh_model_with_currents(Y, t, params):
    v, m, h, n = Y
    INa = params['gna'] * m**3 * h * (v - params['vna'])
    IK = params['gk'] * n**4 * (v - params['vk'])
    IL = params['gl'] * (v - params['vl'])
    dvdt = (params['i'] - INa - IK - IL) / params['c']
    
    am = 0.1*(v+40)/(1 - np.exp(-(v+40)/10))
    bm = 4*np.exp(-(v+65)/18)
    ah = 0.07*np.exp(-(v+65)/20)
    bh = 1/(1 + np.exp(-(v+35)/10))
    an = 0.01*(v+55)/(1 - np.exp(-(v+55)/10))
    bn = 0.125*np.exp(-(v+65)/80)
    
    dmdt = am*(1 - m) - bm*m
    dhdt = ah*(1 - h) - bh*h
    dndt = an*(1 - n) - bn*n
    
    return [dvdt, dmdt, dhdt, dndt], INa, IK, IL

# Time array for the simulation, 50 milliseconds
t = np.linspace(0, 50, 1000)

# Solve the HH model and retrieve the ionic currents
def solve_hh_with_currents(initial_conditions, t, params):
    INa_values = np.zeros_like(t)
    IK_values = np.zeros_like(t)
    IL_values = np.zeros_like(t)

    def hh_ode_wrapper(Y, t, params):
        (derivatives), INa, IK, IL = hh_model_with_currents(Y, t, params)
        return derivatives

    solution = odeint(hh_ode_wrapper, initial_conditions, t, args=(params,))
    v, m, h, n = solution.T

    for i, val in enumerate(solution):
        _, INa, IK, IL = hh_model_with_currents(val, t[i], params)
        INa_values[i] = INa
        IK_values[i] = IK
        IL_values[i] = IL
    
    return v, m, h, n, INa_values, IK_values, IL_values

v, m, h, n, INa, IK, IL = solve_hh_with_currents(initial_conditions, t, params)

# Define colors for the plots
colors = {
    'v': 'blue',
    'm': 'red',
    'h': 'green',
    'n': 'purple',
    'INa': 'orange',
    'IK': 'brown',
    'IL': 'grey'
}

# Plot the results with descriptive text
fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
ax[0].plot(t, v, color=colors['v'], label='Membrane Potential V(t)')
ax[0].set_ylabel('Voltage (mV)')
ax[0].set_title('Membrane potential changes during action potential')
ax[0].legend()

ax[1].plot(t, m, color=colors['m'], label='Sodium Activation Variable m(t)')
ax[1].set_ylabel('m (Sodium Activation)')
ax[1].set_title('Sodium channel activation during action potential')
ax[1].legend()

ax[2].plot(t, h, color=colors['h'], label='Sodium Inactivation Variable h(t)')
ax[2].set_ylabel('h (Sodium Inactivation)')
ax[2].set_title('Sodium channel inactivation during action potential')
ax[2].legend()

ax[3].plot(t, n, color=colors['n'], label='Potassium Activation Variable n(t)')
ax[3].set_ylabel('n (Potassium Activation)')
ax[3].set_title('Potassium channel activation during action potential')
ax[3].legend()

ax[4].plot(t, INa, color=colors['INa'], label='Sodium Current INa(t)')
ax[4].plot(t, IK, color=colors['IK'], label='Potassium Current IK(t)')
ax[4].plot(t, IL, color=colors['IL'], label='Leak Current IL(t)')
ax[4].set_ylabel('Current (µA/cm²)')
ax[4].set_title('Ionic currents during action potential')
ax[4].legend()

ax[4].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
