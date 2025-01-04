#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
single_cell_full.py

Adds:
- Ca2+ handling (SERCA, leak)
- IP3R, RyR
- Possibly some partial Kir, leak to keep ~-70 mV
- Patch clamp from t=100..400 ms

Author: [Your Name]
Date: [Date]
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1) Global / Helper
FARADAY = 96485.0
TEMPERATURE = 310.0
GAS_CONST = 8314.0
Z_CA = 2.0
CELL_VOL = 2e-12  # L
ER_VOL   = CELL_VOL * 0.2
CAPACITANCE = 0.94  # pF

def safe_exp(x, cap=50.0):
    if x> cap: x= cap
    if x< -cap: x= -cap
    return math.exp(x)

# --- 2) Model Parameters (example) ---
params = {
    # Ion concentrations
    "K_o": 5.4, "K_i": 140.0,
    "Ca_o":2.0, "Ca_i_init":1e-4,  # initial cytosol Ca
    "Na_in":15.0, "Na_out":140.0,

    # Currents
    "g_Kir": 0.1,
    "Kir_shift":15.0,
    "Kir_slope":6.0,
    "g_leak":0.05,
    "E_leak": -70.0,

    # Ca flux
    "k_serca":0.1,
    "Km_serca":0.5,
    "leak_rate_er":0.01,
    "g_ip3r":0.05,
    "g_ryr": 0.01,

    # Patch clamp
    "I_patch": -20.0,
    "stim_start":100.0,
    "stim_end":400.0,

    # Mem. / geometry
    "C_m": CAPACITANCE,
    # Times
    "t_max":600.0,
}

# --- 3) Single-Cell State Vector
# y = [V, Ca_cyt, Ca_er, IP3, ... you can add gating if needed]
# For demonstration: minimal approach with Ca
def single_cell_rhs_full(t, y, p):
    """
    y = [V, Ca_cyt, Ca_er]
    We'll keep it simpler than your full 23-state version, but illustrate.
    """
    V, Ca_cyt, Ca_er = y

    # 3.1) Kir current
    E_K = (25.7)*math.log(p["K_o"]/p["K_i"])  # ~ RT/F at 35C
    shift= p["Kir_shift"]
    slope= p["Kir_slope"]
    denomKir= 1.0 + safe_exp((V - E_K - shift)/ slope)
    I_Kir = p["g_Kir"]*(V- E_K)/ denomKir

    # 3.2) leak
    I_leak= p["g_leak"]*(V - p["E_leak"])

    # 3.3) patch clamp
    I_inject= 0.0
    if p["stim_start"] <= t <= p["stim_end"]:
        I_inject = p["I_patch"]

    # 3.4) Mem eqn
    dVdt = - (I_Kir + I_leak + I_inject)/ p["C_m"]

    # 3.5) Ca flux
    # SERCA
    J_serca= p["k_serca"]*( Ca_cyt**2)/( Ca_cyt**2+ p["Km_serca"]**2)
    # ER leak
    J_leak= p["leak_rate_er"]*( Ca_er - Ca_cyt)
    # IP3R or RyR (just a simplified constant flux if Ca_er> Ca_cyt)
    # Real model would do gating, but let's do a small flux to show concept
    flux_ip3r= p["g_ip3r"]*(Ca_er- Ca_cyt)
    flux_ryr = p["g_ryr"]*(Ca_er- Ca_cyt)
    J_release= flux_ip3r + flux_ryr

    # Cytosolic Ca eqn
    # We'll convert I_Kir etc. to Ca if we want, but let's keep it simpler
    dCa_cyt= J_leak + J_release - J_serca
    # ER Ca eqn
    dCa_er = - (J_release - J_leak)

    return [dVdt, dCa_cyt, dCa_er]

def run_simulation_full(p):
    t_span = (0.0, p["t_max"])
    t_eval = np.linspace(0, p["t_max"], 2001)
    # initial conditions
    V0= -70.0
    Ca_cyt0= p["Ca_i_init"]
    Ca_er0= 0.5  # example

    y0= [V0, Ca_cyt0, Ca_er0]

    def rhs_wrapper(t, Y):
        return single_cell_rhs_full(t, Y, p)

    sol= solve_ivp(rhs_wrapper, t_span, y0, t_eval=t_eval,
                   method='RK45', rtol=1e-6, atol=1e-9)
    return sol

def plot_results_full(sol):
    t= sol.t
    V= sol.y[0,:]
    Ca_i= sol.y[1,:]
    Ca_er= sol.y[2,:]

    plt.figure(figsize=(9,8))

    plt.subplot(3,1,1)
    plt.plot(t, V, 'b')
    plt.title("Single-Cell Full-ish Model")
    plt.ylabel("Vm (mV)")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, Ca_i, 'r', label="Cytosol Ca")
    plt.plot(t, Ca_er,'g', label="ER Ca")
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    # If you had IP3 or Mito Ca, you'd plot them. We'll just re-plot Ca_i for example
    plt.plot(t, Ca_i, 'r', label="Ca_i again")
    plt.xlabel("Time (ms)")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    sol= run_simulation_full(params)
    plot_results_full(sol)
