#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
single_cell_minimal.py

Minimal single-cell model:
- Outward current: Kir2.1-like
- Inward "leak" with reversal at -70 mV (to hold stable resting potential)
- Patch clamp from t=100 to 400 ms

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1) Constants ---
FARADAY = 96485.0
CAPACITANCE = 10.0  # pF, example
R_T_OVER_F = 25.7   # mV at ~ room temp or 26.7 at 37C; pick approximate

# --- 2) Model Parameters ---
params = {
    # Kir channel
    "g_Kir": 0.5,        # nS
    "Ko": 5.4,           # external K+ (mM)
    "Ki": 140.0,         # internal K+ (mM)
    "shift_kir": 15.0,   # shift
    "slope_kir": 6.0,    # slope

    # "Leak" with E_leak = -70 mV
    "g_leak": 0.2,       # nS
    "E_leak": -70.0,     # mV

    # Patch clamp
    "I_patch": -20.0,    # pA (negative => hyperpolarizing)
    "stim_start": 100.0,
    "stim_end":   400.0,

    # Single-cell
    "C_m": CAPACITANCE,  # pF
    # initial voltage
    "V_init": -70.0
}

# --- 3) Single-Cell ODE ---
def single_cell_rhs(t, V, p):
    """
    V: membrane potential (mV)
    dV/dt = - (I_kir + I_leak + I_patch_if_any) / C_m
    Returns scalar dV/dt
    """
    # Unpack
    gK   = p["g_Kir"]
    Ko   = p["Ko"]
    Ki   = p["Ki"]
    shift= p["shift_kir"]
    slope= p["slope_kir"]
    gleak= p["g_leak"]
    Eleak= p["E_leak"]
    Cm   = p["C_m"]

    # Reversal potential for K+ (mV)
    # approximate Nernst => RT/F ~ 25.7 or 26.7
    E_K  = R_T_OVER_F * np.log(Ko / Ki)

    # Kir current
    # I_kir = gK * (V - E_K) / [1 + exp((V - E_K - shift)/ slope)]
    denom = 1.0 + np.exp((V - E_K - shift)/ slope)
    I_kir = gK * (V - E_K)/ denom

    # Leak
    I_leak= gleak * (V - Eleak)

    # Patch clamp
    I_inject = 0.0
    if p["stim_start"] <= t <= p["stim_end"]:
        I_inject = p["I_patch"]  # pA

    # Membrane eqn
    dVdt = - (I_kir + I_leak + I_inject) / Cm  # (pA / pF) => mV/ms
    return dVdt

def run_simulation(p):
    """
    Integrate from t=0..600 ms, single variable: V(t).
    """
    t_span = (0.0, 600.0)
    t_eval = np.linspace(0, 600.0, 2001)

    def rhs_wrapper(t, Varr):
        V = Varr[0]
        dVdt = single_cell_rhs(t, V, p)
        return [dVdt]

    y0 = [p["V_init"]]  # initial voltage
    sol = solve_ivp(rhs_wrapper, t_span, y0,
                    t_eval=t_eval, method='RK45',
                    rtol=1e-6, atol=1e-9)

    return sol

def plot_results(sol):
    t = sol.t
    V = sol.y[0]

    plt.figure(figsize=(8,5))
    plt.plot(t, V, 'b', label="Vm")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Minimal Single-Cell Model (Kir + Leak)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    sol = run_simulation(params)
    plot_results(sol)
