#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multicell_pericyte_minimal.py

A minimal PDE-like model for 'pericytes' using:
  - Membrane potential (V)
  - Kir2.1 current (I_kir)
  - Simple background / leak current (I_bg)
  - Gap-junction coupling between cells
  - Optional patch-clamp injection current I_app in one cell

Author: [Your Name]
Date: [Date]

Usage:
  python multicell_pericyte_minimal.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# For clarity, define an index for the membrane potential
STATE_V = 0

###############################################################################
# SINGLE-CELL ODE: Minimal pericyte
# This function returns dV/dt given the cell's parameters and potential.
###############################################################################
def single_cell_rhs_minimal(t, y_cell, p, cell_index):
    """
    Minimal single-cell ODE for a pericyte with only:
      - Kir current
      - A background (leak) current
      - Optional patch-clamp I_app

    Parameters
    ----------
    t : float
        current time (ms)
    y_cell : array
        the state variables for *one* cell, here just [V] in mV
    p : dict
        parameter dictionary
    cell_index : int
        index of the cell (to see if it's the stimulated cell, etc.)

    Returns
    -------
    dy/dt : array of size = number of states (1 here)
    """
    # Unpack
    V = y_cell[STATE_V]  # membrane potential (mV)
    
    # (A) Kir current
    # For a real pericyte Kir2.1: GKir ~ 0.2–0.5 nS, depends on scRNA/patch data
    # Here: I_kir = GKir * sqrt(K_out) * ((V - E_K) / [1 + exp((V - E_K - deltaV)/k_kir)])
    # Example numeric values:
    GKir   = p['GKir']              # nS/mM^0.5
    Ko     = p['K_o']               # external K+ (mM)
    E_K    = p['E_K']               # typical ~ -80 to -40 mV, or RT/F log(Ko/Ki)
    deltaV = p['deltaV_kir']        # about 25 mV for half-block
    k_kir  = p['k_kir']             # slope
    n_kir  = 0.5                    # exponent for sqrt(Ko)
    
    # We often want sqrt(Ko) to scale the Kir conductance
    # Then multiply by the sigmoidal gating factor
    denom_kir = (1.0 + math.exp((V - E_K - deltaV)/k_kir))
    if denom_kir < 1e-9:
        denom_kir = 1e-9
    I_kir = GKir * (Ko**n_kir) * ( (V - E_K) / denom_kir )  # in pA

    # (B) Background/Leak current
    # I_bg = G_bg*(V - E_bg) in pA, where G_bg in nS
    # E_bg ~ -30 mV for a typical background or the so-called "resting" potential
    G_bg  = p['G_bg']   # nS
    E_bg  = p['E_bg']   # mV
    I_bg  = G_bg*(V - E_bg)  # pA

    # (C) Optional patch clamp
    I_app = 0.0
    if (cell_index == p['stim_cell']) and (p['stim_start'] <= t <= p['stim_end']):
        I_app = p['I_app_val']  # pA (positive or negative)

    # (D) Sum
    I_sum = I_kir + I_bg + I_app  # pA

    # (E) dV/dt
    # dV/dt = -I_sum / Cm, with I in pA, Cm in pF
    Cm = p['Cm']  # pF
    dVdt = - (I_sum) / Cm

    return np.array([dVdt])

###############################################################################
# PDE-LIKE APPROACH
# If we want N cells in a line, each has V, with gap junction coupling:
#    I_gj ~ g_gap * (V_neighbor - V_i), or a second-difference approach.
###############################################################################
def multicell_rhs(t, Y, p):
    """
    PDE-like approach for N cells, each with a single state V.
    We do a gap-junction or diffusive coupling.

    Y : 1D array of length N (since each cell has 1 state).
    p : dictionary of parameters.

    Returns
    -------
    dYdt : same shape as Y
    """
    N = p['N_cells']
    dYdt = np.zeros(N, dtype=float)

    # For each cell
    for i in range(N):
        # Single-cell ODE
        y_i = np.array([Y[i]])  # shape(1,) for the single state
        dy_i = single_cell_rhs_minimal(t, y_i, p, cell_index=i)

        # Add gap-junction term: 
        # Usually: I_gj = g_gap*(V_left + V_right - 2*V_i), or simpler for 1D chain
        V_i = Y[i]
        # left neighbor
        if i == 0:
            V_left = V_i
        else:
            V_left = Y[i-1]
        # right neighbor
        if i == N-1:
            V_right = V_i
        else:
            V_right = Y[i+1]

        # The net gap-junction current from left+right
        # Let g_gap be in nS, then current in pA, 
        # dV/dt_gj = (I_gj / Cm). 
        I_gj = p['g_gap']*((V_left + V_right - 2.0*V_i))

        # Convert from pA to dV/dt
        dVdt_gj = I_gj / p['Cm']  # pA / pF = mV/ms (assuming 1 pF => 1 mV/ms per pA)
        
        # Sum
        dYdt[i] = dy_i[0] + dVdt_gj
    
    return dYdt

###############################################################################
# RUN SIM
###############################################################################
def run_multicell_sim(p):
    # Build initial condition: all cells start at -40 mV, e.g.
    N = p['N_cells']
    Y0 = np.full(N, p['V_init'], dtype=float)

    t_span = (0.0, p['t_final'])
    t_eval = np.linspace(t_span[0], t_span[1], 601)  # about 600 points

    def rhs_wrapper(t, y):
        return multicell_rhs(t, y, p)

    sol = solve_ivp(rhs_wrapper, t_span, Y0,
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-6, atol=1e-9)
    return sol

###############################################################################
# PLOT
###############################################################################
def plot_results(sol, p, label=""):
    t = sol.t
    Y = sol.y  # shape (N, len(t))
    N = p['N_cells']

    # Plot voltages
    plt.figure(figsize=(10,6))
    for i in range(N):
        plt.plot(t, Y[i,:], label=f"Cell {i}")
    plt.title(f"Minimal PDE-like Pericyte Model (N={N}), {label}")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

###############################################################################
# MAIN
###############################################################################
if __name__=="__main__":
    # Example parameter set
    p = {
        # Basic single-cell Kir
        'GKir': 0.3,       # nS/mM^0.5, example
        'K_o': 5.4,        # mM
        'E_K': -80.0,      # mV
        'deltaV_kir': 25.0,  # mV
        'k_kir': 7.0,      # slope factor

        # Background current
        'G_bg': 0.1,       # nS
        'E_bg': -30.0,     # mV

        # Membrane capacitance
        'Cm': 10.0,        # pF, typical for small cell

        # Gap-junction coupling
        'g_gap': 0.2,      # nS, example

        # PDE / multi-cell
        'N_cells': 5,
        'V_init': -40.0,   # initial potential for all cells
        't_final': 600.0,  # ms

        # Patch clamp
        'stim_cell': 0,      # e.g. cell index 0 is clamped
        'I_app_val': +20.0,  # pA
        'stim_start': 100.0,
        'stim_end': 400.0
    }

    # Run
    sol = run_multicell_sim(p)

    # Plot
    plot_results(sol, p, label=f"I_app={p['I_app_val']} pA on cell {p['stim_cell']}")
