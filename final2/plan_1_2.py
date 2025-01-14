#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
next_steps_minimal_pericyte_full.py

A minimal PDE-like approach for pericyte cells with Kir + leak currents.
Includes:
    1) single_cell_rhs (Kir + leak)
    2) multicell_rhs for PDE-like coupling
    3) run_sim for solving PDE over time
    4) A small demonstration in __main__ that:
       - sets parameters
       - runs a simulation
       - plots results

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# Single-Cell Model: Kir + Leak (Minimal)
###############################################################################
def single_cell_rhs(t, y, p, cell_index):
    """
    Returns dy/dt for a single pericyte with Kir and a background leak.

    y[0] = V (membrane potential, in mV)
    """

    V = y[0]

    # Unpack parameters
    gKir    = p['gKir']      # Kir conductance (nS)
    EK      = p['EK']        # Reversal potential for K  (mV)
    gLeak   = p['gLeak']     # Leak conductance (nS)
    ELeak   = p['ELeak']     # Reversal potential for leak (mV)
    Cm      = p['Cm']        # Membrane capacitance (pF)
    Vhalf   = p['Kir_Vhalf'] # half-activation shift for Kir
    k_slope = p['Kir_slope'] # slope factor for Kir activation

    # iKir (in pA)
    iKir = gKir * (V - EK) / (1.0 + np.exp((V - EK - Vhalf)/k_slope))

    # iLeak (in pA)
    iLeak = gLeak * (V - ELeak)

    # External patch clamp current if we are injecting
    I_app = 0.0
    if (cell_index == p['stim_cell']) and (p['stim_start'] <= t <= p['stim_end']):
        I_app = p['I_app_val']

    # dV/dt in mV/ms:
    # Note: iKir + iLeak + I_app are in pA; 1 pA = 1e-12 A
    #       Cm is in pF; 1 pF = 1e-12 F
    # => dV/dt = - ( sum of currents ) / Cm  [units check out, in mV/ms]
    dVdt = - ( iKir + iLeak + I_app ) / Cm

    return [dVdt]


###############################################################################
# PDE-like approach
###############################################################################
def multicell_rhs(t, Y, p):
    """
    PDE-like system for N pericytes, each with (1) state: V.
    Gap junction coupling on V is added to single cell RHS.

    Y is an array of length N, representing V for each cell.
    """
    N     = p['N_cells']
    Cm    = p['Cm']
    g_gap = p['g_gap']
    dx    = p['dx']  # spacing
    nvar  = 1

    dYdt = np.zeros_like(Y)

    for i in range(N):
        idx = i*nvar
        y_i = Y[idx : idx + nvar]

        # Single-cell component
        dy_i = single_cell_rhs(t, y_i, p, cell_index=i)
        V_i  = y_i[0]

        # Gap-junction conduction:
        if i==0:
            V_left = V_i
        else:
            V_left = Y[(i-1)*nvar]

        if i==N-1:
            V_right = V_i
        else:
            V_right = Y[(i+1)*nvar]

        # gap current = g_gap * (neighbor potentials - 2*Vi)/(dx^2)
        # add that as dV/dt_contrib = I_gj / Cm
        I_gj = g_gap * ((V_left + V_right - 2.0*V_i)/(dx*dx))
        dVdt_gj = I_gj / Cm

        dYdt[idx] = dy_i[0] + dVdt_gj

    return dYdt


###############################################################################
# Simulation wrapper
###############################################################################
def run_sim(p):
    """
    Sets up the PDE system, calls solve_ivp, returns solution object.
    """
    N    = p['N_cells']
    nvar = 1

    # Build initial condition
    Y0 = np.zeros(N*nvar)
    for i in range(N):
        # set each cell's initial V
        Y0[i] = p['Vinit']

    # Time span
    t_span  = (0.0, p['t_final'])
    t_eval  = np.linspace(0.0, p['t_final'], p['num_points'])

    def rhs_wrap(t, Y):
        return multicell_rhs(t, Y, p)

    sol = solve_ivp(rhs_wrap, t_span, Y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-9)
    return sol


###############################################################################
# MAIN DEMO
###############################################################################
if __name__ == "__main__":

    # Example parameter dictionary
    p = {
        # Kir
        'gKir'     : 0.2,      # nS
        'EK'       : -80.0,    # mV
        'Kir_Vhalf': 25.0,     # mV offset
        'Kir_slope': 7.0,      # slope factor

        # Leak
        'gLeak'    : 0.05,     # nS
        'ELeak'    : -40.0,    # mV

        # Membrane
        'Cm'       : 10.0,     # pF

        # PDE / multicell
        'N_cells'  : 5,
        'g_gap'    : 0.02,     # nS
        'dx'       : 1.0,      # arbitrary spacing in "um" or so

        # Stim
        'stim_cell': 0,        # index of cell we inject
        'I_app_val': 50.0,     # pA, injection
        'stim_start': 20.0,    # ms
        'stim_end'  : 80.0,    # ms

        # initial
        'Vinit'    : -50.0,    # mV

        # time
        't_final'  : 200.0,    # ms
        'num_points': 201
    }

    # Run once
    sol = run_sim(p)

    # Extract
    t = sol.t
    Vall = sol.y  # shape = (N_cells, len(t)) but watch array shape

    # We built Y in 1D, so if N=5, shape is (5, #points).
    # Typically, sol.y dimension is (#states, #time_points).
    # N=5 => 5 rows, each row is that cell's trajectory in time.
    # But watch out that each row is sol.y[i,:], but i might be 0..4

    # Plot time courses for each cell
    plt.figure(figsize=(8,5))
    for i in range(p['N_cells']):
        plt.plot(t, Vall[i,:], label=f"Cell {i}")

    plt.title("Voltage vs Time (Kir + Leak, PDE-like Coupling)")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally, final snapshot
    V_final = Vall[:, -1]
    plt.figure()
    plt.plot(range(p['N_cells']), V_final, 'o-', label="Final V")
    plt.xlabel("Cell index")
    plt.ylabel("V (mV)")
    plt.title("Final voltage distribution across the array")
    plt.legend()
    plt.show()
