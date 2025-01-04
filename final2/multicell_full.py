#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multicell_full.py

Builds a PDE-like chain of N cells, each using the single_cell_rhs_full from
File 2. We'll embed that code here for a self-contained script, or you can
import from single_cell_full if you prefer.

Patch clamp in one cell, observe conduction.

Author: [Your Name]
Date: [Date]
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1) We'll copy the single_cell_rhs_full approach
FARADAY= 96485.0
TEMPERATURE=310.0
GAS_CONST=8314.0
Z_CA= 2.0
CELL_VOL= 2e-12
ER_VOL= CELL_VOL*0.2
CAPACITANCE= 0.94

def safe_exp(x, cap=50.0):
    if x> cap: x= cap
    elif x< -cap: x= -cap
    return math.exp(x)

def single_cell_rhs_full(t, y, p):
    # y= [V, Ca_cyt, Ca_er]
    V, Ca_cyt, Ca_er = y

    # Kir + leak + patch clamp
    E_K= 25.7* math.log(p["K_o"]/ p["K_i"])  # simplified
    denomKir= 1.0 + safe_exp((V- E_K - p["Kir_shift"])/ p["Kir_slope"])
    I_Kir= p["g_Kir"]*(V- E_K)/ denomKir

    I_leak= p["g_leak"]*(V- p["E_leak"])

    # Patch clamp
    I_inject= 0.0
    if p["stim_start"]<= t <= p["stim_end"]:
        I_inject= p["I_patch"]

    dVdt= - (I_Kir + I_leak + I_inject)/ p["C_m"]

    # Ca flux
    J_serca= p["k_serca"]*( Ca_cyt**2)/( Ca_cyt**2+ p["Km_serca"]**2)
    J_leak= p["leak_rate_er"]*( Ca_er- Ca_cyt)
    flux_ip3r= p["g_ip3r"]*(Ca_er- Ca_cyt)
    flux_ryr = p["g_ryr"] *(Ca_er- Ca_cyt)
    J_release= flux_ip3r+ flux_ryr

    dCa_cyt= J_leak + J_release - J_serca
    dCa_er = - (J_release - J_leak)

    return [dVdt, dCa_cyt, dCa_er]

# 2) PDE-like
def multicell_rhs(t, Y, p):
    """
    We have N cells, each cell has 3 states => total 3*N
    We'll store them as: cell0: [V0, Ca0, CaER0], cell1: [V1, Ca1, CaER1], ...
    Gap junction couples only the membrane potentials V_i
    """
    N = p["N_cells"]
    nvar= 3
    g_gap= p["g_gap"]
    dx= p["dx"]
    cm= p["C_m"]

    dYdt= np.zeros_like(Y)
    for i in range(N):
        idx0= i*nvar
        V_i= Y[idx0]
        Ca_i= Y[idx0+1]
        Ca_er_i= Y[idx0+2]

        # single cell derivative
        sc_deriv= single_cell_rhs_full(t, [V_i, Ca_i, Ca_er_i], p)
        dV_i, dCa_i, dCa_er_i = sc_deriv

        # gap junction on V_i
        if i>0:
            V_left= Y[(i-1)* nvar]
        else:
            V_left= V_i
        if i< N-1:
            V_right= Y[(i+1)* nvar]
        else:
            V_right= V_i

        # discrete 2nd difference
        I_gj= g_gap* ((V_left + V_right - 2.0* V_i)/(dx**2))
        dV_i += (I_gj/cm)

        # fill in
        idx1= idx0+ nvar
        dYdt[idx0]= dV_i
        dYdt[idx0+1]= dCa_i
        dYdt[idx0+2]= dCa_er_i

    return dYdt

def run_multicell_sim(p):
    N= p["N_cells"]
    nvar= 3
    Y0= []
    for i in range(N):
        # each cell starts ~ -70 mV, 1e-4 cyt Ca, 0.5 ER Ca
        Y0 += [-70.0, 1e-4, 0.5]

    t_span= (0.0, p["t_max"])
    t_eval= np.linspace(0, p["t_max"], 1001)

    def rhs_wrap(t, Y):
        return multicell_rhs(t, Y, p)

    sol= solve_ivp(rhs_wrap, t_span, Y0,
                   t_eval=t_eval, method='RK45',
                   rtol=1e-6, atol=1e-9)
    return sol

def plot_multicell(sol, p):
    t= sol.t
    Y= sol.y
    N= p["N_cells"]
    nvar= 3

    plt.figure(figsize=(8,6))
    for i in range(N):
        idxV= i*nvar
        V_i= Y[idxV,:]
        plt.plot(t, V_i, label=f"Cell {i}")
    plt.title(f"Multi-cell PDE: V (Stim from {p['stim_start']} to {p['stim_end']})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # optionally plot Ca_i for each cell
    plt.figure(figsize=(8,6))
    for i in range(N):
        Ca_i= Y[i*nvar+1,:]
        plt.plot(t, Ca_i, label=f"Cell {i} Ca")
    plt.title("Multi-Cell PDE: Ca_i")
    plt.xlabel("Time (ms)")
    plt.ylabel("Ca_i (mM)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # define parameters
    p= {
        "K_o": 5.4, "K_i":140.0,
        "g_Kir":0.1, "Kir_shift":15.0, "Kir_slope":6.0,
        "g_leak":0.05, "E_leak":-70.0,
        "k_serca":0.1, "Km_serca":0.5,
        "leak_rate_er":0.01, "g_ip3r":0.05, "g_ryr":0.01,

        "I_patch": -20.0,
        "stim_start":100.0,
        "stim_end":400.0,

        "C_m":0.94,
        "t_max":300.0,

        "N_cells":5,
        "g_gap":0.02,
        "dx":1.0
    }

    sol= run_multicell_sim(p)
    plot_multicell(sol, p)
