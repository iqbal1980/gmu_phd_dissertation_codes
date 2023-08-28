import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
import random


def HK_deltas_vstim_vresponse_graph():
    max_val = 0.5
    min_val = 0.5

    for counter in np.arange(min_val, max_val):  
        ggapval = counter * 1
        print(f"ggapval={ggapval}")
        A = simulate_process(ggapval)
        x = A[:, 0]
        y = A[:, 98:135]

        # Saving y values
        np.savetxt('y.txt', y)

        # Plot
        plt.figure()
        plt.plot(x, y, linewidth=3)
        plt.title(f"G gap = {ggapval}")
        plt.savefig(f"Image{ggapval}.png")
        print("saved 1")
        
        plot_data2(A)
        
        print("saved 1")


#@njit
#@njit(parallel=True)
#@njit(parallel=False)
@njit
def simulate_process(g_gap_value):
    #start_time = time.time()

    dt = 0.001
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    cm = 9.4

    a = 0.01
    dx = 0.06
    g_gap = g_gap_value

    F = 9.6485e4
    R = 8.314e3
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

    I_bg = np.zeros(Ng)
    I_kir = np.zeros(Ng)
    distance_m = np.zeros(Ng)
    vstims = np.zeros(Ng)
    vresps = np.zeros(Ng)

    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = 50.0
        else:
            I_app[99] = 0.0

        K_o = 5

        for kk in range(Ng):
            E_K = (R * 293 / F) * np.log(K_o/150)

            I_bg[kk] = 0.658 * (Vm[kk] + 30)
            I_kir[kk] = 0.94 * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))

            if kk == 0:
                Vm[kk] += eki1 * (Vm[kk + 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng - 1:
                Vm[kk] += eki1 * (Vm[kk - 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                Vm[kk] += eki1 * (Vm[kk + 1] + Vm[kk - 1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            distance_m[kk] = kk * dx
            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

    return A
    
def plot_data2(A):  
    dx = 0.06
    D = np.abs(A[399998, 98:135] - A[99998, 98:135]) / np.abs(A[99998, 98:135])[0]

    distance_m = dx * np.arange(99, 136)
    plt.figure()
    plt.plot(distance_m, D, '.', markersize=8)
    c = np.polyfit(distance_m, D, 1)
    y_est = np.polyval(c, distance_m)
    plt.plot(distance_m, y_est, 'r--', linewidth=2)
    plt.savefig(f"Image2.png")
    #end_time = time.time()
    #print(f"Time taken: {end_time - start_time} seconds")

    return A

# Running the main function
HK_deltas_vstim_vresponse_graph()
