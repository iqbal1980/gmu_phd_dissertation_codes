import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
#Random number not needed really
random_number = 1#random.random()

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.0, Ibg_init=0.0, Ikir_coef=0.94, cm=9.4, dx=0.06, K_o=5):
    max_val = 0.51
    min_val = 0.5
    images = []

    for counter in np.arange(min_val, max_val, 0.01):
        ggapval = counter * ggap
        print(f"ggapval={ggapval}")
        A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, cm, dx, K_o)
        x = A[:, 0]
        y = A[:, 98:135]

        # Plot
        plt.figure()
        plt.plot(x, y, linewidth=3)
        plt.title(f"G gap = {ggapval}")
        images.append(f"Image{ggapval}.png")
        plt.savefig(f"Image{ggapval}.png")
        print("saved 1")
        
        plot_data2_modified(A)
        
        print("saved 2")

    return images

@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o):
    dt=0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

    I_bg = np.zeros(Ng) + Ibg_init
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

        for kk in range(Ng):
            E_K = (R * 293 / F) * np.log(K_o/150)

            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))

            # ... Rest of the function body remains largely unchanged ...
            if kk == 0:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                Vm[kk] += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 98:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 99:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 100:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]            

        A[j, 0] = t
        A[j, 1:] = Vm

    return A


def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]

    distance_m = dx * np.arange(99, 136)
    plt.figure()
    plt.plot(distance_m, D, '.', markersize=8)
    c = np.polyfit(distance_m, D, 1)
    y_est = np.polyval(c, distance_m)
    plt.plot(distance_m, y_est, 'r--', linewidth=2)
    plt.savefig(f"Image2.png")

 

#Bayesian Optimization
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=22.823350409810097, Ibg_init=1.4137515444140243, Ikir_coef=0.8242785834941551, cm=9.043352128462157, dx=0.050635413384145275, K_o=3.327389680049543)

#XGBoost
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.9651001054207696, Ibg_init=0.5064782981616258, Ikir_coef=0.9214774072209153, cm=9.27411333561133, dx=0.06705029426722967, K_o=3.2849319288716146)

#Random forrests
#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.11367203272084944, Ibg_init=0.5538705683100204, Ikir_coef=0.938252665035215, cm=8.925352662805283, dx=0.06013602000111776, K_o=4.947632840951089)

#NN
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=11.13429464, Ibg_init=1.00481258, Ikir_coef=0.57666017, cm=9.04141524, dx=0.05815578, K_o=3.05022504)


