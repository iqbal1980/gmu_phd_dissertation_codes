import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
#Random number not needed really
random_number = 1#random.random()

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.0, Ibg_init=0.0, Ikir_coef=0.94, dt=0.001, 
                                                cm=9.4, a=0.01, dx=0.06, F=9.6485e4, R=8.314e3, K_o=5):
    max_val = 0.51
    min_val = 0.5
    images = []

    for counter in np.arange(min_val, max_val, 0.01):
        ggapval = counter * ggap
        print(f"ggapval={ggapval}")
        A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
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
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o):
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

# Running the main function
#images_generated_v2 = HK_deltas_vstim_vresponse_graph_modified_v2()
#[8.84510392e-01 8.09027238e-01 9.54874355e-01 3.66613713e-03
# 8.11326638e+00 2.08684071e-02 6.20330263e-02 9.50275294e+04
# 8.39921763e+03 5.41599188e+00]
#Minimum loss is: -45.21070098876953

#HK_deltas_vstim_vresponse_graph_modified_v2(ggap=8.84510392e-01, Ibg_init=8.09027238e-01, Ikir_coef=9.54874355e-01, dt=3.66613713e-03, cm=8.11326638e+00, a=2.08684071e-02, dx=6.20330263e-02, F=9.50275294e+04, R=8.39921763e+03, K_o=5.41599188e+00)



HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.7063396809126381, Ibg_init=0.059861218165099164, Ikir_coef=0.8274999913672687, dt=0.0005045791269128754, cm=10.82864286519611, a=0.028176076424087322, dx=0.0690530159558452, F=99464.51479180687, R=8230.843376780089, K_o=4.009279420871547)