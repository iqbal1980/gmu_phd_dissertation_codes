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


    ggapval = ggap
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
        
    plot_data2_modified(A,ggap,False)
    plot_data2_modified(A,ggap,True)
 
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


def plot_data2_modified(A,ggap,withReference=False):  
    dx = 0.06
    #D = np.abs(A[399998, 98:135] - A[99998, 98:135]) / np.abs(A[99998, 98:135])[0]
    D = np.abs(A[99998, 98:135])[0] / np.abs(A[399998, 98:135] - A[99998, 98:135])

    distance_m = dx * np.arange(99, 136)
    plt.figure()
    plt.plot(distance_m, D, '.', markersize=8)
    
    # Your existing line
    c = np.polyfit(distance_m, D, 1)
    y_est = np.polyval(c, distance_m)
    plt.plot(distance_m, y_est, 'r--', linewidth=2)
    
    ref=""
    if withReference == True:
        # New line for y = 0.5x - 0.01
        #Reference equation from experimental data
        y_new = -0.0000000000001 * distance_m + 0.6
        plt.plot(distance_m, y_new, 'b-', linewidth=2, label='y = -0.0000000000001 + 0.6 reference line')
        plt.legend()  # Add this to show the legend for the lines
        ref = "ref"
    
    plt.savefig(f"Image2_{ggap}{ref}.png")
    

    

 

#python optimization_problem_bayesian_2.py
#Best parameters: [2.092320984437425, 0.5931913860906698, 0.47609881403078547, 9.936789495154567, 0.03316747334594141, 6.692255212880238]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=2.092320984437425, Ibg_init=0.5931913860906698, Ikir_coef=0.47609881403078547, cm=9.936789495154567, dx=0.03316747334594141, K_o=6.692255212880238)
 
 
#python optimization_problem_random_forrest2.py
#Best parameters: [32.1625607616559, 0.13296341062974382, 0.7628400772372818, 10.177001933628631, 0.05722130082505892, 3.5635767953764033]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=32.1625607616559, Ibg_init=0.13296341062974382, Ikir_coef=0.7628400772372818, cm=10.177001933628631, dx=0.05722130082505892, K_o=3.5635767953764033)

#python optimization_problem_xgboost2.py
#Best parameters: [4.394206805886486, 0.11299456111492417, 0.9749393022450061, 8.688225714077948, 0.04717194922428382, 3.343099923926932]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=4.394206805886486, Ibg_init=0.11299456111492417, Ikir_coef=0.9749393022450061, cm=8.688225714077948, dx=0.04717194922428382, K_o=3.343099923926932)


#python optimization_problem_nn_pytorch2.py
#[21.14202366  1.08657416  0.44870776  8.81294067  0.03774739  2.28285353]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=21.14202366, Ibg_init=1.08657416, Ikir_coef=0.44870776, cm=8.81294067, dx=0.03774739, K_o=2.28285353)
#[19.3014593   0.62623789  0.59692329  8.52092586  0.05107232  2.85419173]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=19.3014593, Ibg_init=0.62623789, Ikir_coef=0.59692329, cm=8.52092586, dx=0.05107232, K_o=2.85419173)

#Evolution completed!
#Best individual is:  [22.39622634489695, 0.1, 1.2, 9.743527475837842, 0.05406888485755828, 3.0556291008205645]  with fitness:  (5.242795248020311e-07,)
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=22.39622634489695, Ibg_init=0.1, Ikir_coef=1.2, cm=9.743527475837842, dx=0.05406888485755828, K_o=3.0556291008205645)
