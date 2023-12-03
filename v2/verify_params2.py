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
    #D = np.abs(A[99998, 98:135])[0] / np.abs(A[399998, 98:135] - A[99998, 98:135])
    D = np.abs(A[399998, 98:135] - A[99000, 98:135]) / np.abs(A[99000, 98:135])[0]


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
#[8.793390327350657, 0.44055947309344856, 0.36563980948379843, 9.134128054090077, 0.031411672744577845, 6.544541605346618]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=8.793390327350657, Ibg_init=0.44055947309344856, Ikir_coef=0.36563980948379843, cm=9.134128054090077, dx=0.031411672744577845, K_o=6.544541605346618)

#python optimization_problem_xgboost2.py
#Best parameters: [24.256766315116298, 0.7776899906496383, 0.66856005089868, 10.157542602233152, 0.0546219384433932, 6.4778802634092]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=24.256766315116298, Ibg_init=0.7776899906496383, Ikir_coef=0.66856005089868, cm=10.157542602233152, dx=0.0546219384433932, K_o=6.4778802634092)


#python optimization_problem_nn_pytorch2.py
#[13.64531954  0.15540899  0.77110982  9.75910406  0.0859075   3.92747039]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=13.64531954, Ibg_init=0.15540899, Ikir_coef=0.77110982, cm=9.75910406, dx=0.0859075, K_o=3.92747039)



#Evolution completed!
#Best individual is:  [13.400292101196492, 1.1102014918378211, 0.9370462188563706, 10.794802380985196, 0.039247762132637205, 6.484883589879831]  with fitness:  (2.668165895722002e-06,)
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=13.400292101196492, Ibg_init=1.1102014918378211, Ikir_coef=0.9370462188563706, cm=10.794802380985196, dx=0.039247762132637205, K_o=6.484883589879831)

