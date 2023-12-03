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
    plt.ylim(-0.5, 0.9)
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
    

    

Ibg_init_val = 0.7*0.94 

#python optimization_problem_bayesian_2.py
#Best parameters: [5.663117485986784, 0.48832107909365974, 9.107468635253378, 0.09, 6.813249673453096] 
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=5.663117485986784, Ibg_init=Ibg_init_val, Ikir_coef=0.48832107909365974, cm=9.107468635253378, dx=0.09, K_o=6.813249673453096)
 
 
 
#python optimization_problem_random_forrest2.py
 
#python optimization_problem_xgboost2.py
#[29.512762154105392, 0.5986815284200908, 10.090817786403688, 0.07152346229938092, 6.412888685179357]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=29.512762154105392, Ibg_init=Ibg_init_val, Ikir_coef=0.5986815284200908, cm=10.090817786403688, dx=0.07152346229938092, K_o=6.412888685179357)



#python optimization_problem_nn_pytorch2.py
#Optimal parameters are: [22.3397405   0.96957738  8.57933038  0.06315757  6.32059941]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=22.3397405, Ibg_init=Ibg_init_val, Ikir_coef=0.96957738, cm=8.57933038, dx=0.063157572, K_o=6.32059941)


#Evolution completed!
#Best individual is:  [14.740220873667019, 0.8596520488469712, 9.103441742596651, 0.05355931595905949, 5.780039772718714]
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=14.740220873667019, Ibg_init=Ibg_init_val, Ikir_coef=0.8596520488469712, cm=9.103441742596651, dx=0.05355931595905949, K_o=5.780039772718714)
