import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#Random number not needed really
random_number = 1#random.random()

global_dx=1 #We are using a network model so there is no spacial discretisation

  


MIN_VALUE = -10000  # Updated example constant for safe_log
MAX_VALUE = 1e150   # Updated example constant for safe_exponential

@njit
def safe_log(x):
    if x <= 0:
        # Return a constant value for non-positive inputs
        return MIN_VALUE
    return np.log(x)



@njit
def safe_exponential(x, n):
    # Check for potential overflow/underflow conditions
    if x > 700 or x < -700:
        # Return a constant value to avoid overflow/underflow
        return MAX_VALUE
    if n > 100:  # Adjust this threshold as needed
        # Large n can lead to overflow even if x is moderate
        return MAX_VALUE

    return np.power(np.exp(x), n)

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o):
    max_val = 0.51
    min_val = 0.5
    images = []


    ggapval = ggap
    print(f"ggapval={ggapval}")
    A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, cm, dx, K_o)
    x = A[:, 0]
    y = A[:, 101:135]

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
            #E_K = (R * 293 / F) * np.log(K_o/150)
            E_K = (R * 293 / F) * safe_log(K_o/150)
            

            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            #I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + safe_exponential((Vm[kk] - E_K - 25) / 7, 1)))         

            
            # ... Rest of the function body remains largely unchanged ...
            if kk == 0:
                Vm[kk] += random_number * 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
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


from scipy.stats import linregress

def plot_data2_modified(A, ggap, withReference=False):  
    cellLength = 60 #in Microns
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / np.abs(A[99000, 101:135])[0]
    distance_m = cellLength * np.arange(102-102, 136-102)
    
    print(A[399998, 101:135])
    print(A[99000, 101:135])
    print(D)
    print(distance_m)
    
    
    # Fit the exponential function to the data
    popt, pcov = curve_fit(safe_exponential, distance_m, D, maxfev=1000)
    
    # Calculate R^2
    residuals = D - safe_exponential(distance_m, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((D - np.mean(D))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    plt.figure()
    plt.ylim(-0.5, 12)
    plt.plot(distance_m, D, '.', markersize=8)
    
    # Plot the fitted exponential curve
    x_fit = np.linspace(min(distance_m), max(distance_m), 100)
    y_fit = safe_exponential(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r--', linewidth=2)
    
    if withReference:
        # Plot the reference line
        y_ref = np.exp(-0.003 * distance_m)
        plt.plot(distance_m, y_ref, 'b-', linewidth=2)
        # Add custom legend entries
        plt.legend(['Data', f'Fit: y = e^({popt[0]:.3f}x), $R^2$ = {r_squared:.4f}', 
                    'Reference: y = e^(-0.003x)'], loc='upper left')
    
    plt.title('Chart Title')
    plt.savefig(f"Image2_{ggap}{'_ref' if withReference else ''}.png")
    
    return f"Image2_{ggap}{'_ref' if withReference else ''}.png"


    

    

Ibg_init_val = 0.7*0.94 

 


#Evolution completed!
#Best individual is:  [0.3015830801507125, 0.94]  with fitness:  (0.00573706841604791,)
HK_deltas_vstim_vresponse_graph_modified_v2(ggap=0.3015830801507125, Ibg_init=Ibg_init_val, Ikir_coef=0.94, cm=9.4, dx=global_dx, K_o=3)




