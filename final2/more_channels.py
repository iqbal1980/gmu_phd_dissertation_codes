import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

MIN_VALUE = -80
MAX_VALUE = 40

@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)
 
@njit
def exponential_function(x, a):
    return np.exp(a * x) 


#Gene expression rate coeficient?
#I diddn't include all channels, just some 
#this is just a proof of concept!
E_KIR2_1 = 1.0
E_KIR2_2 = 0.8
E_KIR2_3 = 0.6
E_KIR4_1 = 0.9
E_KIR6_1 = 0.7
E_KIR6_2 = 0.7
E_KV1_3 = 0.5
E_KV1_5 = 0.4
E_KV2_1 = 0.6
E_KV7_4 = 0.3
E_BK = 0.8
E_CAV1_2 = 0.4
E_CAV1_3 = 0.4
E_CAV3_1 = 0.5
E_CAV3_2 = 0.5
E_TRPV4 = 0.6
E_TRPM4 = 0.7
E_P2RX4 = 0.3
E_P2RX7 = 0.3
E_CLCN3 = 0.6



G_KIR2_1 = 1.0
G_KIR2_2 = 1.0
G_KIR2_3 = 1.0
G_KIR4_1 = 1.0
G_KIR6_1 = 1.0
G_KIR6_2 = 1.0
G_KV1_3 = 1.0
G_KV1_5 = 1.0
G_KV2_1 = 1.0
G_KV7_4 = 1.0
G_BK = 1.0
G_CAV1_2 = 1.0
G_CAV1_3 = 1.0
G_CAV3_1 = 1.0
G_CAV3_2 = 1.0
G_TRPV4 = 1.0
G_TRPM4 = 1.0
G_P2RX4 = 1.0
G_P2RX7 = 1.0
G_CLCN3 = 1.0

E_K = -80.0
E_CA = 120.0
E_NA = 50.0
E_CL = -60.0



@njit(parallel=False)
def ode_system(t, y, g_gap, I_app_val, cm, dx):
    Ng = len(y)
    Vm = y

    I_kir2_1 = E_KIR2_1 * G_KIR2_1 * (Vm - E_K)
    I_kir2_2 = E_KIR2_2 * G_KIR2_2 * (Vm - E_K)
    I_kir2_3 = E_KIR2_3 * G_KIR2_3 * (Vm - E_K)
    I_kir4_1 = E_KIR4_1 * G_KIR4_1 * (Vm - E_K)
    I_kir6_1 = E_KIR6_1 * G_KIR6_1 * (Vm - E_K)
    I_kir6_2 = E_KIR6_2 * G_KIR6_2 * (Vm - E_K)
    I_kv1_3 = E_KV1_3 * G_KV1_3 * (Vm - E_K)
    I_kv1_5 = E_KV1_5 * G_KV1_5 * (Vm - E_K)
    I_kv2_1 = E_KV2_1 * G_KV2_1 * (Vm - E_K)
    I_kv7_4 = E_KV7_4 * G_KV7_4 * (Vm - E_K)
    I_bk = E_BK * G_BK * (Vm - E_K)
    I_cav1_2 = E_CAV1_2 * G_CAV1_2 * (Vm - E_CA)
    I_cav1_3 = E_CAV1_3 * G_CAV1_3 * (Vm - E_CA)
    I_cav3_1 = E_CAV3_1 * G_CAV3_1 * (Vm - E_CA)
    I_cav3_2 = E_CAV3_2 * G_CAV3_2 * (Vm - E_CA)
    I_trpv4 = E_TRPV4 * G_TRPV4 * (Vm - E_NA)
    I_trpm4 = E_TRPM4 * G_TRPM4 * (Vm - E_NA)
    I_p2rx4 = E_P2RX4 * G_P2RX4 * (Vm - E_NA)
    I_p2rx7 = E_P2RX7 * G_P2RX7 * (Vm - E_NA)
    I_clcn3 = E_CLCN3 * G_CLCN3 * (Vm - E_CL)

    I_app = np.zeros_like(Vm)
    I_app[99] = I_app_val if 100 <= t <= 400 else 0.0

    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_kir2_1[kk] + I_kir2_2[kk] + I_kir2_3[kk] + I_kir4_1[kk] + I_kir6_1[kk] + I_kir6_2[kk] + I_kv1_3[kk] + I_kv1_5[kk] + I_kv2_1[kk] + I_kv7_4[kk] + I_bk[kk] + I_cav1_2[kk] + I_cav1_3[kk] + I_cav3_1[kk] + I_cav3_2[kk] + I_trpv4[kk] + I_trpm4[kk] + I_p2rx4[kk] + I_p2rx7[kk] + I_clcn3[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_kir2_1[kk] + I_kir2_2[kk] + I_kir2_3[kk] + I_kir4_1[kk] + I_kir6_1[kk] + I_kir6_2[kk] + I_kv1_3[kk] + I_kv1_5[kk] + I_kv2_1[kk] + I_kv7_4[kk] + I_bk[kk] + I_cav1_2[kk] + I_cav1_3[kk] + I_cav3_1[kk] + I_cav3_2[kk] + I_trpv4[kk] + I_trpm4[kk] + I_p2rx4[kk] + I_p2rx7[kk] + I_clcn3[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_kir2_1[kk] + I_kir2_2[kk] + I_kir2_3[kk] + I_kir4_1[kk] + I_kir6_1[kk] + I_kir6_2[kk] + I_kv1_3[kk] + I_kv1_5[kk] + I_kv2_1[kk] + I_kv7_4[kk] + I_bk[kk] + I_cav1_2[kk] + I_cav1_3[kk] + I_cav3_1[kk] + I_cav3_2[kk] + I_trpv4[kk] + I_trpm4[kk] + I_p2rx4[kk] + I_p2rx7[kk] + I_clcn3[kk] + I_app[kk])


    return dVm_dt
    
t_span = (0, 600)
Ng = 200




def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, cm, dx, I_app_val, t_span, Ng):
    y0 = np.ones(Ng) * (-33) # Initial condition


    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, I_app_val, cm, dx), method='RK45')

    x = sol.t
    y = sol.y[100:134]

    plt.figure()
    plt.plot(x, y.T, linewidth=3)
    plt.title(f"G gap = {ggap}")
    plt.xlabel("Time")
    plt.ylabel("Vm")
    plt.show()
    
    
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200

HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.008702984240095, cm=11.0, dx=1, I_app_val=-70, t_span=t_span, Ng=Ng)


    