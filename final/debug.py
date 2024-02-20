# Constants for safe_log and safe_exponential
MIN_VALUE = -10000
MAX_VALUE = 1e150

import numpy as np


def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)


def safe_exponential(x, n):
    if x > 700 or x < -700 or n > 100:
        return MAX_VALUE
    return np.power(np.exp(x), n)

F = 9.6485e4
R = 8.314e3
K_o=3
g_gap = 0.3015830801507125
dt=0.001
dx=1
cm=9.4
Ibg_init = 0.7 * 0.94
Ikir_coef=0.94

eki1 = (g_gap * dt) / (dx**2 * cm)
print(eki1)


eki2 = dt / cm
print(eki2)

Vm = -33

E_K = (R * 293 / F) * safe_log(K_o/150)
I_bg = Ibg_init * (Vm + 30)
I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + safe_exponential((Vm - E_K - 25) / 7, 1)))

print(E_K)
print(I_bg)
print(I_kir)