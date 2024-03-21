import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import operator
import math
import random
from deap import gp, creator, base, tools, algorithms
import logging


# Define a logging function
def log_best(gen, stats):
    best_ind = hof.get()[0]
    best_fitness = best_ind.fitness.values[0]
    best_expr = gp.stringify(best_ind)
    logging.info(f"Generation {gen}: Best fitness = {best_fitness}, Expression = {best_expr}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Constants
MIN_VALUE = -80
MAX_VALUE = 40
MIN_VM = -80  # Minimum physiologically reasonable value for Vm
MAX_VM = 40   # Maximum physiologically reasonable value for Vm

# Safe logarithm function
@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)

# Exponential functions with @njit for performance
@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
    # Constants within the function
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y

    # Calculating currents
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))

    # Application current adjustment
    I_app = np.zeros_like(Vm)
    I_app[99] = I_app_val if 100 <= t <= 400 else 0.0

    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])

    return dVm_dt

def run_simulation(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index):
    y0 = np.ones(Ng) * (-33)  # Initial condition

    # Solving the ODE with specified method
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val), method='Radau')

    # Extract the voltage for the specified time and cell.
    time_idx = np.argmin(np.abs(sol.t - time_in_ms))  # Find index of the closest time point to time_in_ms
    voltage = sol.y[cell_index, time_idx]  # Extract voltage

    return voltage

# Define a protected division function
def protected_div(x1, x2):
    if abs(x2) < 1e-10:
        return 1.0
    else:
        return x1 / x2

# Generate training data
training_points = []
param_bounds = [
    np.linspace(0.1, 35, 10),  # ggap (10 values)
    np.linspace(0.90, 0.96, 5),  # Ikir_coef (5 values)
    np.linspace(8, 11, 4),     # cm (4 values)
    np.linspace(1, 8, 8),       # K_o (8 values)
    np.linspace(-70, 70, 15), # I_app_val (15 values)
    np.linspace(1, 600000, 100), # time_in_ms (10 values)
    #np.linspace(1, 201, 50)
    np.arange(1, 201)  # cell_index (all values from 1 to 200)
]

num_points = 1000  # Number of training points
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200

for _ in range(num_points):
   
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    time_in_ms = np.random.choice(param_bounds[5])
    cell_index = np.random.choice(param_bounds[6])
    print("hi")
    voltage = run_simulation(ggap, Ibg_init_val, Ikir_coef, cm, dx=1, K_o=K_o, I_app_val=I_app_val, t_span=t_span, Ng=Ng, time_in_ms=time_in_ms, cell_index=cell_index)
    print("voltage="+voltage)
    training_points.append((ggap, Ikir_coef, cm, K_o, I_app_val, time_in_ms, cell_index, voltage))

# Define the primitive set
pset = gp.PrimitiveSet("MAIN", 8)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(math.log, 2)
pset.addPrimitive(math.tan, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sqrt, 1)  # Add square root operator
pset.addPrimitive(operator.pow, 2)  # Add power operator
pset.addPrimitive(operator.pow, 3)  # Add power operator
pset.addPrimitive(operator.pow, 4)  # Add power operator
pset.addEphemeralConstant("const", lambda: random.uniform(-1, 1))

pset.renameArguments(ARG0='ggap')
pset.renameArguments(ARG1='Ikir_coef')
pset.renameArguments(ARG2='cm')
pset.renameArguments(ARG3='K_o')
pset.renameArguments(ARG4='I_app_val')
pset.renameArguments(ARG5='time_in_ms')
pset.renameArguments(ARG6='cell_index')
pset.renameArguments(ARG7='Ibg_init')  # Assuming Ibg_init is a constant

# Define the fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

def evaluate_individual(individual, points):
    func = gp.compile(individual, pset)
    fitness = sum(abs(func(*point[:-1]) - point[-1]) for point in points)
    return fitness,

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual, points=training_points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#Set up the genetic programming algorithm
pop = toolbox.population(n=500)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)
for gen, logbook in enumerate(log, start=1):
    log_best(gen, logbook)

#Print the best individual
print("Best individual: ", gp.stringify(hof.get()[0]))

#Example usage
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
time_in_ms = 300 # Example time in ms
cell_index = 100 # Example cell index

voltage = run_simulation(ggap=20.008702984240095, Ibg_init=Ibg_init_val, Ikir_coef=0.9, cm=11.0, dx=1, K_o=4.502039575569403, I_app_val=-70, t_span=t_span, Ng=Ng, time_in_ms=time_in_ms, cell_index=cell_index)
print(f"Voltage at time {time_in_ms}ms for cell index {cell_index}: {voltage}")