import numpy as np
from deap import base, creator, tools, algorithms
from numba import njit
import pickle
import os
from scipy.optimize import curve_fit


# Random number not needed really
random_number = 1

global_dx = 1  # No spatial discretisation in network model

# Constants for safe_log and safe_exponential
MIN_VALUE = -80 # Minimum physiologically reasonable value for Vm
MAX_VALUE = 40 # Maximum physiologically reasonable value for Vm



# Functions from verify_params11.py for numerical stability
@njit
def safe_log(x):
    MIN_VALUE = -80
    if x <= 0:
        return MIN_VALUE
    return np.log(x)

@njit
def exponential_function(x, a):
    return np.exp(a * x) 

@njit
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)

# Simulate process function (similar to verify_params11.py)
@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o):
    dt = 0.001
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
            E_K = (R * 293 / F) * safe_log(K_o/150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm += random_number * 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                new_Vm += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VALUE), MIN_VALUE)

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

    return A


# Fitness function adapted from plot_data2_modified in verify_params11.py
def objective(params):
    print("Starting objective function...")
    
    # Clip each parameter to lie within its bounds.
    for i in range(len(params)):
        low, high = param_bounds[i]
        params[i] = np.clip(params[i], low, high)
    
    
    ggap, Ikir_coef, cm, K_o = params
    
    dx = global_dx
    Ibg_init = 0.7 * 0.94
    
    # Run the simulation with the provided parameters
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)

    #----------------------------------------------------------------------------------------------------------
    
    cellLength = 60  # in Microns
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / np.abs(A[99000, 101:135])[0]
    
    #print(D)

    distance_m = cellLength * np.arange(102-102, 136-102)
    
    A_initial = D[0]
    B_initial = np.log(D[1] / D[0]) / (distance_m[1] - distance_m[0])
    
    #print(distance_m)

    # Check for NaNs in D and distance_m before curve_fit
    if np.any(np.isnan(D)) or np.any(np.isnan(distance_m)):
        print("NaN detected in D or distance_m")
        print(f"D: {D}, distance_m: {distance_m}")
        return None

    try:
        # Fit the experimental data to the exponential decay function
        popt, _ = curve_fit(exponential_decay_function, distance_m, D, p0=[A_initial, B_initial])
        A, B = popt

        # Generate simulated exponential decay with fitted parameters
        simulated_decay = exponential_decay_function(distance_m, A, B)

        # Reference exponential decay function
        reference_decay = 1 * np.exp(-0.003 * distance_m)

        # Calculate the loss as sum of squared differences
        loss = np.sum((simulated_decay - reference_decay) ** 2)
        
        if loss <= 0.5:
            with open('good_genes.txt', 'a') as file:
                file.write(f'Params: {params}, Loss: {loss}\n')
                
    except RuntimeError as e:
        print("Error in curve fitting:", e)
        return None

    print("Completed objective function with loss:", loss)
    return (loss,)






##################################################################################################################
# Define the individual and its fitness.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()



param_bounds = [
    (0.1, 35),  # ggap
    (0.90, 0.96),  # Ikir_coef
    (8, 11),     # cm
    (1, 8)       # K_o
]

# Attribute generator
toolbox.register("attr_float", np.random.uniform, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [np.random.uniform(low, high) for low, high in param_bounds])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective)

# Checkpointing functions
def save_checkpoint(population, filename="checkpoint.pkl"):
    with open(filename, "wb") as cp_file:
        pickle.dump(population, cp_file)

def load_checkpoint(filename="checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as cp_file:
            return pickle.load(cp_file)
    return None

def main():
    checkpoint_file = "checkpoint.pkl"
    
    # Check if a checkpoint file exists
    pop = load_checkpoint(checkpoint_file)
    if pop is None:
        print("No checkpoint found. Initializing population...")
        #pop = toolbox.population(n=60000)
        pop = toolbox.population(n=1000)
    else:
        print("Resuming from checkpoint.")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Starting evolution...")
    for gen in range(50):  # Specify the number of generations
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        for i, ind in enumerate(pop):
                    ind.fitness.values = toolbox.evaluate(ind)
                    #if (i + 1) % 50 == 0:
                    if (i + 1) % 10 == 0:
                        save_checkpoint(pop, checkpoint_file)

        hof.update(pop)
        record = stats.compile(pop)
        print(record)

        # Save checkpoint every generation
        save_checkpoint(pop, checkpoint_file)

    print("Evolution completed!")
    print("Best individual is: ", hof[0], " with fitness: ", hof[0].fitness)

    return pop, stats, hof


##################################################################################################################
















if __name__ == "__main__":
    main()














