import numpy as np
from deap import base, creator, tools, algorithms
from numba import njit
import pickle
import os
from scipy.optimize import curve_fit


random_number = 1


def exp_func(x, a, b):
    return a * np.exp(b * x)

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
            E_K = (R * 293 / F) * np.log(K_o / 150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))

            if kk == 0:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                Vm[kk] += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif 98 <= kk <= 100:
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





def objective(params):
    print("Starting objective function...")
    
    # Clip each parameter to lie within its bounds.
    for i in range(len(params)):
        low, high = param_bounds[i]
        params[i] = np.clip(params[i], low, high)
    
    ggap, Ikir_coef = params
    
    cm = 9.4
    dx = 0.06
    K_o = 3
    Ibg_init = 0.7 * 0.94
    
    # Run the simulation with the provided parameters
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)
    
    # Exponential fit to the simulation data
    distance_m = dx * np.arange(99, 136)
    simulation_data = A[399998, 98:135] / A[99000, 98:135][0]  # normalized
    
    # Check for inf or NaN values in the simulation data
    if np.any(np.isnan(simulation_data)) or np.any(np.isinf(simulation_data)):
        print("Invalid values detected in simulation data.")
        return 1e10,  # High loss value

    # Initial guess for the parameters a and b
    #initial_guess = [1.1815, -0.052] #Fron actual data? y = 0.5191e-4E-05x
    initial_guess = [0.519, -0.00004] #Fron actual data? y = 0.5191e-4E-05x

    try:
        # Perform the curve fit
        popt, pcov = curve_fit(exp_func, distance_m, simulation_data, p0=initial_guess)
        # Calculate the R-squared value
        residuals = simulation_data - exp_func(distance_m, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((simulation_data - np.mean(simulation_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Use the R-squared value to calculate the loss
        loss = (1 - r_squared)**2
    except RuntimeError as e:
        print("Error in curve fitting:", e)
        loss = 1e10  # Assign a high loss value if fitting fails
    
    if np.isnan(loss) or np.isinf(loss):
        print("NAN or INF loss detected with params:", params)
        loss = 1e10
    
    print("Completed objective function with loss:", loss)
    return loss,

    
    
    


# Define the individual and its fitness.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

param_bounds = [
    (0.1, 35),  # ggap
    (0.94, 0.94),  # Ikir_coef
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
        pop = toolbox.population(n=60000)
    else:
        print("Resuming from checkpoint.")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Starting evolution...")
    for gen in range(100):  # Specify the number of generations
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        for i, ind in enumerate(pop):
                    ind.fitness.values = toolbox.evaluate(ind)
                    if (i + 1) % 50 == 0:
                        save_checkpoint(pop, checkpoint_file)

        hof.update(pop)
        record = stats.compile(pop)
        print(record)

        # Save checkpoint every generation
        save_checkpoint(pop, checkpoint_file)

    print("Evolution completed!")
    print("Best individual is: ", hof[0], " with fitness: ", hof[0].fitness)

    return pop, stats, hof

if __name__ == "__main__":
    main()