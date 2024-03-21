import numpy as np
import random
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
from numba import njit
import numpy as np

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

# Example of a simplified ode_system function
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


# Define the simulation function (adjusted to work with DEAP individual parameters)
def run_simulation(individual, t_span, Ng, time_in_ms, cell_index):
    ggap, Ikir_coef, cm, K_o, I_app_val = individual
    y0 = np.ones(Ng) * (-33)
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, 0.7 * 0.94, Ikir_coef, cm, 1, K_o, I_app_val), method='Radau')
    time_idx = np.argmin(np.abs(sol.t - time_in_ms))
    voltage = sol.y[cell_index, time_idx]
    return voltage

# Evaluation function for DEAP
def evalSimulation(individual):
    TARGET_VOLTAGE = -30  # Example target voltage, adjust as needed
    voltage = run_simulation(individual, (0, 600), 200, 300, 100)  # Adjust time_in_ms and cell_index as needed
    fitness = abs(voltage - TARGET_VOLTAGE)
    return (fitness,)

# Set up DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_ggap", random.uniform, 0.1, 35)
toolbox.register("attr_Ikir_coef", random.uniform, 0.90, 0.96)
toolbox.register("attr_cm", random.uniform, 8, 11)
toolbox.register("attr_K_o", random.uniform, 1, 8)
toolbox.register("attr_I_app_val", random.uniform, -70, 70)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_ggap, toolbox.attr_Ikir_coef, toolbox.attr_cm, toolbox.attr_K_o, toolbox.attr_I_app_val), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalSimulation)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Evolutionary algorithm
def main():
    random.seed(64)
    population = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(population))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values =             fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    return population, best_ind

if __name__ == "__main__":
    final_population, best_solution = main()

