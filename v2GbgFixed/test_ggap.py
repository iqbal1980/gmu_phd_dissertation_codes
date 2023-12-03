import numpy as np
from deap import base, creator, tools, algorithms
from skopt.space import Real
from numba import njit

random_number = 1


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

def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    c = np.polyfit(distance_m, D, 1)


 
def objective(params):
    print("Starting objective function...")
    
    # Clip each parameter to lie within its bounds.
    for i in range(len(params)):
        low, high = param_bounds[i]
        params[i] = np.clip(params[i], low, high)
        
    #ggap, Ibg_init, Ikir_coef, cm, dx, K_o = params
    ggap, Ikir_coef= params
    
    cm = 9.4
    dx = 0.06
    K_o = 3
    Ibg_init = 0.7*0.94
    
    #Run the simulation with the provided parameters
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o)
    
    dx = 0.06
    #D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    D = np.abs(A[399998, 98:135] - A[99000, 98:135]) / np.abs(A[99000, 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    coefficients = np.polyfit(distance_m, D, 1)
    #loss = (coefficients[0] - 2)**2 + (coefficients[1] - 3.2)**2
    loss = (coefficients[0] + 0.0000000000001)**2 + (coefficients[1] - 0.6)**2 # 
    if np.isnan(loss) or np.isinf(loss):
        print("NAN loss detected with params:", params)
        loss = 1e10
        
    print("Completed objective function!")
    return loss, 

# Define the individual and its fitness. The fitness should be minimized in this case.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

param_bounds = [
    (0.1, 35),  # ggap
    #(0.1, 1.5),      # Ibg_init
    (0.94, 0.94),  # Ikir_coef
    #(8, 11),     # cm
    #(0.01, 0.09),  # dx
    #(1, 8)       # K_o
]


# Attribute generator 
toolbox.register("attr_float", np.random.uniform, 0, 1)

# Modify the individual initialization
def init_individual(icls, content):
    return icls(content)

def init_population(pcls, ind_init, n):
    return pcls(ind_init() for _ in range(n))

# Register the new initialization methods
toolbox.register("individual_guess", init_individual, creator.Individual)
toolbox.register("population_guess", init_population, list, toolbox.individual_guess)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [np.random.uniform(low, high) for low, high in param_bounds])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)




toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective)

def main():
    print("Initializing population...")
    pop = toolbox.population(n=6000)
    print("Population initialized.")
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Starting evolution...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=True)
    print("Evolution completed!")                   
    
    print("Best individual is: ", hof[0], " with fitness: ", hof[0].fitness)

    return pop, stats, hof

if __name__ == "__main__":
    main()






