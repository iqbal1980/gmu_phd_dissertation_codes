import geppy as gep
from deap import creator, base, tools
import numpy as np
import pandas as pd
import random
import operator
import math
from geppy.tools import crossover as gep_crossover
from geppy.tools import mutation as gep_mutation
from deap import algorithms


head_length = 10  # Define the head length for the genes
rnc_array_length = 5  # Define the length of the RNC array

# Import your dataset
df = pd.read_csv('simulation_data.csv')
input_data = df[['ggap', 'Ibg_init', 'Ikir_coef', 'cm', 'dx', 'K_o']].values
output_data = df['Output'].values

# Define protected functions
def protectedDiv(left, right):
    return 1 if right == 0 else left / right

def protectedLog(x):
    return math.log(abs(x) + 1)
    
    
def rnc_gen():
    return random.uniform(-1, 1)  # Example RNC generator function

# other protected functions ...

# Create primitive set
pset = gep.PrimitiveSet('MAIN', input_names=['ggap', 'Ibg_init', 'Ikir_coef', 'cm', 'dx', 'K_o'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protectedDiv, 2)
pset.add_function(protectedLog, 1)
# add other functions and terminals ...

# Create fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=head_length, rnc_gen=rnc_gen, rnc_array_length=rnc_array_length)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=2, linker=operator.add)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    func = toolbox.compile(individual)
    # Calculate squared errors
    sqerrors = ((func(*in_vars) - out_var) ** 2 for in_vars, out_var in zip(input_data, output_data))
    return math.fsum(sqerrors) / len(input_data),  # mean squared error

toolbox.register('compile', gep.compile_, pset=pset)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=3)
# ... [previous code remains unchanged]

toolbox.register('mate', gep_crossover.crossover_one_point)
toolbox.register('mutate', gep_mutation.mutate_uniform, pset=pset, ind_pb='2p')

# ... [rest of the code]

# Example setup for gene generator






# Define statistics and hall-of-fame
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
hof = tools.HallOfFame(1)

# Run Genetic Programming
pop = toolbox.population(n=100000)
algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 40, stats=stats, halloffame=hof)

# Print the best individual
best_ind = hof[0]
print(best_ind)
