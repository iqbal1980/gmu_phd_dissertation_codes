import pandas as pd
import operator
from deap import gp, tools, base, creator, algorithms
import random
import functools
import numpy
import math

# Define a protected log function
def protectedAsin(x):
    return math.asin(max(min(x, 1), -1))

def protectedAcos(x):
    return math.acos(max(min(x, 1), -1))

def protectedLog(x):
    return math.log(max(x, 1e-6))  # Using a small positive constant instead of 0

def protectedSqrt(x):
    return math.sqrt(max(x, 0))



def protectedLog(x):
    return math.log(abs(x) + 1)

def protectedDiv(left, right):
    if right == 0:
        return 1
    return left / right


# Read data from CSV
df = pd.read_csv('simulation_data.csv')
input_data = df[['ggap', 'Ibg_init', 'Ikir_coef', 'cm', 'dx', 'K_o']].values
output_data = df['Output'].values

# Define Primitive Set for Symbolic Regression
pset = gp.PrimitiveSet("MAIN", arity=6)  # arity=6 for six input variables
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)  # A division operator that protects against division by zero
pset.addPrimitive(operator.neg, 1)
# Adding new primitives to the primitive set


pset.addPrimitive(protectedAsin, 1)
pset.addPrimitive(protectedAcos, 1)
pset.addPrimitive(protectedLog, 1)


pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.tan, 1)


pset.addPrimitive(protectedLog, 1)  # Using protected log
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(protectedSqrt, 1) # Using protected sqrt
pset.addPrimitive(lambda x: x**3, 1, name="cube")  # Cube function
# Use functools.partial for pi and rand101 to avoid pickling issues
pset.addEphemeralConstant("pi", functools.partial(lambda: math.pi))
pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))
pset.renameArguments(ARG0='ggap', ARG1='Ibg_init', ARG2='Ikir_coef', ARG3='cm', ARG4='dx', ARG5='K_o')

# Define Fitness Function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing the error
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points, target):
    # Transform the tree expression into a callable function
    func = toolbox.compile(expr=individual)

    try:
        sqerrors = ((func(*args) - targ) ** 2 for args, targ in zip(points, target))
        mse = math.fsum(sqerrors) / len(points)
        if numpy.isinf(mse) or numpy.isnan(mse):
            return float('inf'),
        return mse,
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
        # Assign a high penalty in case of an error
        return float('inf'),



toolbox.register("evaluate", evalSymbReg, points=input_data, target=output_data)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Run Genetic Programming
pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)

algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, stats=stats, halloffame=hof)

# Best solution
print(hof[0])
