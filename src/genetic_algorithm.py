#!/usr/bin/env python
# coding=utf-8

# Katie Abrahams
# abrahake@pdx.edu
# ML Independent Study
# Winter 2016

import deap
# from deap import creator, base, tools, algorithms
import random

#########################
# Genetic algorithm setup
#########################

# Sourced from deap.readthedocs.org/en/1.0.x/overview.html
# first thing to do is make appropriate type for your problem.
# DEAP enables you to build your own types
# single objective minimizing fitness named FitnessMin:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Individual class derived from a list with a fitness attribute set to the just created fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

# Init:
# Once the types are created, fill them with sometimes random values
# initialize populations from individuals that are themselves initialized with random float numbers
IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operators
# Operators are just like initalizers, excepted that some are already implemented in the tools module.
# Create and register operators with the toolbox:
# registered functions are renamed by the toolbox to allow genericity,
# so that the algorithm does not depend on operators name

def evaluate(individual):
    return sum(individual),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

###############################################################################

############
# Algorithm
############

# use one of the algorithms from the algorithms module
# The algorithms module is intended to contain some specific algorithms
# in order to execute common evolutionary algorithms
# Sourced from deap.readthedocs.org/en/1.0.x/api/algo.html#module-deap.algorithms

# The algorithm takes in a population and evolves it in place using the varAnd() method.
# returns the optimized population and a Logbook with the statistics of the evolution (if any).
# The logbook will contain the generation number, the number of evalutions for each generation
# and the statistics if a Statistics if any.
# The cxpb and mutpb arguments are passed to the varAnd() function.

# Algorithm goes as follows:
# 1) It evaluates the individuals with an invalid fitness.
# 2) It enters the generational loop where the selection procedure is applied to
# entirely replace the parental population. The 1:1 replacement ratio of this algorithm
# requires the selection procedure to be stochastic and to select multiple times the same individual, for example, selTournament() and selRoulette().
# 3) It applies the varAnd() function to produce the next generation population.
# 4) It evaluates the new individuals and compute the statistics on this population.
# 5) When ngen generations are done, the algorithm returns a tuple with the final
# population and a Logbook of the evolution.

# Pseudocode
# evaluate(population)
# for g in range(ngen):
#     population = select(population, len(population))
#     offspring = varAnd(population, toolbox, cxpb, mutpb)
#     evaluate(offspring)
#     population = offspring

"""
Parameters:

    population – A list of individuals.
    toolbox – A Toolbox that contains the evolution operators.
    cxpb – The probability of mating two individuals.
    mutpb – The probability of mutating an individual.
    ngen – The number of generation.
    stats – A Statistics object that is updated inplace, optional.
    halloffame – A HallOfFame object that will contain the best individuals, optional.
    verbose – Whether or not to log the statistics.

Returns:

The final population and a Logbook with the statistics of the evolution.
"""
# deap.algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen)
