#!/usr/bin/env python
# coding=utf-8

# Katie Abrahams
# abrahake@pdx.edu
# ML Independent Study
# Winter 2016

import deap
from deap import creator, base, tools, algorithms
import random
from neural_net_ga import *

####################
# Program parameters
####################

# use genetic algorithm parameters from paper
# for use in genetic_algorithm() and mutate()
# cxpb – The probability of mating two individuals.
# mutpb – The probability of mutating an individual.
# ngen – The number of generation.
CXPB, MUTPB, NGEN = 0.6, 0.001, 20


#########################
# Genetic algorithm setup
#########################

# Sourced from deap.readthedocs.org/en/1.0.x/overview.html
# first thing to do is make appropriate type for your problem.
# DEAP enables you to build your own types
# single objective maximizing fitness named FitnessMax:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Individual class derived from a list with a fitness attribute set to the just created fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

# Init:
# Once the types are created, fill them with sometimes random values
# initialize populations from individuals that are themselves initialized with random float numbers
IND_SIZE = 17

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, 1) # used to create a vector of 0s and 1s
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
# requires the selection procedure to be stochastic and to select multiple times the same
# individual, for example, selTournament() and selRoulette().
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
# separated out initial population generation from the rest of the deap generatic algorithm
def initial_ga_population(pop_size):
    """
    Genetic algorithm for feature subset selection
    :param pop_size: population size
    :return pop:
    """
    pop = toolbox.population(n=pop_size)
    return pop

# deap.algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen)


# complete generational algorithm (sans initial population generation)
def gen_algorithm(pop):
    """
    Genetic algorithm for feature subset selection
    :param pop: GA population
    :return pop:
    """
    # pop = toolbox.population(n=pop_size)
    # print "pop before changes: ", len(pop)  # len pop_size with 10 items at each index
    # use genetic algorithm parameters from paper
    # CXPB, MUTPB, NGEN = 0.6, 0.001, 20

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

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
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

###############################################################################


def genetic_cross(gen_pop_one, gen_pop_two):
    """
    Randomly combine from the two lists of genetic populations
    Note: "population" here refers to something that functions more like a chromosome
    :return crossed_population:
    """
    combined_pop = []
    gene = []
    for i, j in zip(gen_pop_one, gen_pop_two):
        gene.append(i)
        gene.append(j)
        combined_pop.append(random.choice(gene))

    return combined_pop

###############################################################################


def mutate(gene):
    """
    Mutate a genetic algorithm population string
    :param gene:
    :return:
    """
    options = [0,1]
    for nucleotide in gene:
        # if chance of mutation is greater than random, mutate a random spot in the gene
        if MUTPB > random.random():
            # chose the spot that will mutate
            mutation_location = random.randint(0, len(nucleotide))
            # choose new value from the options of 0 and 1
            nucleotide[mutation_location] = random.choice(options)
    return gene


###############################################################################


def main():
    test = 5
    population = gen_algorithm(test)
    print "test population len after function exit: ", len(population)

if __name__ == "__main__":
    main()