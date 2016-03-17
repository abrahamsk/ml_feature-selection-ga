#!/usr/bin/env python
# coding=utf-8

import deap
from deap import creator, base, tools, algorithms
import random

####################
# Program parameters
####################

# use genetic algorithm parameters specified in
# Feature Subset Selection Using a Genetic Algorithm (Yang, Honavar)
# for use in genetic_algorithm() and mutate()
# cxpb – The probability of mating two individuals.
# mutpb – The probability of mutating an individual.
# ngen – The number of generation.
# CXPB, MUTPB, NGEN = 0.6, 0.001, 20
CXPB, MUTPB, NGEN = 0.5, 0.2, 40
# change MUTPB to make mutation happen more or less often

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

##############
# GA algorithm
##############

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

################
# Simple example
################
# This algorithm reproduce the simplest evolutionary algorithm
# as presented in ch7 of Evolutionary Computation 1: Basic Algorithms and Operators (Back, 2000)
# deap.algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen)

# separate out initial population generation from the rest of the deap genetic algorithm
def initial_ga_population(pop_size):
    """
    Genetic algorithm initial population
    :param pop_size: population size
    :return pop:
    """
    pop = toolbox.population(n=pop_size)
    return pop

##########################
# More verbose alternative
##########################
# ref: deap.readthedocs.org/en/master/overview.html
# complete generational algorithm sans initial population generation
# see initial_ga_population(pop_size) above for initial pop generation
def gen_algorithm(pop_size):
    """
    Genetic algorithm for feature subset selection
    :param pop: GA population
    :return pop:
    """
    pop = toolbox.population(n=pop_size)
    # print "initial population:\n", pop  # len pop_size with 10 items at each index
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
        # print "offspring:\n", offspring

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
    # print "pop at the end of gen algorithm:\n", pop

    return pop

###############################################################################

##########################
# Non-library GA functions
##########################


def genetic_cross(gen_pop_one, gen_pop_two):
    """
    Randomly combine from the two lists of genetic populations
    Note: "population" here refers to something that functions more like a chromosome
    :param gen_pop_one:
    :param gen_pop_two:
    :return crossed_population:
    """
    # print "--------------------------------------------"
    combined_pop = [[]]
    pop_one = []
    pop_two = []
    gene = []

    zipped = zip(gen_pop_one, gen_pop_two)
    # print "zipped:\n",zipped

    for i, j in zipped:
        gene = []
        gene.append(i)
        gene.append(j)
        # print "gene in loop:\n", gene
        pop_one.append(random.choice(gene))
        pop_two.append(random.choice(gene))

    # build combined population from genetic cross
    for i in pop_one:
        combined_pop[0].append(i)
    for i in pop_two:
        combined_pop[0].append(i)

    # print "combined pop:\n", combined_pop
    # print "--------------------------------------------"
    return combined_pop

###############################################################################


def mutate(gene):
    """
    Mutate a genetic algorithm population string
    :param gene:
    :return:
    """
    # print "--------------------------------------------"
    # mutation options are 0 or 1
    options = [0,1]
    # count the number of times a mutation occurred
    mutation_counter = 0

    # select the number of mutations
    num_mutations = random.randint(0, len(gene[0])-1)
    # print "num mutations", num_mutations

    # for each potential mutation, run random chance of mutation
    # and mutate if MUTPB (mutation probability) is greater than random
    for i in xrange(num_mutations):
        for nucleotide in gene:
            # print "mutating?"
            # if chance of mutation is greater than random, mutate a random spot in the gene
            if random.random() < MUTPB:
                mutation_counter += 1
                # print "mutating!"
                # chose the spot that will mutate
                mutation_location = random.randint(0, len(nucleotide)-1)
                # print "len nucleotide", len(nucleotide)-1
                # print "mutation location", mutation_location
                # choose new value from the options of 0 and 1
                # print "element before mutation", nucleotide[mutation_location]
                mutation = random.choice(options)
                # print "mutation", mutation
                nucleotide[mutation_location] = mutation
                # print "element after mutation", nucleotide[mutation_location]
            # else:
                # print "no mutation."
    print "number of mutations:", mutation_counter
    # print "--------------------------------------------"
    return gene


###############################################################################


def create_gen_population():
    """
    create initial genetic alg feature selection strings
    :return ga_population, ga_population_deux:
    """

    ##############################################################
    # GA Feature
    # Population
    # 16 features in row of X (neural net input) + 1 for bias
    # use 17 for number of population in GA: pop will be dim 1x17
    #############################################################
    # ga_population = gen_algorithm(1)
    ga_population = initial_ga_population(1)
    # print "population before mod:", ga_population
    # Overwrite last digit to be 1 so feature selection in
    # neural net will always select bias input when building
    # feature subset for GA-selected neural net input
    # (GA selects features based on the presence of 1s in the GA population string)
    # In the non-GA neural net input, X is the input:
    # X has been concatenated with a column of 1s to use for bias input
    ga_population[-1][-1] = 1
    print "population after bias input mod:\n", ga_population

    # another GA for combining with first GA pop
    # ga_population_deux = initial_ga_population(1)
    # # overwrite last digit for neural net bias
    # ga_population_deux[-1][-1] = 1
    # print "population two after bias input mod:\n", ga_population_deux

    return ga_population


#######################################################################
# Use a combo of DEAP and non-DEAP functions for crossover and mutation
#######################################################################
def genetic_algorithm(population):
    """
    Run genetic cross on parents and mutate offspring
    Uses a mix of DEAP library and original functions
    :param ga_population:
    :param ga_population_deux:
    :return:
    """
    # run for NGEN number of generations
    # for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)
    # print "offspring:\n", offspring

    # Apply crossover and mutation on the offspring
    # for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #     if random.random() < CXPB:
    #         toolbox.mate(child1, child2)
    #         del child1.fitness.values
    #         del child2.fitness.values

    # get every second item of offspring (offspring[start:end:step] start through not past end, by step)
    sub_pop_one = offspring[0][::2]
    sub_pop_two = offspring[0][1::2]

    # apply crossover
    ga_population_crossed = genetic_cross(sub_pop_one, sub_pop_two)
    # add last digit for neural net bias
    ga_population_crossed[0].append(1)
    print "Genetic cross:\n", ga_population_crossed
    # print "len genetic cross:", len(ga_population_crossed[0])

    # time to mutate!
    ga_population_mutated = mutate(ga_population_crossed)
    # overwrite last digit for neural net bias
    ga_population_mutated[-1][-1] = 1
    print "Mutated:\n", ga_population_mutated

    # print "orig len", len(ga_population[0])
    # print "mutated len", len(ga_population_mutated[0])

    return ga_population_mutated


def main():
    ga_population = create_gen_population()
    ga_population = genetic_algorithm(ga_population)


if __name__ == "__main__":
    main()