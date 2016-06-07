import numpy as np
import math 
import plotting as plot
from fitness import Fitness
from geneticAlgorithm import GeneticAlgorithm
from differentialEvolution import DifferentialEvolution
from PSO import PSO
from GASelection import *
from GACrossOver import *
from GAMutation import *
from DEMutation import *
from DESelection import *
from DECrossOver import *


dom = (-5, 12)
np.set_printoptions(suppress=True)

# plot.plot3d(fitness)
# plot.plotcontour(fitness)

####### Genetic Algorithm execution

# gen = GeneticAlgorithm(200, dom)
# gen.fitness(fitness1)

# gen.run(GASelectionTournament(), GACrossOverLinearOperator(), GAMutationGaussian())

# print gen.population

####### Differential Evolution execution

# de = DifferentialEvolution(10, dom)
# de.fitness(fitness1)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# print de.population

####### Particle Swarm Optimization

# pso = PSO(10, dom, 1, 1)
# print pso.population

# pso.fitness(fitness1)

# pso.run()
# print pso.population

Fitness.initAeroData()
Fitness.LSE([0.2, 0.1, 0.3])









