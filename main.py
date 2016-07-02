import numpy as np
import math 
import plotting as plot
import matplotlib.pyplot as plt
from fitnessFunctions import Fitness
from EAs.geneticAlgorithm import GeneticAlgorithm
from EAs.differentialEvolution import DifferentialEvolution
from PSO.PSO import PSO
from EAs.SelectionAlgs.GASelection import *
from EAs.CrossOverAlgs.GACrossOver import *
from EAs.MutationAlgs.GAMutation import *
from EAs.MutationAlgs.DEMutation import *
from EAs.SelectionAlgs.DESelection import *
from EAs.CrossOverAlgs.DECrossOver import *
from FuzzyHelpers.fuzzyAlgorithm import *
from FuzzyHelpers.fuzzySets import *
from FuzzyHelpers.fuzzyInference import *
from FuzzyHelpers.defuzzyfication import *
from fileHelper import FileExtractor

dom = (-5, 12)
dom2 = (-100,100)

np.set_printoptions(suppress=True)


#plt.show()

# plot.plot3d(fitness)
# plot.plotcontour(fitness)

####### Genetic Algorithm execution

# gen = GeneticAlgorithm(200, 2, dom)
# gen.fitness(fitness1)

# gen.run(GASelectionTournament(), GACrossOverLinearOperator(), GAMutationGaussian())

# print gen.population

####### Differential Evolution execution

# de = DifferentialEvolution(10, 2, dom)
# de.fitness(Fitness.fitness1)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# print de.population

####### Particle Swarm Optimization

# pso = PSO(10, dom, 1, 1)
# print pso.population

# pso.fitness(fitness1)

# pso.run()
# print pso.population

####### Diff Evolution with LSE

# Fitness.initAeroData()

# de = DifferentialEvolution(20, 4, dom2)
# print de.population

# de.fitness(Fitness.LSE)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# print de.population

####### Diff Evolution with LSEwR

# Fitness.initAeroData()

# de = DifferentialEvolution(20, 4, dom2)
# print de.population

# de.fitness(Fitness.LSEwR)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# print de.population

########Fuzzy Algorithm Mandani

# fuzzy_alg = Fuzzy()
# fuzzy_alg.setFuzzyInputSets(TriangularFuzzySets([(-15,40), (30,50), (45,55), (50,70), (60,120)]))
# fuzzy_alg.setFuzzyInputSets(TriangularFuzzySets([(-90,10), (0,50), (40,90), (80,100), (90,140), (130,180), (170,270)]))
# fuzzy_alg.setFuzzyOutputSets(TriangularFuzzySets([(-30,-15), (-25,-5), (-10,0), (-5,5), (0,10), (5,25), (15,30)]))

# ruleMatrix = FileExtractor.ruleMatrix()

# solution = fuzzy_alg.run(MandaniInference(ruleMatrix), [47.5, 99], AggregateThenCentroid())
# solution2 = fuzzy_alg.run(MandaniInference(ruleMatrix), [47.5, 99], CentroidThenAggregate())
# print solution 
# print solution2 

########Fuzzy Algorithm Takagi-Sugeno

Fitness.initGauss3Data("gauss3.dat")

takagiInstance = TakagiSugenoInference(Fitness.takagiSugeno)

print takagiInstance.fuzzyInterval
print takagiInstance.ruleMatrix

fuzzy_alg = Fuzzy()
fuzzy_alg.setFuzzyInputSets(GaussianFuzzySets(takagiInstance.fuzzyInterval))

solution = [fuzzy_alg.run(takagiInstance, [i]) for i in range(0,250)]











