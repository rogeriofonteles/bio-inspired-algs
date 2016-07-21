import numpy as np
import math 
import plotting as plot
import plotFunctions as pltF
import matplotlib.pyplot as plt
from numpy import linalg as LA
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
from ANN.simplePercepton import *
from ANN.multiLayeredPerceptron import *
from ANN.activationFunction import *
from ANN.trainingMode import *
from fileHelper import FileExtractor
from timeDecorator import TimeDecorator

dom = (-5.12, 5.12)
dom2 = (-1, 1)
dom3 = (-1000,1000)

np.set_printoptions(suppress=True)

# plt.show()

# plot.plot3d(Fitness.fitness1)
# plot.plotcontour(Fitness.fitness1)

####### Genetic Algorithm execution

# timeVector = []

# for i in [10, 100, 200]:
# 	gen = GeneticAlgorithm(i, 2, dom)
# 	gen.fitness(Fitness.fitness1)

# 	de = DifferentialEvolution(i, 2, dom)
# 	de.fitness(Fitness.fitness1)

# 	pso = PSO(i, dom, 1, 1)
# 	pso.fitness(Fitness.fitness1)

# 	timeVector.append([TimeDecorator.time(gen.run, GASelectionTournament(), GACrossOverLinearOperator(), GAMutationGaussian(0.01)),
# 			 		   TimeDecorator.time(de.run, DESelectionBest(), DECrossOverBin(), DEMutationRand1()),
# 			 		   TimeDecorator.time(pso.run)])

# 	pltF.Plot.show(gen.plot[0], de.plot[0], pso.plot[0])
# 	plt.show()

# [plt.plot([10, 100, 200], np.transpose(timeVector)[i]) for i in range(3)]
# plt.show()

####### Differential Evolution execution

# de = DifferentialEvolution(100, 2, dom)
# de.fitness(Fitness.fitness1)

# TimeDecorator.time(de.run, DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# pltF.Plot.show(de.plot[0], de.plot[1])
# plt.show()


####### Particle Swarm Optimization

# pso = PSO(10, dom, 1, 1)
# print pso.population

# pso.fitness(fitness1)

# pso.run()
# print pso.population

####### Diff Evolution with LSE

# Fitness.initAeroData("aerogerador.dat")

# chromossomeLength = 6	

# de = DifferentialEvolution(10, chromossomeLength, dom2)
# #print de.population

# de.fitness(Fitness.LSE)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# best = de.returnBest()

# print best, Fitness.LSE(best)

# polyfit = np.polyfit(Fitness.aerodata[:,0], Fitness.aerodata[:,1], 5)

# print polyfit, Fitness.LSE(polyfit[::-1])

# plt.subplot(211)
# plt.plot(np.linspace(0,14.4, 100), [sum([best[i]*(j**i) for i in range(chromossomeLength)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.subplot(212)
# plt.plot(np.linspace(0,14.4, 100), [sum([polyfit[6-1-i]*(j**i) for i in range(6)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.plot(Fitness.aerodata[:,0], Fitness.aerodata[:,1], "rx")

# plt.show()

####### PSO with LSE

# Fitness.initAeroData("aerogerador.dat")

# chromossomeLength = 6

# pso = PSO(100, chromossomeLength, dom2, 1.5, 1.5)
# #print de.population

# pso.fitness(Fitness.LSE)

# pso.run()

# best = pso.returnBest()

# print best, Fitness.LSE(best)

# polyfit = np.polyfit(Fitness.aerodata[:,0], Fitness.aerodata[:,1], 5)
	
# print polyfit, Fitness.LSE(polyfit[::-1])

# plt.subplot(211)
# plt.plot(np.linspace(0,14.4, 100), [sum([best[i]*(j**i) for i in range(chromossomeLength)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.subplot(212)
# plt.plot(np.linspace(0,14.4, 100), [sum([polyfit[5-i]*(j**i) for i in range(6)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.plot(Fitness.aerodata[:,0], Fitness.aerodata[:,1], "rx")

# plt.show()

####### Diff Evolution with LSEwR

# Fitness.initAeroData("aerogerador.dat")

# chromossomeLength = 6

# de = DifferentialEvolution(10, chromossomeLength, dom2)

# de.fitness(Fitness.LSEwR)

# de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())

# best = de.returnBest()

# pltF.Plot.show(de.plot[0], de.plot[1])

# print best, Fitness.LSEwR(best)

# polyfit = np.polyfit(Fitness.aerodata[:,0], Fitness.aerodata[:,1], 5)

# print polyfit, Fitness.LSEwR(polyfit[::-1])

# plt.subplot(211)
# plt.plot(np.linspace(0,14.4, 100), [sum([best[i]*(j**i) for i in range(chromossomeLength)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.subplot(212)
# plt.plot(np.linspace(0,14.4, 100), [sum([polyfit[5-i]*(j**i) for i in range(6)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.plot(Fitness.aerodata[:,0], Fitness.aerodata[:,1], "rx")

# plt.show()

####### PSO with LSEwR

# Fitness.initAeroData("aerogerador.dat")

# chromossomeLength = 4

# pso = PSO(100, chromossomeLength, dom3, 1.5, 1.5)
# #print de.population

# pso.fitness(Fitness.LSEwR)

# pso.run()

# best = pso.returnBest()

# print best, Fitness.LSE(best)

# polyfit = np.polyfit(Fitness.aerodata[:,0], Fitness.aerodata[:,1], 5)
	
# print polyfit, Fitness.LSE(polyfit[::-1])

# plt.subplot(211)
# plt.plot(np.linspace(0,14.4, 100), [sum([best[i]*(j**i) for i in range(chromossomeLength)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.subplot(212)
# plt.plot(np.linspace(0,14.4, 100), [sum([polyfit[5-i]*(j**i) for i in range(6)]) for j in np.linspace(0,14.4, 100)], "b-")
# plt.plot(Fitness.aerodata[:,0], Fitness.aerodata[:,1], "rx")

# plt.show()


########Fuzzy Algorithm Mandani

# fuzzy_alg = Fuzzy()
# fuzzy_alg.setFuzzyInputSets(TriangularFuzzySets([(-15,35), (30,50), (45,55), (50,70), (65,115)]))
# fuzzy_alg.setFuzzyInputSets(TriangularFuzzySets([(-90,10), (0,50), (40,90), (80,100), (90,140), (130,180), (170,270)]))
# fuzzy_alg.setFuzzyOutputSets(TriangularFuzzySets([(-30,-15), (-25,-5), (-10,0), (-5,5), (0,10), (5,25), (15,30)]))

# ruleMatrix = FileExtractor.ruleMatrix()

# solution = fuzzy_alg.run(MandaniInference(ruleMatrix), [47.5, 99], AggregateThenCentroid())
# solution2 = fuzzy_alg.run(MandaniInference(ruleMatrix), [47.5, 99], CentroidThenAggregate())
# print solution 
# print solution2 

# #######Fuzzy Algorithm Takagi-Sugeno

# Fitness.initGauss3Data("gauss3.dat")

# polyfitOrder = 30
# polyfit = np.polyfit(Fitness.gauss3[:,1], Fitness.gauss3[:,0], polyfitOrder)

# plt.plot(range(250), [sum([polyfit[polyfitOrder-i]*(j**i) for i in range(polyfitOrder+1)]) for j in range(250)], "b-")
# plt.show()

# print polyfit

# takagiInstance = TakagiSugenoInference(Fitness.takagiSugeno)

# print takagiInstance.fuzzyInterval
# print takagiInstance.ruleMatrix

# fuzzy_alg = Fuzzy()
# fuzzy_alg.setFuzzyInputSets(GaussianFuzzySets(takagiInstance.fuzzyInterval))

# solution = [fuzzy_alg.run(takagiInstance, [i]) for i in range(0,250)]

# ####### LSM Learning

# inputMatrix, outputMatrix = FileExtractor.LSMMatrices("pacientes.txt", "patologias.txt")

# A = outputMatrix*LA.pinv(inputMatrix)



# ####### Neural Network

inputMatrix, outputMatrix = FileExtractor.ANNMatrices("pacientes.txt", "patologias.txt")

######## Simple Perceptron Network

# accuracyVector = []
# classAccuracyVector = []
# its = 100

# for i in range(its):
# 	spn = SimplePerceptronNetwork(0.01, inputMatrix, outputMatrix, SignalActivation())
# 	accuracyVector.append(spn.trainAndTest())
# 	classAccuracyVector.append(spn.classAccuracy())
# 	print i

# plt.plot(range(its), accuracyVector, "b-")
# plt.plot([0,its], [np.min(accuracyVector), np.min(accuracyVector)], "r-")
# plt.plot([0,its], [np.max(accuracyVector), np.max(accuracyVector)], "r-")
# plt.plot([0,its], [np.mean(accuracyVector), np.mean(accuracyVector)], "g-")
# plt.plot(range(its), [np.mean(accuracyVector) - np.std(accuracyVector) for i in range(its)], "yo")
# plt.plot(range(its), [np.mean(accuracyVector) + np.std(accuracyVector) for i in range(its)], "yo")

# plt.show()

# for i in range(len(classAccuracyVector[0])):
# 	plt.plot(range(its), np.array(classAccuracyVector)[:,i])
# 	plt.show()

######## Multi-Layered Perceptron

accuracyVector = []
classAccuracyVector = []
its = 20

for i in range(its):
	mlp = MultiLayeredPerceptron(0.1, inputMatrix, outputMatrix, SigmoidActivation(), SigmoidActivation(), 20, DETraining(0.000, 0.01))
	accuracyVector.append(mlp.trainAndTest())
	classAccuracyVector.append(mlp.classAccuracy())
	print i

plt.plot(range(its), accuracyVector, "b-")
plt.plot([0,its], [np.min(accuracyVector), np.min(accuracyVector)], "r-")
plt.plot([0,its], [np.max(accuracyVector), np.max(accuracyVector)], "r-")
plt.plot([0,its], [np.mean(accuracyVector), np.mean(accuracyVector)], "g-")
plt.plot(range(its), [np.mean(accuracyVector) - np.std(accuracyVector) for i in range(its)], "yo")
plt.plot(range(its), [np.mean(accuracyVector) + np.std(accuracyVector) for i in range(its)], "yo")

plt.show()

for i in range(len(classAccuracyVector[0])):
	plt.plot(range(its), np.array(classAccuracyVector)[:,i])
	plt.show()

#print spn.classify([ -1.,   2.,   2.,   0.,   3.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,   2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,   0.,   0.,   0.,   1.,   0.,  55.])


######## Multi-Layered Perceptron for Regression

# Fitness.initGauss3Data("gauss3.dat")

# accuracyVector = []

# inputMatrix = [[i] for i in np.linspace(0,1, 250)]
# outputMatrix = [[i] for i in Fitness.gauss3[:,0]]

# mlp = MultiLayeredPerceptron(0.001, inputMatrix, outputMatrix, SigmoidActivation(), LinearActivation(), 100, OnlineTraining(0.00, 30))
# accuracyVector.append(mlp.trainAndTest())

# plt.plot(np.linspace(0,1, 250), [mlp.mode.forwardExecution([-1, i], mlp.hiddenPerceptrons, mlp.outputPerceptrons) for i in np.linspace(0,1, 250)], "b-")
# plt.show()






