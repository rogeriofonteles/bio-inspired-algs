import numpy as np
from metaHeuristics import MetaHeuristics

class EvolutionaryAlgorithm(MetaHeuristics):

	def __init__(self, populationNumber, dim, (x,y)):
		self.dom = (x,y)
		MetaHeuristics.__init__(self)
		MetaHeuristics.generateFuncPopulation(self, populationNumber, dim, (x,y))
		#MetaHeuristics.generateEquallySpreadPopulation(self, populationNumber, (x,y))	


	def selectionAlgorithm(self, selectionAlg):		
		selectionAlg.setParams(self)
		return selectionAlg.run()


	def crossoverAlgorithm(self, crossoverAlg):		
		crossoverAlg.setParams(self)
		return crossoverAlg.run()


	def mutationAlgorithm(self, mutationAlg):
		mutationAlg.setParams(self) 		
		return mutationAlg.run()
		

	def returnBest(self):
		return self.population[np.argmin([self.fitness(self.population[i]) for i in range(len(self.population))])]




