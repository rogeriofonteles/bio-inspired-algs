import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm as evolAlg

class GeneticAlgorithm(evolAlg):

	def __init__(self, populationNumber, dim, (x,y)):
		evolAlg.__init__(self, populationNumber, dim, (x,y))		
	

	def run(self, selectionAlg, crossoverAlg, mutationAlg):		
		for num in range(1,200):			
			self.parentsVector = self.selectionAlgorithm(selectionAlg)							
			self.offsprings = self.crossoverAlgorithm(crossoverAlg)						
			self.population =self.mutationAlgorithm(mutationAlg)

			self.plot[0].saveForPlot(self.population, self.fitness, "best")
			self.plot[1].saveForPlot(self.population, self.fitness, "average")