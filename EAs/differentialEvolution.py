import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm as evolAlg

class DifferentialEvolution(evolAlg):

	def __init__(self, populationNumber, dim, (x,y)):
		evolAlg.__init__(self, populationNumber, dim, (x,y))		


	def run(self, selectionAlg, crossoverAlg, mutationAlg):		
		for num in range(1,200):			
			self.parentsVector = self.mutationAlgorithm(mutationAlg)							
			self.offsprings = self.crossoverAlgorithm(crossoverAlg)						
			self.population = self.selectionAlgorithm(selectionAlg)		
			print num

			self.plot[0].saveForPlot(self.population, self.fitness, "best")
			self.plot[1].saveForPlot(self.population, self.fitness, "average")			