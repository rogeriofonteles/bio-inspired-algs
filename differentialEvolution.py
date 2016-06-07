import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm as evolAlg

class DifferentialEvolution(evolAlg):

	def __init__(self, populationNumber, dim, (x,y)):
		evolAlg.__init__(self, populationNumber, dim, (x,y))		

	def fitness(self, fitness):
		self.fitness = fitness	

	def run(self, selectionAlg, crossoverAlg, mutationAlg):		
		for num in range(1,1000):			
			self.parentsVector = self.mutationAlgorithm(mutationAlg)							
			self.offsprings = self.crossoverAlgorithm(crossoverAlg)						
			self.population = self.selectionAlgorithm(selectionAlg)			
			

		#print np.array([self.fitness(x,y) for [x,y] in self.population])
			
