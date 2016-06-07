import numpy as np
from evolutionaryAlgorithm import EvolutionaryAlgorithm as evolAlg

class GeneticAlgorithm(evolAlg):

	def __init__(self, populationNumber, (x,y)):
		evolAlg.__init__(self, populationNumber, (x,y))		

	def fitness(self, fitness):
		self.fitness = fitness	

	def run(self, selectionAlg, crossoverAlg, mutationAlg):		
		for num in range(1,200):			
			self.parentsVector = self.selectionAlgorithm(selectionAlg)							
			self.offsprings = self.crossoverAlgorithm(crossoverAlg)						
			self.population =self.mutationAlgorithm(mutationAlg)

		#print np.array([self.fitness(x,y) for [x,y] in self.population])
			


	





