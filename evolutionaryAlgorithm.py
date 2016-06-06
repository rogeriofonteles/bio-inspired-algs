import numpy as np
from metaHeuristics import MetaHeuristics

class EvolutionaryAlgorithm(MetaHeuristics):

	def __init__(self, populationNumber, (x,y)):
		MetaHeuristics.__init__(self)
		MetaHeuristics.generateFuncPopulation(self, populationNumber, (x,y))	

	def selectionAlgorithm(self, selectionAlg):		
		selectionAlg.setParams(self)
		return selectionAlg.run()

	def crossoverAlgorithm(self, crossoverAlg):		
		crossoverAlg.setParams(self)
		return crossoverAlg.run()

	def mutationAlgorithm(self, mutationAlg):
		mutationAlg.setParams(self) 		
		return mutationAlg.run()




