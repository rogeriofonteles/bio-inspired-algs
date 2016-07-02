import numpy as np

class DESelection:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class DESelectionBest(DESelection):	

	def run(self):		
		finalPopulation = []
		for [test, offspring] in zip(self.genAlgObj.population, self.genAlgObj.offsprings):
			testFitness = self.genAlgObj.fitness(test)
			offsprinFitness = self.genAlgObj.fitness(offspring)
			if testFitness <= offsprinFitness:
				
				finalPopulation.append(test)
			else:
				
				finalPopulation.append(offspring)

	 	return finalPopulation
		
		


		