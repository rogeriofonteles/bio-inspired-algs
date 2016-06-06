import numpy as np

class DESelection:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class DESelectionBest(DESelection):	

	def run(self):				
		return [(test if (self.genAlgObj.fitness(test) < self.genAlgObj.fitness(offspring)) else (offspring)) for [test, offspring] in zip(self.genAlgObj.population, self.genAlgObj.offsprings)]


		