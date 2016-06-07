import numpy as np

class DECrossOver:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class DECrossOverBin(DECrossOver):

	def __init__(self):
		self.prob_cross = 0.5
		self.fixed_position = 0

	def __elementCrossOver(self, i, j):
		return self.genAlgObj.parentsVector[i][j] if np.random.rand(1) < 0.5 else self.genAlgObj.population[i][j]

	def run(self):	
		crossOveredOffsprings = np.array([[self.__elementCrossOver(i, j) for j in xrange(len(self.genAlgObj.population[0]))] for i in xrange(len(self.genAlgObj.population))])
		crossOveredOffsprings[::, self.fixed_position] = self.genAlgObj.parentsVector[::, self.fixed_position]
		return crossOveredOffsprings