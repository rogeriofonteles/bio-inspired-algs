import numpy as np
import random as rd

class DEMutation:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class DEMutationRand1(DEMutation):

	def __init__(self):
		self.factor = 0.5

	def run(self):
		indexMatrix = np.array([rd.sample(xrange(len(self.genAlgObj.population)), 3) for i in xrange(len(self.genAlgObj.population))])	
		trials = np.array([(self.genAlgObj.population[x1] + self.factor*(self.genAlgObj.population[x2] - self.genAlgObj.population[x3])) for [x1,x2,x3] in indexMatrix])
		return trials


class DEMutationRand1Fuzzy(DEMutation):

	def __init__(self):
		self.factor = 0.5

	def controlledSum(self, x1, x2, x3, eta):
		result = x1+eta*(x2-x3)
		return [result[i] if result[i] > 0 else 0.1 for i in range(len(result))]


	def run(self):
		indexMatrix = np.array([rd.sample(xrange(len(self.genAlgObj.population)), 3) for i in xrange(len(self.genAlgObj.population))])	
		trials = np.array([self.controlledSum(self.genAlgObj.population[x1], self.genAlgObj.population[x2], self.genAlgObj.population[x3], self.factor) for [x1,x2,x3] in indexMatrix])
		return trials
		