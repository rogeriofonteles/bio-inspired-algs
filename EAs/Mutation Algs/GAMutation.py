import numpy as np
from operator import add

class GAMutation:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class GAMutationUniform(GAMutation):

	def run(self):
		mutationProbabilities = np.random.rand(len(self.genAlgObj.offsprings))

class GAMutationGaussian(GAMutation):

	def run(self):
		gaussianAdds = np.array(np.random.normal(0, 0.01, len(self.genAlgObj.offsprings)*2)).reshape(len(self.genAlgObj.offsprings), 2)		
		return np.array(self.genAlgObj.offsprings)+gaussianAdds
		