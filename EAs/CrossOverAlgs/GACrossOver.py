import numpy as np

class GACrossOver:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj	

class GACrossOverLinearOperator(GACrossOver):

	def __weigthOffspringGeneration(self, parent1, parent2):
		offspringCandidates = [parent1, parent2, parent1+parent2, parent1*1.5-parent2*0.5, parent1*(-0.5)+parent2*1.5]		
		del offspringCandidates[np.argmax(np.array([self.genAlgObj.fitness([x, y]) for [x, y] in offspringCandidates]))]				
		del offspringCandidates[np.argmax(np.array([self.genAlgObj.fitness([x, y]) for [x, y] in offspringCandidates]))]
		del offspringCandidates[np.argmax(np.array([self.genAlgObj.fitness([x, y]) for [x, y] in offspringCandidates]))]
		return offspringCandidates

	def run(self):	
		unorganizedOffsprings = np.ravel(np.array([self.__weigthOffspringGeneration(parent1, parent2) for (parent1, parent2) in self.genAlgObj.parentsVector]))		
		return unorganizedOffsprings.reshape(len(unorganizedOffsprings)/2, 2)


