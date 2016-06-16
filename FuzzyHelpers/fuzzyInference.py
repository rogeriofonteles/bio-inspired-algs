import numpy as np
import math 

class FuzzyInference():

	def __init__(self, ruleMatrix):
		self.ruleMatrix = ruleMatrix
		self.inferVector = list([])


	def setParams(self, pertValues):
		self.pertinenceValues = pertValues


	def pertinenceCombination(self, index, partialVector):				
		if index == len(self.pertinenceValues):						
			self.inferVector.append([zip(*partialVector)[0], np.min(zip(*partialVector)[1])])
		else:				
			for i in range(len(self.pertinenceValues[index])):				
				partialVectorTemp = list(partialVector)				
				if (index==0):
					self.pertinenceCombination(index+1, [self.pertinenceValues[index][i]])
				else:
					partialVectorTemp.append(self.pertinenceValues[index][i])
					self.pertinenceCombination(index+1, partialVectorTemp)				



class MandaniInference(FuzzyInference):
	
	def __init__(self, ruleMatrix):
		FuzzyInference.__init__(self, ruleMatrix) 


	def run(self):
		self.pertinenceCombination(0, [])
		print self.inferVector
		
				

		
