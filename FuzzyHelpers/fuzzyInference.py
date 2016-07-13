import numpy as np
import math 
from EAs.differentialEvolution import DifferentialEvolution
from EAs.MutationAlgs.DEMutation import *
from EAs.SelectionAlgs.DESelection import *
from EAs.CrossOverAlgs.DECrossOver import *
from FuzzyHelpers.fuzzySets import GaussianFuzzySets
from fileHelper import FileExtractor
from numpy import linalg as LA

class FuzzyInference():

	def __init__(self, ruleMatrix):
		self.ruleMatrix = ruleMatrix
		self.inferVector = list([])


	def setParams(self, pertValues):
		self.pertinenceValues = pertValues


	def pertinenceCombination(self, index, partialVector):				
		if index == len(self.pertinenceValues):						
			self.inferVector.append(np.array([zip(*partialVector)[0], np.min(zip(*partialVector)[1])]))
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


	def run(self, value):
		self.pertinenceCombination(0, [])					
		return np.array([(self.ruleMatrix[self.inferVector[i][0]], self.inferVector[i][1]) for i in range(len(self.inferVector))])



class TakagiSugenoInference(FuzzyInference):

	def __init__(self, fitness=None):
		FuzzyInference.__init__(self, []) 
		if fitness is not None:
			self.fuzzyInterval = self.discoverBestFuzzySets(fitness)		
			self.ruleMatrix, foo = self.makeRuleMatrix(self.fuzzyInterval, "gauss3.dat")	


	def discoverBestFuzzySets(self, fitness):
		dom = (1, 250)
		de = DifferentialEvolution(50, 30, dom)
		de.fitness(fitness)
		de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1Fuzzy())
		vectorFuzzySets = de.returnBest()
		de.fitness(vectorFuzzySets, True)
		return [x for x in [((vectorFuzzySets[i], vectorFuzzySets[i+1]) if vectorFuzzySets[(i-10)/2] > 125 else None) for i in range(10,30,2)] if x is not None]


	@staticmethod
	def makeRuleMatrix(intervals, dataFileName):		
		fuzzySets = GaussianFuzzySets(intervals)
		inferedValues = np.array([fuzzySets.pertinence(value) for value in range(0, 250)])			
		sumPertinences = [sum(inferedValues[i,:,1]) for i in range(0, 250)]

		X = [[(inferedValues[i][j][1]/sumPertinences[i] if inferedValues[i][j][1] > 0.000001 else 0) if j<len(inferedValues[i]) else (i*inferedValues[i][j-len(inferedValues[i])][1]/sumPertinences[i] if inferedValues[i][j-len(inferedValues[i])][1] > 0.000001 else 0) for j in range(2*len(inferedValues[i]))] for i in range(0, 250)]		
		pseudoInv = LA.pinv(X)
		yData = FileExtractor.fileData("gauss3.dat")[:,0]

		P = pseudoInv.dot(yData)
		return np.array([[P[i], P[i+len(P)/2]] for i in range(len(P)/2)]), inferedValues


	def run(self, value):				
		return sum([(self.pertinenceValues[0][i][1]/sum(self.pertinenceValues[0,:,1]))*(self.ruleMatrix[int(self.pertinenceValues[0][i][0])][1]*value[0]+self.ruleMatrix[int(self.pertinenceValues[0][i][0])][0]) for i in range(len(self.pertinenceValues))])
		
		
				

		
