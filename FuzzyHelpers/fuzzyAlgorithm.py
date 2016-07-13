import numpy as np
from plotFunctions import Plot
from fuzzyInference import *

class Fuzzy():

	def __init__(self):
		quant = 1
		self.fuzzyInputSets = []
		self.fuzzyOutputSet = []
		self.rules = []		
		self.plot = [Plot() for i in range(quant)]

#Setting Functions####################

	def setFuzzyInputSets(self, fuzzySet):		
		self.fuzzyInputSets.append(fuzzySet)

	
	def setFuzzyOutputSets(self, sets):
		self.fuzzyOutputSet.append(sets)

	
	def setRules(self, rules):
		self.rules = rules

#Algorithms###########################

	def inferenceAlgorithm(self, value, inferenceAlg, pertinenceValues):
		inferenceAlg.setParams(pertinenceValues)
		return inferenceAlg.run(value)

	
	def defuzzyAlgorithm(self, defuzzyAlg, inferedFuzzySets):
		defuzzyAlg.setParams(inferedFuzzySets, self.fuzzyOutputSet)
		return defuzzyAlg.run()


	def run(self, inferenceAlg, value, defuzzyAlg=None):		
		self.plot[0].showFuzzySets(self.fuzzyInputSets)
		pertinenceValues = np.array([self.fuzzyInputSets[i].pertinence(value[i]) for i in range(len(self.fuzzyInputSets))])		
		self.plot[0].showPertinence(pertinenceValues, self.fuzzyInputSets, value)
		inferedFuzzySets = self.inferenceAlgorithm(value, inferenceAlg, pertinenceValues)
		print isinstance(inferenceAlg, TakagiSugenoInference)
		if not isinstance(inferenceAlg, TakagiSugenoInference):				
			self.plot[0].showInference(inferedFuzzySets, self.fuzzyOutputSet)

		if defuzzyAlg is not None:
			centroid = self.defuzzyAlgorithm(defuzzyAlg, inferedFuzzySets)
			self.plot[0].showCentroid(centroid, inferedFuzzySets, self.fuzzyOutputSet)
			return centroid
		else:
			self.plot[0].showRects(inferenceAlg.ruleMatrix, self.fuzzyInputSets)		
			return inferedFuzzySets


	