import numpy as np

class Fuzzy():

	def __init__(self):
		self.fuzzyInputSets = []
		self.fuzzyOutputSet = []
		self.rules = []		

#Setting Functions####################

	def setFuzzyInputSets(self, fuzzySet):		
		self.fuzzyInputSets.append(fuzzySet)

	
	def setFuzzyOutputSets(self, sets):
		self.fuzzyOutputSet = sets

	
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
		pertinenceValues = np.array([self.fuzzyInputSets[i].pertinence(value[i]) for i in range(len(self.fuzzyInputSets))])				
		inferedFuzzySets = self.inferenceAlgorithm(value, inferenceAlg, pertinenceValues)		
		if defuzzyAlg is not None:
			return self.defuzzyAlgorithm(defuzzyAlg, inferedFuzzySets)
		else:
			return inferedFuzzySets


	