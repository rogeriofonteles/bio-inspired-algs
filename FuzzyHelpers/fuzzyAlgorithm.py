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

	def inferenceAlgorithm(self, inferenceAlg, pertinenceValues):
		inferenceAlg.setParams(pertinenceValues)
		return inferenceAlg.run()

	
	def defuzzyAlgorithm(self, defuzzyAlg):
		defuzzyAlg.setParams(self)
		return defuzzyAlg.run()


	def run(self, inferenceAlg, value, defuzzyAlg=None):		
		pertinenceValues = np.array([self.fuzzyInputSets[i].pertinence(value[i]) for i in range(len(self.fuzzyInputSets))])		
		print pertinenceValues
		inferedFuzzySets = self.inferenceAlgorithm(inferenceAlg, pertinenceValues)
		if defuzzyAlg is not None:
			centroid = self.defuzzyAlgorithm(defuzzyAlg)

		return centroid


	