import numpy as np
from neuralNetwork import *


class SimplePerceptronNetwork(NeuralNetwork):
	
	def __init__(self, learningFactor, _inputMatrix, _outputMatrix, activationFunc):		
		NeuralNetwork.__init__(self, _inputMatrix, _outputMatrix, activationFunc)	
		self.perceptrons = [Perceptron(learningFactor, len(self.inputMatrix[0]), self.activationFunction) for i in range(len(self.outputMatrix[0]))]	


	def training(self):
		for i in range(200):
			permutedIndex = np.random.permutation(len(self.trainingInput))
			trainingInputNow = [self.trainingInput[k] for k in permutedIndex]
			trainingOutputNow = [self.trainingOutput[k] for k in permutedIndex]
			for trainingExample, targetOutput in zip(trainingInputNow, trainingOutputNow):				
				for j in range(len(self.perceptrons)):				
					localGradient = targetOutput[j] - self.perceptrons[j].activationFunction.run(self.perceptrons[j].w.dot(np.array(trainingExample)))
					self.perceptrons[j].training(trainingExample, localGradient)


	def testing(self):
		for testI, testO in zip(self.testInput, self.testOutput):
			perceptronOutput = self.classify(testI)
			if not 1 in perceptronOutput:
				classPerceptronOutput = len(testO)
			else:
				classPerceptronOutput = list(perceptronOutput).index(1)						
			
			self.confusionMatrix[list(testO).index(1)][classPerceptronOutput] += 1 


	def classify(self, testInput):		
		return np.array([self.perceptrons[i].testing(testInput) for i in range(len(self.perceptrons))])


	def trainAndTest(self):
		self.trainingInput, self.testInput = self.separateTraining(self.inputMatrix)
		self.trainingOutput, self.testOutput = self.separateTraining(self.outputMatrix)			
		self.training()
		self.testing()
		return self.accuracy()


