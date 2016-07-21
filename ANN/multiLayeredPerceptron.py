import numpy as np
from neuralNetwork import *
import math


class MultiLayeredPerceptron(NeuralNetwork):
	
	def __init__(self, learningFactor, _inputMatrix, _outputMatrix, activationFuncHidden, activationFuncOutput, Q, _mode):		
		NeuralNetwork.__init__(self, _inputMatrix, _outputMatrix)	
		self.hiddenPerceptrons = [Perceptron(learningFactor, len(self.inputMatrix[0]), activationFuncHidden) for i in range(Q)]
		self.outputPerceptrons = [Perceptron(learningFactor, Q+1, activationFuncOutput) for i in range(len(self.outputMatrix[0]))]	
		self.mode = _mode


	def training(self, trainingInput, trainingOutput, hiddenPerceptrons, outputPerceptrons):
		self.mode.training(trainingInput, trainingOutput, hiddenPerceptrons, outputPerceptrons)

	
	def testing(self):
		for testI, testO in zip(self.testInput, self.testOutput):			
			perceptronOutput = self.classify(testI)
			if not 1 in perceptronOutput:
				classPerceptronOutput = len(testO)
			else:
				classPerceptronOutput = list(perceptronOutput).index(1)						
			
			self.confusionMatrix[list(testO).index(1)][classPerceptronOutput] += 1 


	def classify(self, testInput):		
		output = self.mode.forwardExecution(testInput, self.hiddenPerceptrons, self.outputPerceptrons)		
		return [(1 if output[i] > 0.5 else 0) for i in range(len(output))]


	def trainAndTest(self):
		self.trainingInput, self.testInput = self.separateTraining(self.inputMatrix)
		self.trainingOutput, self.testOutput = self.separateTraining(self.outputMatrix)			
		self.training(self.trainingInput, self.trainingOutput, self.hiddenPerceptrons, self.outputPerceptrons)
		self.testing()
		print self.confusionMatrix
		return self.accuracy()
