import numpy as np
from neuralNetwork import *
import math


class MultiLayeredPerceptron(NeuralNetwork):
	
	def __init__(self, learningFactor, _inputMatrix, _outputMatrix, activationFunc, Q):		
		NeuralNetwork.__init__(self, _inputMatrix, _outputMatrix, activationFunc)	
		self.hiddenPerceptrons = [Perceptron(learningFactor, len(self.inputMatrix[0]), self.activationFunction) for i in range(Q)]
		self.outputPerceptrons = [Perceptron(learningFactor, Q+1, self.activationFunction) for i in range(len(self.outputMatrix[0]))]	
			

	def training(self):		
		error=251
		count = 0
		while(math.fabs(error) > 50):
			error=0			
			permutedIndex = np.random.permutation(len(self.trainingInput))
			trainingInputNow = [self.trainingInput[k] for k in permutedIndex]
			trainingOutputNow = [self.trainingOutput[k] for k in permutedIndex]			
			for trainingExample, targetOutput in zip(trainingInputNow, trainingOutputNow):				
				outputs = self.forwardExecution(trainingExample)			
				# classifiedOutputs = self.classify(trainingExample)
				error += sum([math.fabs(targetOutput[i]-outputs[i]) for i in range(len(outputs))])				
				self.backPropagation(trainingExample, targetOutput, outputs)
			count+=1
			print count, error


	def forwardExecution(self, trainingExample):
		hiddenOutputs = [neuron.testing(trainingExample) for neuron in self.hiddenPerceptrons]
		hiddenOutputs.insert(0, -1)
		return [neuron.testing(hiddenOutputs) for neuron in self.outputPerceptrons]


	def backPropagation(self, trainingExample, targetOutputVector, outputs):
		errorOutputLayer = [(targetOutput - NNoutput)*neuron.activationFunction.derivative(NNoutput) for (NNoutput, neuron, targetOutput) in zip(outputs, self.outputPerceptrons, targetOutputVector)]			
		errorHiddenLayer = [self.hiddenPerceptrons[i].activationFunction.derivative(self.hiddenPerceptrons[i].testing(trainingExample))*sum([neuronError*neuron.w[i+1] for neuronError, neuron in zip(errorOutputLayer, self.outputPerceptrons)]) for i in range(len(self.hiddenPerceptrons))]		
		[neuron.training(trainingExample, errorHiddenNeuron) for (neuron, errorHiddenNeuron) in zip(self.hiddenPerceptrons, errorHiddenLayer)]
		hiddenOutputs = [neuron.testing(trainingExample) for neuron in self.hiddenPerceptrons]
		hiddenOutputs.insert(0, -1)		
		[neuron.training(hiddenOutputs, errorOutputNeuron) for (neuron, errorOutputNeuron) in zip(self.outputPerceptrons, errorOutputLayer)]	


	def testing(self):
		for testI, testO in zip(self.testInput, self.testOutput):			
			perceptronOutput = self.classify(testI)
			#print perceptronOutput, testO
			if not 1 in perceptronOutput:
				classPerceptronOutput = len(testO)
			else:
				classPerceptronOutput = list(perceptronOutput).index(1)						
			
			self.confusionMatrix[list(testO).index(1)][classPerceptronOutput] += 1 


	def classify(self, testInput):		
		output = self.forwardExecution(testInput)		
		return [(1 if output[i] > 0.5 else 0) for i in range(len(output))]


	def trainAndTest(self):
		self.trainingInput, self.testInput = self.separateTraining(self.inputMatrix)
		self.trainingOutput, self.testOutput = self.separateTraining(self.outputMatrix)			
		self.training()
		self.testing()
		print self.confusionMatrix
		return self.accuracy()
