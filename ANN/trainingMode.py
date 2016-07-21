import numpy as np
import math 
from EAs.differentialEvolution import DifferentialEvolution
from EAs.MutationAlgs.DEMutation import *
from EAs.SelectionAlgs.DESelection import *
from EAs.CrossOverAlgs.DECrossOver import *


class Training():

	def __init__(self, _beta, _threshold):
		self.beta = _beta
		self.errorThreshold = _threshold


	def forwardExecution(self, trainingExample, hiddenPerceptrons, outputPerceptrons):
		hiddenOutputs = [neuron.testing(trainingExample) for neuron in hiddenPerceptrons]
		hiddenOutputs.insert(0, -1)
		return [neuron.testing(hiddenOutputs) for neuron in outputPerceptrons]


	def updateLearningFactor(self, hiddenPerceptrons, outputPerceptrons):
		for i in range(len(hiddenPerceptrons)):
			hiddenPerceptrons[i].eta = hiddenPerceptrons[i].eta/float(1+hiddenPerceptrons[i].eta*self.beta)
		for i in range(len(outputPerceptrons)):
			outputPerceptrons[i].eta = outputPerceptrons[i].eta/float(1+outputPerceptrons[i].eta*self.beta)



class OnlineTraining(Training):

	def __init__(self, _beta, _threshold):
		Training.__init__(self, _beta, _threshold)


	def training(self, trainingInput, trainingOutput, hiddenPerceptrons, outputPerceptrons):		
		count = 0
		error = self.errorThreshold + 1
		permutedIndex = range(len(trainingInput))
		while(math.fabs(error) > self.errorThreshold):
			error=0			
			permutedIndex = np.random.permutation(len(trainingInput))			
			#permutedIndex = range(len(trainingInput))
			trainingInputNow = [trainingInput[k] for k in permutedIndex]
			trainingOutputNow = [trainingOutput[k] for k in permutedIndex]	
			for trainingExample, targetOutput in zip(trainingInputNow, trainingOutputNow):				
				outputs = self.forwardExecution(trainingExample, hiddenPerceptrons, outputPerceptrons)			
				self.backPropagation(trainingExample, targetOutput, outputs, hiddenPerceptrons, outputPerceptrons)			
				error += sum([(targetOutput[i]-outputs[i])**2 for i in range(len(outputs))])
			error = error/float(len(trainingInputNow)*len(trainingOutput[0]))			
			self.updateLearningFactor(hiddenPerceptrons, outputPerceptrons)
			count+=1
			#print count, error#, [hiddenPerceptron.w for hiddenPerceptron in hiddenPerceptrons], [outputPerceptrons[0].w]
	

	def backPropagation(self, trainingExample, targetOutput, outputs, hiddenPerceptrons, outputPerceptrons):
		errorOutputLayer = [(targetOutput[i] - outputs[i])*outputPerceptrons[i].activationFunction.derivative(outputs[i]) for i in range(len(outputs))]			
		errorHiddenLayer = [hiddenPerceptrons[i].activationFunction.derivative(hiddenPerceptrons[i].testing(trainingExample))*sum([errorOutputLayer[j]*outputPerceptrons[j].w[i+1] for j in range(len(errorOutputLayer))]) for i in range(len(hiddenPerceptrons))]		
		hiddenOutputs = [-1]
		[hiddenOutputs.append(neuron.testing(trainingExample)) for neuron in hiddenPerceptrons]
		[outputPerceptrons[i].training(hiddenOutputs, errorOutputLayer[i]) for i in range(len(outputPerceptrons))]
		[hiddenPerceptrons[i].training(trainingExample, errorHiddenLayer[i]) for i in range(len(hiddenPerceptrons))]
		
		


	

class OfflineTraining(Training):

	def __init__(self, _beta, _threshold):
		Training.__init__(self, _beta, _threshold)

	
	def training(self, trainingInput, trainingOutput, hiddenPerceptrons, outputPerceptrons):		
		count = 0
		error = self.errorThreshold + 1
		while(math.fabs(error) > self.errorThreshold):
			error=0			
			permutedIndex = np.random.permutation(len(trainingInput))
			#permutedIndex = range(len(trainingInput))
			trainingInputNow = [trainingInput[k] for k in permutedIndex]
			trainingOutputNow = [trainingOutput[k] for k in permutedIndex]	
			errorOutputTensor = np.zeros((len(outputPerceptrons[0].w), len(outputPerceptrons))) 
			errorHiddenTensor = np.zeros((len(hiddenPerceptrons[0].w), len(hiddenPerceptrons)))
			for trainingExample, targetOutput in zip(trainingInputNow, trainingOutputNow):				
				outputs = self.forwardExecution(trainingExample, hiddenPerceptrons, outputPerceptrons)			
				_errorOutputTensor, _errorHiddenTensor = self.backPropagation(trainingExample, targetOutput, outputs, hiddenPerceptrons, outputPerceptrons)			
				errorOutputTensor += _errorOutputTensor
				errorHiddenTensor += _errorHiddenTensor										
			self.weightUpdate(hiddenPerceptrons, outputPerceptrons, errorOutputTensor, errorHiddenTensor)
			self.updateLearningFactor(hiddenPerceptrons, outputPerceptrons)

			for trainingExample, targetOutput in zip(trainingInputNow, trainingOutputNow):				
				outputs = self.forwardExecution(trainingExample, hiddenPerceptrons, outputPerceptrons)
				error += sum([(targetOutput[i]-outputs[i])**2 for i in range(len(outputs))])
			error = error/float(len(trainingInputNow)*len(trainingOutput[0]))	

			count+=1
			#print count, error#, [hiddenPerceptron.w for hiddenPerceptron in hiddenPerceptrons], [outputPerceptrons[0].w]


	def updateLearningFactor(self, hiddenPerceptrons, outputPerceptrons):
		for i in range(len(hiddenPerceptrons)):
			hiddenPerceptrons[i].eta = hiddenPerceptrons[i].eta/float(1+hiddenPerceptrons[i].eta*self.beta)
		for i in range(len(outputPerceptrons)):
			outputPerceptrons[i].eta = outputPerceptrons[i].eta/float(1+outputPerceptrons[i].eta*self.beta)


	def backPropagation(self, trainingExample, targetOutputVector, outputs, hiddenPerceptrons, outputPerceptrons):
		GV = [neuron.testing(trainingExample) for neuron in hiddenPerceptrons]
		GV.insert(0, -1)
		errorOutputLayer = [(targetOutput - NNoutput)*neuron.activationFunction.derivative(NNoutput) for (NNoutput, neuron, targetOutput) in zip(outputs, outputPerceptrons, targetOutputVector)]			
		errorOutputTensor = np.outer(GV, errorOutputLayer)

		errorHiddenLayer = [hiddenPerceptrons[i].activationFunction.derivative(hiddenPerceptrons[i].testing(trainingExample))*sum([neuronError*neuron.w[i+1] for neuronError, neuron in zip(errorOutputLayer, outputPerceptrons)]) for i in range(len(hiddenPerceptrons))]		
		errorHiddenTensor = np.outer(trainingExample, errorHiddenLayer)
		return errorOutputTensor, errorHiddenTensor


	def weightUpdate(self, hiddenPerceptrons, outputPerceptrons, errorOutputTensor, errorHiddenTensor):
		for i in range(len(outputPerceptrons)):
			outputPerceptrons[i].training(np.transpose(errorOutputTensor)[i], 1)
		for i in range(len(hiddenPerceptrons)):
			hiddenPerceptrons[i].training(np.transpose(errorHiddenTensor)[i], 1)




class DETraining(Training):

	def __init__(self, _beta, _threshold):
		self.beta = _beta
		self.errorThreshold = _threshold


	def training(self, trainingInput, trainingOutput, hiddenPerceptrons, outputPerceptrons):
		self.hiddenPerceptrons = hiddenPerceptrons
		self.outputPerceptrons = outputPerceptrons
		self.trainingInput = trainingInput
		self.trainingOutput = trainingOutput

		dom = (0, 1000)

		de = DifferentialEvolution(100, len(hiddenPerceptrons[0].w)*len(hiddenPerceptrons)+len(outputPerceptrons[0].w)*len(outputPerceptrons), dom)
		de.fitness(self.fitness)

		de.run(DESelectionBest(), DECrossOverBin(), DEMutationRand1())
		self.fitness(de.returnBest())

		hiddenPerceptrons = self.hiddenPerceptrons
		outputPerceptrons = self.outputPerceptrons

	
	def fitness(self, chromossome):
		error = 0
		i=0
		numHidden=0
		numOutput=0
		while i<len(chromossome):
			if(numHidden < len(self.hiddenPerceptrons)):
				j = i+len(self.hiddenPerceptrons[numHidden].w)				
				self.hiddenPerceptrons[numHidden].w=chromossome[i:j]
				numHidden += 1
			else:
				j = i+len(self.outputPerceptrons[numOutput].w)
				self.outputPerceptrons[numOutput].w=chromossome[i:j]
				numOutput+=1
			i = j

		for trainingExample, targetOutput in zip(self.trainingInput, self.trainingOutput):				
			outputs = self.forwardExecution(trainingExample, self.hiddenPerceptrons, self.outputPerceptrons)
			error += sum([(targetOutput[i]-outputs[i])**2 for i in range(len(outputs))])
		return error/float(len(self.trainingInput)*len(self.trainingOutput[0]))

