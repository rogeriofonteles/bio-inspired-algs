import numpy as np

class FileExtractor():

	@staticmethod
	def fileData(fileName, separator=None):		
		file = open(fileName)
		return np.array([map(float, line.strip().split(separator)) for line in file])


	@staticmethod
	def ruleMatrix():
		rulematrixFile = open("rulematrix.dat")
		return np.array([map(int, line.strip().split()) for line in rulematrixFile])


	@staticmethod
	def ANNMatrices(inputFileName, outputFileName):
		unarrangedInputMatrix = FileExtractor.fileData(inputFileName)
		unarrangedOutputMatrix = FileExtractor.fileData(outputFileName)

		inputMatrix = [unarrangedInputMatrix[:,i] for i in range(len(unarrangedInputMatrix[0]))]
		outputMatrix = [unarrangedOutputMatrix[:,i] for i in range(len(unarrangedInputMatrix[0]))]

		return np.array(inputMatrix), np.array(outputMatrix)

	@staticmethod
	def LSMMatrices(inputFileName, outputFileName):
		inputMatrix = FileExtractor.fileData(inputFileName)
		outputMatrix = FileExtractor.fileData(outputFileName)

		return np.array(inputMatrix), np.array(outputMatrix)

		
