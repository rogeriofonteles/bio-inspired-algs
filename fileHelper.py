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