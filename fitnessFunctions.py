import numpy as np
import math
import plotting as plot
import matplotlib.pyplot as plt
from numpy import linalg as LA
from fileHelper import FileExtractor
from FuzzyHelpers.fuzzyAlgorithm import *
from FuzzyHelpers.fuzzyInference import TakagiSugenoInference


class Fitness():

	aerodata = []
	gauss3 = []
	w = 1

	@staticmethod
	def fitness1(x):
		return 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*math.pi*x[0])+math.cos(2*math.pi*x[1]))

	@staticmethod
	def fitness2(x):
		return x[0]**2 + x[1]**2

	@classmethod
	def initAeroData(cls, fileName):
		cls.aerodata = FileExtractor.fileData(fileName, "\t")		

	@classmethod
	def initGauss3Data(cls, fileName):
		cls.gauss3 = FileExtractor.fileData(fileName)		

	@classmethod
	def LSE(cls, beta):		
		squareError = np.array([(sum([beta[i]*(data[0]**i) for i in range(len(beta))]) - data[1])**2 for data in cls.aerodata])
		return squareError.sum()

	@classmethod
	def LSEwR(cls, beta):
		squareError = np.array([(sum([beta[i]*(data[0]**i) for i in range(len(beta))]) - data[1])**2 for data in cls.aerodata])
		return squareError.sum() + cls.w*LA.norm(beta)**2

	@classmethod
	def takagiSugeno(cls, chromossome, is_print=None):		
		intervals = np.array([((chromossome[2*i+len(chromossome)/3], chromossome[2*i+(len(chromossome)/3)+1]) if chromossome[i] > 125 else None) for i in range(len(chromossome)/3)])
		intervals = np.array([x for x in intervals if x is not None])				

		if len(intervals) == 0:
			return sum([(0 - cls.gauss3[i][0])**2 for i in range(0,250)])
		else:				
			sugenoRuleMatrix, inferedValues = TakagiSugenoInference.makeRuleMatrix(intervals, "gauss3.dat")	
			sumPertinences = [sum(inferedValues[i][:,1]) for i in range(0, 250)]				

			result = [sum([((inferedValues[i][j][1]/sumPertinences[i])*(sugenoRuleMatrix[inferedValues[i][j][0]][1]*i+sugenoRuleMatrix[inferedValues[i][j][0]][0]) if sumPertinences[i] > 0 else 0) for j in range(len(inferedValues[i]))]) for i in range(0, 250)]	
			
			if is_print:
				plt.plot(range(0, 250), result, "b-")		
				plt.plot(range(0, 250), cls.gauss3[:,0], "rx")		
				plt.show()
			
			return sum([(result[i] - cls.gauss3[i][0])**2 for i in range(len(inferedValues))])


	@staticmethod
	def gaussian(value, mean, var):
		return math.e**(-(1/float(2))*((value-mean)**2)/float(var))





