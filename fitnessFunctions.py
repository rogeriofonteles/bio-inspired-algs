import numpy as np
import math
from numpy import linalg as LA

class Fitness():

	aerodata = []
	w = 1

	@staticmethod
	def fitness1(x):		
		return 20 + x[0]**2 + x[1]**2 - 10*(math.cos(2*math.pi*x[0])+math.cos(2*math.pi*x[1]))

	@staticmethod
	def fitness2(x):		
		return x[0]**2 + x[1]**2

	@classmethod
	def initAeroData(cls):		
		aerodatFile = open("aerogerador.dat")		
		cls.aerodata = np.array([map(float, line.strip().split('\t')) for line in aerodatFile])				

	@classmethod
	def LSE(cls, beta):			
		squareError = (np.array([[data[0]**i for i in xrange(len(beta))] for data in cls.aerodata]).dot(np.array(beta)) - np.array([data[1] for data in cls.aerodata]).T)**2
		return squareError.sum()

	@classmethod
	def LSEwR(cls, beta):			
		squareError = (np.array([[data[0]**i for i in xrange(len(beta))] for data in cls.aerodata]).dot(np.array(beta)) - np.array([data[1] for data in cls.aerodata]).T)**2
		return squareError.sum() + cls.w*LA.norm(beta)**2
