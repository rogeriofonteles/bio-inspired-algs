import numpy as np

class MetaHeuristics:

	def __init__(self):
		self.population = []	
		self.offsprings = []
		self.parentsVector = []

	def generateFuncPopulation(self, populationNumber, dim, (d1,d2)):
		self.population = np.random.rand(populationNumber, dim)*(d2-d1)+np.ones((populationNumber, dim))*d1

	def generateEquallySpreadPopulation(self, maskSize, (d1,d2)):
		xv, yv = np.meshgrid(np.arange(d1,d2,(d2-d1)/float(maskSize)), np.arange(d1,d2,(d2-d1)/float(maskSize)))				
		flattedPoints = np.array(np.ravel([zip(x,y) for (x, y) in zip(xv,yv)]))
		self.population = np.array([(flattedPoints[i], flattedPoints[i+1]) for i in np.arange(0, len(flattedPoints), 2)])

	def fitness(self, fitness):
		self.fitness = fitness

	