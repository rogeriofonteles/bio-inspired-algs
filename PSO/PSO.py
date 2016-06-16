import numpy as np
from metaHeuristics import MetaHeuristics

class PSO(MetaHeuristics):

	def __init__(self, populationNumber, (x, y), _c1, _c2):
		MetaHeuristics.__init__(self)
		# MetaHeuristics.generateFuncPopulation(self, populationNumber, (x,y))	
		MetaHeuristics.generateEquallySpreadPopulation(self, populationNumber, (x,y))	

		self.particleBest = []
		self.particleVelocities = np.zeros(self.population.shape)
		self.globalBest = []
		self.c1 = _c1
		self.c2 = _c2		
		self.w = 0.3

	def velocityUpdate(self):		
		self.particleVelocities = np.array([(self.w*self.particleVelocities[i] + self.c1*np.random.random_sample()*(self.particleBest[i] - self.population[i]) + self.c2*np.random.random_sample()*(self.globalBest - self.population[i])) for i in xrange(len(self.population))])

	def positionUpdate(self):		
		self.population = self.population + self.particleVelocities

	def particleBestSelection(self):
		if len(self.particleBest) == 0:
			self.particleBest = self.population			
		else:
			self.particleBest = np.array([self.particleBest[i] if self.fitness(self.particleBest[i]) < self.fitness(self.population[i]) else self.population[i] for i in xrange(len(self.population))])

	def globalBestSelection(self):
		self.globalBest = self.particleBest[np.argmin([self.fitness(particleBestValue) for particleBestValue in self.particleBest])]

	def run(self):
		for i in xrange(20):
			self.particleBestSelection()
			self.globalBestSelection()
			self.velocityUpdate()
			self.positionUpdate()




