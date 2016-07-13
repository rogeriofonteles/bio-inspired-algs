import numpy as np
from plotFunctions import Plot

class Defuzzification():
	
	def setParams(self, _inferedInterval, _fuzzyOutputSet):
		self.inferedInterval = Defuzzification.unitedInference(_inferedInterval)		
		self.fuzzyOutputSet = _fuzzyOutputSet


	@classmethod
	def unitedInference(cls, inferedInterval):
		unitedInf = np.array(sorted(inferedInterval, key=lambda inference: inference[0]))		
		correctedInf = []
		for inf in unitedInf:
			try:
				i = [int(element) for element in np.array(correctedInf)[:,0]].index(inf[0])									
				if unitedInf[i][1] < inf[1]:
					unitedInf[i][1] = inf[1]				
			except Exception:
			 	correctedInf.append(inf)
			print correctedInf

		return np.array(correctedInf)			 	
			 

	def testIntervalTest(self, pertinenceSet):
		if pertinenceSet[0] not in self.inferedInterval[:,0]:
			return 0
		else:
			return pertinenceSet[1] if pertinenceSet[1] < self.inferedInterval[list(self.inferedInterval[:,0]).index(pertinenceSet[0])][1] else self.inferedInterval[list(self.inferedInterval[:,0]).index(pertinenceSet[0])][1]


	def aggregate(self, _set=None):			

		if(_set is not None):				
			return np.dot(_set[:,0],_set[:,1])/sum(_set[:,1])
		else:
			aggregateSet = list([])
			for i in range(self.fuzzyOutputSet[0].setIntervals[0][0], self.fuzzyOutputSet[0].setIntervals[len(self.fuzzyOutputSet[0].setIntervals)-1][1], 1):				
				pertinenceOutput = np.array([self.testIntervalTest(pertinenceSet) for pertinenceSet in np.array(self.fuzzyOutputSet[0].pertinence(i))])																			
				if len(pertinenceOutput) != 0:									
					aggregateSet.append([i, np.max(pertinenceOutput) if len(pertinenceOutput) != 0 else 0])
			return np.array(aggregateSet)



class AggregateThenCentroid(Defuzzification):

	def run(self):		
		return self.centroid(self.aggregate())

	def centroid(self, _set):
		print _set
		return np.dot(_set[:,0],_set[:,1])/sum(_set[:,1])

class CentroidThenAggregate(Defuzzification):

	def run(self):
		centroids = self.centroid()
		print centroids[:,0]
		print centroids[:,1]
		Plot.showTrapezoids(self.fuzzyOutputSet, centroids)
		print centroids
		return self.aggregate(centroids)


	def centroid(self):
		return np.array([((self.fuzzyOutputSet[0].setIntervals[int(interval[0])][0]+self.fuzzyOutputSet[0].setIntervals[int(interval[0])][1])/float(2), interval[1]/float(2)) for interval in self.inferedInterval])
		
		



