import numpy as np

class Defuzzification():
	
	def setParams(self, _inferedInterval, _fuzzyOutputSet):
		self.inferedInterval = _inferedInterval
		self.fuzzyOutputSet = _fuzzyOutputSet


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
			for i in range(self.fuzzyOutputSet.setIntervals[0][0], self.fuzzyOutputSet.setIntervals[len(self.fuzzyOutputSet.setIntervals)-1][1], 1):				
				pertinenceOutput = np.array([[pertinenceSet[0], self.testIntervalTest(pertinenceSet)] for pertinenceSet in np.array(self.fuzzyOutputSet.pertinence(i))])																			
				if len(pertinenceOutput) != 0:									
					aggregateSet.append([i, pertinenceOutput[np.argmax(pertinenceOutput[:,0])][1] if len(pertinenceOutput) != 0 else 0])
			return np.array(aggregateSet)



class AggregateThenCentroid(Defuzzification):

	def run(self):		
		return self.centroid(self.aggregate())

	def centroid(self, _set):
		return np.dot(_set[:,0],_set[:,1])/sum(_set[:,1])

class CentroidThenAggregate(Defuzzification):

	def run(self):
		return self.aggregate(self.centroid())

	def centroid(self):
		return np.array([((self.fuzzyOutputSet.setIntervals[int(interval[0])][0]+self.fuzzyOutputSet.setIntervals[int(interval[0])][1])/float(2), interval[1]/float(2)) for interval in self.inferedInterval])
		
		



