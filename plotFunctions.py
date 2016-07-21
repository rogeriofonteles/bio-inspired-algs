import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from FuzzyHelpers.fuzzySets import *
import numpy as np


class Plot:

	def __init__(self):
		self.xValues = []
		self.yValues = []
		self.mode = {"best":0, "average":1}

	def saveForPlot(self, population, fitness, _mode):
		if self.mode[_mode] == 1: 
			fitnessValues = [fitness(chromossome) for chromossome in population]
			self.yValues.append(sum(fitnessValues)/float(len(fitnessValues)))
		else:			
			self.yValues.append(min([fitness(chromossome) for chromossome in population]))

	@staticmethod
	def show(*plotList):
		[plt.plot(range(len(plot.yValues)), plot.yValues) for plot in plotList]
		

	@classmethod
	def prepFuzzyData(cls, fuzzySet):
		if isinstance(fuzzySet, TriangularFuzzySets):
			plotSets = np.array([([interval[0], (interval[0]+interval[1])/float(2), interval[1]], [0, 1, 0]) for interval in fuzzySet.setIntervals])			
		elif isinstance(fuzzySet, TrapezoidFuzzySets):
			plotSets = np.array([(interval, [0, 1, 1, 0]) for interval in fuzzySet.setIntervals])
		elif isinstance(fuzzySet, GaussianFuzzySets):
			plotSets = np.array([(np.linspace(0, 250, 1000), [math.e**(-(1/float(2))*((value-interval[0])**2)/float(interval[1])) for value in np.linspace(0, 250, 1000)]) for interval in fuzzySet.setIntervals])

		plotSets[0][1][0] = 1
		plotSets[len(plotSets)-1][1][len(plotSets[len(plotSets)-1][1])-1] = 1

		return plotSets


	@classmethod
	def prepInferenceFuzzyData(cls, fuzzySet, inferenceValues):		
		if isinstance(fuzzySet, TriangularFuzzySets):
			plotSets = np.array([([interval[0], (interval[0]+interval[1])/float(2), interval[len(interval)-1]], [0, 1, 0]) for interval in fuzzySet.setIntervals])						
			inferenceSets = np.array([([fuzzySet.setIntervals[int(inference[0])][0], 
										fuzzySet.setIntervals[int(inference[0])][0] + inference[1]*((fuzzySet.setIntervals[int(inference[0])][0]+fuzzySet.setIntervals[int(inference[0])][1])/float(2) - fuzzySet.setIntervals[int(inference[0])][0]), 
										fuzzySet.setIntervals[int(inference[0])][len(fuzzySet.setIntervals[int(inference[0])])-1] - inference[1]*((fuzzySet.setIntervals[int(inference[0])][0]+fuzzySet.setIntervals[int(inference[0])][1])/float(2) - fuzzySet.setIntervals[int(inference[0])][0]),
										fuzzySet.setIntervals[int(inference[0])][len(fuzzySet.setIntervals[int(inference[0])])-1]]
										,
										[0, inference[1], inference[1], 0]) for inference in inferenceValues])
		elif isinstance(fuzzySet, TrapezoidFuzzySets):
			plotSets = np.array([(interval, [0, 1, 1, 0]) for interval in fuzzySet.setIntervals])
			inferenceSets = np.array([([interval[0], (interval[0]+interval[1])/float(2), interval[len(interval)-1]], [0, inference[1], inference[1], 0]) for inference in inferenceValues])

		plotSets[0][1][0] = 1
		plotSets[len(plotSets)-1][1][len(plotSets[len(plotSets)-1][1])-1] = 1		

		return plotSets, inferenceSets


	def showPertinence(self, pertinenceValues, fuzzySets, value):
		plotSets = [Plot.prepFuzzyData(fuzzySet) for fuzzySet in fuzzySets]
		i = 0
		for plotSet in plotSets:
			plt.subplot(len(plotSets), 1, i+1)			
			[(plt.plot(plotSet[j][0], plotSet[j][1], "b-") if j in pertinenceValues[i,:,0] else plt.plot(plotSet[j][0], plotSet[j][1], "r-")) for j in range(len(plotSet))]						
			[plt.plot([value[i], value[i], plotSet[0][0][0]], [0, pertinence, pertinence], "g-") for pertinence in pertinenceValues[i,:,1]]
			plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])						
			i = i+1
		plt.show()


	def unitedInference(self, inferedInterval):
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


	def showInference(self, inferenceValues, fuzzyOutputSet):
		plotSet, inferenceSet = Plot.prepInferenceFuzzyData(fuzzyOutputSet[0], inferenceValues)						
				
		for plotInference in inferenceSet:
			[plt.plot(plot[0], plot[1], "b-") for plot in plotSet]			
			plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])		

			lines = plt.plot(plotInference[0], plotInference[1])
			plt.setp(lines, color='r', linewidth=2.0)
			plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])
			plt.show()

		plotSet, inferenceSet = Plot.prepInferenceFuzzyData(fuzzyOutputSet[0], self.unitedInference(inferenceValues))

		fig, ax = plt.subplots()
		patches = []
		N = 5

		[plt.plot(plot[0], plot[1], "b-") for plot in plotSet]			
		plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])		

		for plotInference in inferenceSet:						
		    polygon = Polygon(zip(plotInference[0], plotInference[1]), False)
		    patches.append(polygon)

		p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0, edgecolor="blue")		

		ax.add_collection(p)
		plt.show()

	
	def showFuzzySets(self, fuzzySets):
		plotSets = [Plot.prepFuzzyData(fuzzySet) for fuzzySet in fuzzySets]			
		i = 1	
		for plotSet in plotSets:
			plt.subplot(len(plotSets), 1, i)
			[plt.plot(plot[0], plot[1], "b-") for plot in plotSet]
			plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])						
			i = i+1
		plt.show()


	@staticmethod
	def showTrapezoids(fuzzySets, centroids):
		plotSets = [Plot.prepFuzzyData(fuzzySet) for fuzzySet in fuzzySets]			
		i = 1	
		for plotSet in plotSets:
			plt.subplot(len(plotSets), 1, i)
			[plt.plot(plot[0], plot[1], "b-") for plot in plotSet]
			plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])						
			i = i+1

		plt.plot(centroids[:,0], centroids[:,1], "ro")

		plt.show()		


	def showCentroid(self, centroid, inferenceValues, fuzzyOutputSet):		
		plotSet, inferenceSet = Plot.prepInferenceFuzzyData(fuzzyOutputSet[0], self.unitedInference(inferenceValues))

		fig, ax = plt.subplots()
		patches = []		

		[plt.plot(plot[0], plot[1], "b-") for plot in plotSet]			
		plt.axis([plotSet[0][0][0], plotSet[len(plotSet)-1][0][len(plotSet[len(plotSet)-1][0])-1], 0, 2])		

		for plotInference in inferenceSet:						
		    polygon = Polygon(zip(plotInference[0], plotInference[1]), False)
		    patches.append(polygon)

		p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0, edgecolor="blue")		

		ax.add_collection(p)		

		plt.plot([centroid, centroid], [0, 0.1], "g-")

		plt.show()


	def showRects(self, ruleMatrix, fuzzySets):
		plotSets = [Plot.prepFuzzyData(fuzzySet) for fuzzySet in fuzzySets]					

		x = np.linspace(math.floor(plotSets[0][0][0][0]), math.floor(plotSets[0][len(plotSets[0])-1][0][len(plotSets[0][len(plotSets[0])-1][0])-1]), 1000)		

		[plt.plot(x, rule[0]+rule[1]*x, "r-") for rule in ruleMatrix]
		plt.axis([math.floor(plotSets[0][0][0][0]), math.floor(plotSets[0][len(plotSets[0])-1][0][len(plotSets[0][len(plotSets[0])-1][0])-1]), 0, 250])						
		plt.show()


	

