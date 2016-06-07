import numpy as np
import random as rd

class GASelection:

	def setParams(self, genAlgObj):
		self.genAlgObj = genAlgObj

class GASelectionTournament(GASelection):

	def __chooseParentForTournament(self, fitnessVector):
		candidatesIndexes = rd.sample(range(0,len(fitnessVector)), 3) 			
		fitnessVector = np.array(fitnessVector)
		tournamentCandidates = fitnessVector[np.array(candidatesIndexes)]		
		return self.genAlgObj.population[candidatesIndexes[np.argmin(tournamentCandidates)]]

	def run(self):
		fitnessVector = [self.genAlgObj.fitness([x,y]) for [x,y] in self.genAlgObj.population]				
		parentsVector = np.array([( self.__chooseParentForTournament(fitnessVector), self.__chooseParentForTournament(fitnessVector)) for i in range(len(fitnessVector)/2)])
		return parentsVector
	