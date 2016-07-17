import numpy as np
import math

class SignalActivation():

	def run(self, value):
		return 1 if value > 0 else 0



class SigmoidActivation():

	def run(self, value):
		return 1/float(1+np.exp(-value))

	def derivative(self, value):
		return value*(1-value)

