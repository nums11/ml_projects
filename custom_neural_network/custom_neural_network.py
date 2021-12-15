"""
Custom Neural Network with the ability to add N layers
"""

import numpy as np
from tabulate import tabulate

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

class CustomNeuralNetwork(object):
	def __init__(self):
		self.layers = []
		self.num_input_features = None

	def addInputLayer(self, num_features):
		self.num_input_features = num_features

	def addLayer(self, num_units):
		num_units_in_prev_layer = None
		if len(self.layers) > 0:
			num_units_in_prev_layer = self.layers[-1].num_units
		else:
			num_units_in_prev_layer = self.num_input_features
		self.layers.append(Layer(num_units, num_units_in_prev_layer))
	
	def summary(self):
		print("Custom Neural Network")
		print("----------------------------------------------------")
		if self.num_input_features == None:
			print("No layers")
		else:
			table = []
			for i, layer in enumerate(self.layers):
				num_params = layer.W.shape[0] * layer.W.shape[1] + layer.B.shape[0]
				table.append(['Hidden ' + str(i+1), layer.num_units, num_params])
			print(tabulate(table, headers=['Layers', '# Units', '# Params'], tablefmt='orgtbl'))
