import numpy as np
from ml_projects.custom_neural_network.activations import activations, activation_derivatives

class Dense(object):
	def __init__(self, num_units, activation_func):
		self.num_units = num_units
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))
		self.activation_func = activations[activation_func]
		if not activation_func == "softmax":
			self.activation_derivative = activation_derivatives[activation_func]

	def initWeightMatrix(self, num_input_features, layers):
		num_units_in_prev_layer = None
		if len(layers) > 0:
			num_units_in_prev_layer = layers[-1].num_units
		else:
			num_units_in_prev_layer = num_input_features
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01

class Conv2D(object):
	def __init__(self, num_filters, filter_size, activation_func):
		self.num_filters = num_filters
		self.filter_size = filter_size
		np.random.seed(0)
		self.W = np.random.randn(num_filters, filter_size, filter_size)
		print(self.W)
		print(self.W.shape)
		self.B = np.random.randn(num_filters, 1) * 0.01
		print(self.B)
		print(self.B.shape)
		self.activation_func = activations[activation_func]
