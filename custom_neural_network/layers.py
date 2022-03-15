import numpy as np
from ml_projects.custom_neural_network.activations import activations, activation_derivatives

class Dense(object):
	def __init__(self, num_units, activation_func):
		self.num_units = num_units
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.activation_func = activations[activation_func]
		if not activation_func == "softmax":
			self.activation_derivative = activation_derivatives[activation_func]

	def initWeightMatrix(self, prev_layer_output_shape):
		self.W = np.random.randn(self.num_units, prev_layer_output_shape[1]) * 0.01

	def calcOutputShape(self, prev_layer_output_shape):
		self.output_shape = (None, self.num_units)
		print("Output shape for Dense", self.output_shape)

class Conv2D(object):
	def __init__(self, num_filters, filter_size, activation_func):
		self.num_filters = num_filters
		self.filter_size = filter_size
		# np.random.seed(0)
		self.W = np.random.randn(num_filters, filter_size, filter_size)
		self.B = np.random.randn(1, num_filters, 1, 1) * 0.01
		self.activation_func = activations[activation_func]

	def calcOutputShape(self, prev_layer_output_shape):
		n = prev_layer_output_shape[1]
		conv_output_shape = n - self.filter_size + 1
		self.output_shape = (None, self.num_filters, conv_output_shape, conv_output_shape)
		print("Output shape for Conv", self.output_shape)

class Flatten(object):
	def __init__(self):
		pass

	def calcOutputShape(self, prev_layer_output_shape):
		flatten_output_size = 1
		for dimension in range(len(prev_layer_output_shape)):
			if prev_layer_output_shape[dimension] != None:
				flatten_output_size = flatten_output_size * prev_layer_output_shape[dimension]
		self.output_shape = (None, flatten_output_size)


