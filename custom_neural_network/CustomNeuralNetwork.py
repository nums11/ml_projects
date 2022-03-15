"""
Custom Neural Network with the ability to add N layers
"""

import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from ml_projects.custom_neural_network.activations import activations, activation_derivatives
from ml_projects.custom_neural_network.loss_functions import loss_functions
from ml_projects.custom_neural_network.metrics import metrics
from ml_projects.custom_neural_network.helpers import *
from ml_projects.custom_neural_network.layers import *

def sigmoid(x):
  """
  Compute the sigmoid of x
  Arguments:
  x -- A scalar or numpy array of any size.
  Return:
  s -- sigmoid(x)
  """
  s = 1/(1+np.exp(-x))
  return s

class CustomNeuralNetwork(object):
	def __init__(self, loss_func):
		self.layers = []
		self.input_shape = None
		self.loss_func = loss_functions[loss_func]

	def addInputLayer(self, input_shape):
		num_input_dimensions = len(input_shape)
		if num_input_dimensions == 1 or num_input_dimensions == 3:
			self.input_shape = (None,) + input_shape
		else:
			raise Exception("Input shape must be either 1 or 3 dimensional. Got shape", input_shape,
				"with", num_input_dimensions, "dimensions")

	def add(self, layer):
		prev_layer_output_shape = None
		# if this will be the first layer
		if len(self.layers) == 0:
			prev_layer_output_shape = self.input_shape
		else:
			prev_layer_output_shape = self.layers[-1].output_shape

		if isinstance(layer, Dense):
			layer.initWeightMatrix(prev_layer_output_shape)
		layer.calcOutputShape(prev_layer_output_shape)
		self.layers.append(layer)
	
	def summary(self):
		print("Custom Neural Network")
		if len(self.layers) == 0:
			print("No layers")
		else:
			table = []
			for i, layer in enumerate(self.layers):
				layer_type = None
				if isinstance(layer, Dense):
					layer_type = "Dense"
					num_params = layer.W.shape[0] * layer.W.shape[1] + layer.B.shape[0]
				elif isinstance(layer, Flatten):
					layer_type = "Flatten"
					num_params = 0
				else:
					layer_type = "Conv2D"
					num_params = layer.num_filters * layer.filter_size * layer.filter_size + layer.num_filters

				output_shape = layer.output_shape
				table.append([layer_type, output_shape, num_params])
			print(tabulate(table, headers=['Layer Type', 'Output Shape', '# Params'], tablefmt='grid'))

	def fit(self, X, Y, alpha, epochs):
		self.X = X
		if self.loss_func == loss_functions["sparse_categorical_cross_entropy"]:
			Y = oneHot(Y)
		self.Y = Y
		self.alpha = alpha
		self.m = len(self.X)

		self.losses = []
		for epoch in tqdm(range(epochs)):
			self.forwardProp()
			# self.backProp()
			# self.updateWeights()
		return self.losses

	def forwardProp(self, custom_X=None):
		X = self.X if custom_X is None else custom_X
		for i, layer in enumerate(self.layers):
			if i == 0:
				A_prev_layer = X
			else:
				A_prev_layer = self.layers[i-1].A

			if isinstance(layer, Dense):
				layer.Z = np.dot(A_prev_layer, layer.W.T) + layer.B.T
				layer.A =  layer.activation_func(layer.Z)
				assert(layer.Z.shape == (self.m, layer.num_units))
				assert(layer.Z.shape == layer.A.shape)
			elif isinstance(layer, Flatten):
				print("Before flatten prev layer shape", A_prev_layer.shape)
				layer.A = A_prev_layer.reshape(A_prev_layer.shape[0], -1)
				print("Flatten layer shape", layer.A.shape)
			else:
				# Convolutional layer
				layer.Z = convolve(A_prev_layer, layer)
				# layer.Z = convolve2(A_prev_layer, layer)
				# layer.Z = convolve3(A_prev_layer, layer)
				layer.A = layer.activation_func(layer.Z)
				# print("shape", layer.A.shape)
				# print("A")
				# print(layer.A)

		# predictions = self.layers[-1].A
		# if custom_X is None:
		# 	self.losses.append(self.loss_func(predictions, self.Y))
		# return predictions

	def backProp(self):
		# skip over the flatten layers
		for i, layer in reversed(list(enumerate(self.layers))):
			if i == len(self.layers) - 1:
				layer.dZ = layer.A - self.Y
			else:
				layer.dZ = \
					np.dot(self.layers[i+1].dZ, self.layers[i+1].W) * layer.activation_derivative(layer.A)

			if i == 0:
				layer.dW = (1/2) * np.dot(layer.dZ.T, self.X)
			else:
				layer.dW = (1/2) * np.dot(layer.dZ.T, self.layers[i-1].A)

			layer.dB = (1/2) * np.sum(layer.dZ.T, axis=1, keepdims=True)
			assert(layer.dZ.shape == layer.Z.shape)
			assert(layer.dW.shape == layer.W.shape)
			assert(layer.dB.shape == layer.B.shape)

	def updateWeights(self):
		for layer in self.layers:
			layer.W = layer.W - self.alpha * layer.dW
			layer.B = layer.B - self.alpha * layer.dB

	def predict(self, X):
		self.m = len(X)
		return self.forwardProp(custom_X=X)

	def evaluate(self, X, Y, metric):
		predictions = self.predict(X)
		return metrics[metric](predictions, Y)

	def printWeightsDebug(self):
		for i, layer in enumerate(self.layers):
			print("Layer " + str(i) + " weights")
			print(layer.W)