"""
Custom Neural Network with the ability to add N layers
"""

import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from ml_projects.custom_neural_network.activations import activations, activation_derivatives
from ml_projects.custom_neural_network.loss_functions import loss_functions
from ml_projects.custom_neural_network.metrics import metrics
from ml_projects.custom_neural_network.helpers import oneHot

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

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer, activation_func):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))
		self.activation_func = activations[activation_func]
		if not activation_func == "softmax":
			self.activation_derivative = activation_derivatives[activation_func]

class CustomNeuralNetwork(object):
	def __init__(self, loss_func):
		self.layers = []
		self.num_input_features = None
		self.loss_func = loss_functions[loss_func]

	def addInputLayer(self, num_features):
		self.num_input_features = num_features

	def addLayer(self, num_units, activation_func):
		num_units_in_prev_layer = None
		if len(self.layers) > 0:
			num_units_in_prev_layer = self.layers[-1].num_units
		else:
			num_units_in_prev_layer = self.num_input_features
		self.layers.append(Layer(num_units, num_units_in_prev_layer, activation_func))
	
	def summary(self):
		print("Custom Neural Network")
		print("----------------------------------------------------------------------------")
		if self.num_input_features == None or len(self.layers) == 0:
			print("No layers")
		else:
			table = []
			for i, layer in enumerate(self.layers):
				num_params = layer.W.shape[0] * layer.W.shape[1] + layer.B.shape[0]
				layer_name = None
				if i == len(self.layers) - 1:
					layer_name = "Output"
				else:
					layer_name = "Hidden " + str(i+1)
				table.append([layer_name, layer.num_units, num_params, layer.activation_func])
			print(tabulate(table, headers=['Layers', '# Units', '# Params', 'Activation'], tablefmt='orgtbl'))
		print("-------------------------------------")

	def fit(self, X, Y, alpha, epochs):
		self.X = X.T
		if self.loss_func == loss_functions["sparse_categorical_cross_entropy"]:
			print("Going to one hot")
			Y = oneHot(Y)
		self.Y = Y.T
		self.alpha = alpha
		self.m = len(self.X)

		self.losses = []
		for epoch in tqdm(range(epochs)):
			self.forwardProp()
			self.backProp()
			self.updateWeights()
		return self.losses

	def forwardProp(self, custom_X=None):
		X = self.X if custom_X is None else custom_X
		for i, layer in enumerate(self.layers):
			if i == 0:
				A_prev_layer = X
			else:
				A_prev_layer = self.layers[i-1].A
			layer.Z = np.dot(layer.W, A_prev_layer) + layer.B
			layer.A =  layer.activation_func(layer.Z)
			assert(layer.Z.shape == (layer.W.shape[0], A_prev_layer.shape[1]))
			assert(layer.Z.shape == layer.A.shape)

		predictions = self.layers[-1].A
		if custom_X is None:
			self.losses.append(self.loss_func(predictions, self.Y))
		return predictions

	def backProp(self):
		for i, layer in reversed(list(enumerate(self.layers))):
			if i == len(self.layers) - 1:
				layer.dZ = layer.A - self.Y
			else:
				layer.dZ = \
					np.dot(self.layers[i+1].W.T, self.layers[i+1].dZ) * layer.activation_derivative(layer.A)

			if i == 0:
				layer.dW = (1/self.m) * np.dot(layer.dZ, self.X.T)
			else:
				layer.dW = (1/self.m) * np.dot(layer.dZ, self.layers[i-1].A.T)

			layer.dB = (1/self.m) * np.sum(layer.dZ, axis=1, keepdims=True)
			assert(layer.dZ.shape == layer.Z.shape)
			assert(layer.dW.shape == layer.W.shape)
			assert(layer.dB.shape == layer.B.shape)

	def updateWeights(self):
		for layer in self.layers:
			layer.W = layer.W - self.alpha * layer.dW
			layer.B = layer.B - self.alpha * layer.dB

	def predict(self, X):
		return self.forwardProp(custom_X=X.T)

	def evaluate(self, X, Y, metric):
		predictions = self.predict(X)
		return metrics[metric](predictions, Y)

	def printWeightsDebug(self):
		for i, layer in enumerate(self.layers):
			print("Layer " + str(i) + " weights")
			print(layer.W)