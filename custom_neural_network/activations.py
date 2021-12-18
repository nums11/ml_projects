import numpy as np

def relu(x):
	return np.maximum(x, 0)

def relu_derivative(relu_x):
	return np.greater(relu_x, 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
	return sigmoid_x * (1 - sigmoid_x)

def softmax(x):
  s = np.max(x, axis=1)
  s = s[:, np.newaxis] # necessary step to do broadcasting
  e_x = np.exp(x - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis] # dito
  return e_x / div
  # e_x = np.exp(x - np.max(x))
  # return e_x / e_x.sum(axis=0) # only difference

activations = {
	'relu': relu,
	'sigmoid': sigmoid,
	'softmax': softmax
}

activation_derivatives = {
	'relu': relu_derivative,
	'sigmoid': sigmoid_derivative
}