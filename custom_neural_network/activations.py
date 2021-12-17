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
	return np.exp(x - np.max(x)) / sum(np.exp(x - np.max(x)))


activations = {
	'relu': relu,
	'sigmoid': sigmoid,
	'softmax': softmax
}

activation_derivatives = {
	'relu': relu_derivative,
	'sigmoid': sigmoid_derivative
}