"""
Neural Network with 1 Hidden Layer
Planar Dataset grabbed from Deep Learning Specialization Course 1, Week 3

Using Log loss since it's binary classification.
=> J(W1, b1, W2, b2) = 1/m Σ_i=1 to m L(y^, y) where L is log loss

GD with NNs:
Initialize weights randomly. Biases can be set to 0
for some number of iterations or until convergence:
	Compute y^
	Compute dW2, db2, dW1, db1
	W1 = W1 - α * dW1
	b1 = b1 - α * db1
	W2 = W2 - α * dW2
	b2 = b2 - α * db2

Learnings:
- np ":" means all => arr[0,:] = row 0 all columns
- use .shape to confirm the shape of an np array
- Logistic Regression only performs well on linearly separable data (remember
it is just the classification version of linear regression and just adds the
sigmoid function). It also makes sense if you think of it as a neural network
with no hidden layers then of course it would be too simple.
- Bias terms can be initalized to 0 but weights shouldn't be
- Vectorizing across samples just means increasing the dimensions of X, Z, and A
so that each column represents a sample. Once you know what the dimensions should
be, you can just check the shape to make sure the calculations are being performed
correctly.
- For binary classification we can use log regr. loss. Softmax is only needed
for multiclass classification.
- Just like in Linear Regression, the gradient update rule is the same. The only
difference is in how NNs go about calculating the derivatives.
- For any variable it's derivative should have the same dimension (Z & dZ, W & dW, etc.)
- The calculations for dA & dZ are usually just collapsed into one step so you only need
to calculate dZ).
- In backpropogation, you calculate 3 things for all layers: dZ, dW, and dB. The equations
for these calculations are the same across all layers except for dZ on the last layer
- Using the * operator or np.multiply does element-wise multiplication. Using np.matmul
does matrix multiplication.

TF Learnings
- When creating an input layer the first parameter is actually the num of features
(e.g. shape=(2,) for 2 features). This is a little weird since the rest of stuff has
the first param as the number of rows in the data
- When counting the num of trainable paramters don't forget that it counts the biases.
- You can get the weights and biases for a particular layer i with model.layers[i].get_weights()
"""

import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, plot_decision_boundary_custom, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.zeros((self.num_units, 1))
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

class ShallowNeuralNetwork(object):
	def __init__(self, num_hidden_units):
		self.num_hidden_units = num_hidden_units
		print("Initialized Shallow Neural Network")

	def fit(self, X, Y, alpha, num_iterations):
		print("Fitting")
		self.X = X
		self.Y = Y
		self.m = self.X.shape[1]
		self.learning_rate = alpha
		self.hidden_layer = Layer(self.num_hidden_units, self.X.shape[0])
		self.output_layer = Layer(1, self.num_hidden_units)
		self.costs = []
		for i in tqdm(range(num_iterations)):
			self.forwardProp()
			self.backProp()
		# self.plotCostOverTime()

	def forwardProp(self):
		self.hidden_layer.Z = \
			np.dot(self.hidden_layer.W, self.X) + self.hidden_layer.B
		self.hidden_layer.A = sigmoid(self.hidden_layer.Z)
		self.output_layer.Z = \
			np.dot(self.output_layer.W, self.hidden_layer.A) + self.output_layer.B
		self.output_layer.A = np.tanh(self.output_layer.Z)
		self.costs.append(self.getCost(self.output_layer.A, self.Y))

	def backProp(self):
		dZ_output_layer = self.output_layer.A - self.Y
		dW_output_layer = (1/self.m) * np.dot(dZ_output_layer, self.hidden_layer.A.T)
		db_output_layer = (1/self.m) * np.sum(dZ_output_layer, axis=1, keepdims=True)
		dZ_hidden_layer = \
			np.dot(self.output_layer.W.T, dZ_output_layer) * (self.hidden_layer.A * (1 - self.hidden_layer.A))
		dW_hidden_layer = (1/self.m) * np.dot(dZ_hidden_layer, self.X.T)
		db_hidden_layer = (1/self.m) * np.sum(dZ_hidden_layer, axis=1, keepdims=True)
		self.hidden_layer.W = self.hidden_layer.W - self.learning_rate * dW_hidden_layer
		self.hidden_layer.B = self.hidden_layer.B - self.learning_rate * db_hidden_layer
		self.output_layer.W = self.output_layer.W - self.learning_rate * dW_output_layer
		self.output_layer.B = self.output_layer.B - self.learning_rate * db_output_layer

	def getCost(self, predictions, Y):
		cost = (-1 / self.m) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
		return cost

	def plotCostOverTime(self):
		plt.plot(self.costs)
		plt.show()

	def predict(self, X):
		hidden_layer_Z = np.dot(self.hidden_layer.W, X) + self.hidden_layer.B
		hidden_layer_A = sigmoid(hidden_layer_Z)
		output_layer_Z = np.dot(self.output_layer.W, hidden_layer_A) + self.output_layer.B
		output_layer_A = np.tanh(output_layer_Z)
		return output_layer_A

	def test(self, X, Y):
		predictions = self.predict(X).flatten()
		print("Regular predictions", predictions)
		discrete_predictions = [1 if x >= 0.5 else 0 for x in predictions]
		# print("discrete_predictions",discrete_predictions)
		return getAccuracy(discrete_predictions, Y.flatten())

def getAccuracy(y_predictions, y_test):
	num_correct = np.array([True if i == j else False for i,j in zip(y_predictions, y_test)]).sum()
	return num_correct / len(y_test)


def main():
	X, Y = load_planar_dataset()
	# plt.scatter(X[:,0], X[:,1], c=Y)
	# plt.show()
	# fake = np.array([
	# 	[1,2,3,4],
	# 	[4,5,6,6]
	# ])
	# print(Y)
	# print("X: ", X.shape)
	# print("Y: ", Y.shape)

	# log_reg_classifier = LogisticRegression(random_state=0).fit(X, Y.flatten())
	# plot_decision_boundary(lambda x: log_reg_classifier.predict(x), X.T, Y.T)

	# shallow_nn = ShallowNeuralNetwork(4)
	# shallow_nn.fit(X.T, Y.T, 0.1, 10)
	# # print(X.T.shape)
	# first_sample = np.array([
	# 	[X.T[0][0]],
	# 	[X.T[1][0]]
	# 	])
	# # print("First sample", first_sample)
	# accuracy = shallow_nn.test(X.T,Y.T)
	# print("accuracy", accuracy)
	# shallow_nn_predictions = shallow_nn.predict(X.T)
	# shallow_nn_cost = shallow_nn.getCost(shallow_nn_predictions, Y.T)
	# print(shallow_nn_cost)
	# print(Y.T.shape)

	# plot_decision_boundary_custom(lambda x: shallow_nn.predict(x), X.T, Y.T)

	# Might need to change from tanh to sigmoid
	# TF appears to implement where each row represents a sample and each column represents a unit
	# Implement TF then go back and change implementation to use softmax (check that all the outputs are
	# actually 0 to 1)

	# Creating the model with 12 trainable parameters
	model = Sequential()
	# First parameter of shape is the number of features
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(1, activation='tanh'))
	model.summary()

main()
