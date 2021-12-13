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
"""

import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.zeros((self.num_units, 1))
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

		print("Initialized Layer")

class ShallowNeuralNetwork(object):
	def __init__(self, num_hidden_units):
		self.num_hidden_units = num_hidden_units
		print("Initialized Shallow Neural Network")

	def fit(self, X, Y, alpha, num_iterations):
		print("Fitting")
		self.X = X
		self.Y = Y
		self.m = self.X.shape[1]
		# print(self.Y)
		# print(self.X.shape[0])
		self.hidden_layer = Layer(self.num_hidden_units, self.X.shape[0])
		self.output_layer = Layer(1, self.num_hidden_units)
		self.forwardProp()
		self.backProp()

	def forwardProp(self):
		self.hidden_layer.Z = \
			np.dot(self.hidden_layer.W, self.X) + self.hidden_layer.B
		# print("Z for hidden layer", self.hidden_layer.Z)
		self.hidden_layer.A = sigmoid(self.hidden_layer.Z)
		# print("A for hidden layer", self.hidden_layer.A)
		self.output_layer.Z = \
			np.dot(self.output_layer.W, self.hidden_layer.A) + self.output_layer.B
		# print("Z for output_layer", self.output_layer.Z)
		self.output_layer.A = np.tanh(self.output_layer.Z)
		# print("A for output layer", self.output_layer.A)

	def backProp(self):
		dZ_output_layer = self.output_layer.A - self.Y
		# print(self.output_layer.Z.shape)
		# print(dZ_output_layer.shape)
		dW_output_layer = (1/self.m) * np.dot(dZ_output_layer, self.hidden_layer.A.T)
		# print(self.output_layer.W.shape)
		# print(dW_output_layer.shape)
		db_output_layer = (1/self.m) * np.sum(dZ_output_layer, axis=1, keepdims=True)
		# print(self.output_layer.B.shape)
		# print(db_output_layer.shape)
		dZ_hidden_layer = \
			np.dot(self.output_layer.W.T, dZ_output_layer) * (self.hidden_layer.A * (1 - self.hidden_layer.A))
		print(self.hidden_layer.Z.shape)
		print(dZ_hidden_layer.shape)

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

	nn = ShallowNeuralNetwork(4)
	nn.fit(X.T, Y.T, 0.1, 100)



main()
