"""
Neural Network with 1 Hidden Layer
Planar Dataset grabbed from Deep Learning Specialization Course 1, Week 3

Learnings:
- np ":" means all => arr[0,:] = row 0 all columns
- use .shape to confirm the shape of an np array
- Logistic Regression only performs well on linearly separable data (remember
it is just the classification version of linear regression and just adds the
sigmoid function). It also makes sense if you think of it as a neural network
with no hidden layers then of course it would be too simple.
- Bias terms can be initalized to 0 but weights shouldn't be
"""

import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		print("Weights", self.W)
		self.B = np.zeros((self.num_units, 1))
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

		print("Initialized Layer")

class ShallowNeuralNetwork(object):
	def __init__(self, num_hidden_units):
		self.num_hidden_units = num_hidden_units
		print("Initialized Shallow Neural Network")

	def fit(self, x, y, alpha, num_iterations):
		print("Fitting")
		self.x = x
		self.y = y
		self.hidden_layer = Layer(self.num_hidden_units, self.x.shape[1])
		self.output_layer = Layer(1, self.num_hidden_units)
		self.forwardProp(x)
		self.backProp()

	def forwardProp(self,x):
		features = x[0]
		print("Features", features)
		# newaxis converts 1-d array to 2-d array
		self.hidden_layer.Z = \
			np.dot(self.hidden_layer.W, features)[np.newaxis].T + self.hidden_layer.B
		print("Z for hidden layer", self.hidden_layer.Z)
		self.hidden_layer.A = sigmoid(self.hidden_layer.Z)
		print("A for hidden layer", self.hidden_layer.A)
		self.output_layer.Z = \
			np.dot(self.output_layer.W, self.hidden_layer.A).T + self.output_layer.B
		print("Z for output_layer", self.output_layer.Z)
		self.output_layer.A = np.tanh(self.output_layer.Z)
		print("A for output layer", self.output_layer.A)

	def backProp(self):

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
	nn.fit(X, Y, 0.1, 100)



main()
