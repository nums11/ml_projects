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
"""

import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression

class Node(object):
	def __init__(self):
		print("I'm a node")
		

class ShallowNeuralNetwork(object):
	def __init__(self, hidden_layer_size):
		self.hidden_layer = [Node() for _ in range(hidden_layer_size)]
		print("Initialized Shallow Neural Network")

	def fit(self, x, y, alpha, num_iterations):
		print("Fitting")

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

	nn = ShallowNeuralNetwork(10)


main()
