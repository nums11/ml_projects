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
- My model wasn't training and all I had to do was increase the number of iterations.
If it looks like loss is going down but it's still bad just increase the number of training
iterations lol.

TF Learnings
- When creating an input layer the first parameter is actually the num of features
(e.g. shape=(2,) for 2 features). This is a little weird since the rest of stuff has
the first param as the number of rows in the data
- When counting the num of trainable paramters don't forget that it counts the biases.
- You can get the weights and biases for a particular layer i with model.layers[i].get_weights()
- Binary Cross Entropy is the same as log loss
- An iteration is the processing of a batch but an epoch is the processing of the whole training
set. They are the same in batch GD but in mini-batch GD there are multiple iterations in
an epoch
- When I set the activation function to tanh sometimes it would never converge. I'm not sure
if this was because of the learning rate or tanh. But you are supposed to use sigmoid as
your activation function in the output layer since the binary cross entropy loss expects
values between 0 and 1.
- TF expects the data to be of the form m x n not n x m (each row is a sample not each column)

PyTorch Learnings
- In tf when you're creating your layer you also specify the activation
function. In pytorch, you specify the layer and activtion function
separately
- The standard way to create a NN is actually to implement a class where
you would implement the forward method. The sequential class just seems
to be a faster way
- You don't define the number of units in a layer but rather the input and output shape.
The number of output features is equivalent to the number of units. I don't like this as much.
- You have to convert things to the special torch tensor class instead of just using a
numpy array

Results:
- With TF model can get around 90% accuracy after 1000 epochs with a learning rate of 0.01
- With PT was able to get similar results (around 90%) with same learning rate but it took
around 2000 epochs. And this was with using ADAM.
- With my custom model I can get around 87% accuracy after 10000 epochs with a learning rate
of 0.1. I'm not sure why mine takes so many more epochs but I suspect that it's because I'm
not using any of the optimized version of GD (e.g. Adam). Still it ran through the epochs way
faster than TF and PyTorch. TF took 7.68 seconds to run through 1000 epochs (it was printing
way more though) and PyTorch took 105 seconds to run through 2000 epochs.
- The NN performs poorly on the other datasets though... I think it needs more layers.
"""

import numpy as np
import matplotlib.pyplot as plt
from planar_data_utils import plot_decision_boundary, plot_decision_boundary_custom, sigmoid, load_planar_dataset, load_extra_datasets
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

class Layer(object):
	def __init__(self, num_units, num_units_in_prev_layer):
		self.num_units = num_units
		self.W = np.random.randn(self.num_units, num_units_in_prev_layer) * 0.01
		self.B = np.random.randn(self.num_units, 1) * 0.01
		self.Z = np.zeros((self.num_units, 1))
		self.A = np.zeros((self.num_units, 1))

class ShallowNeuralNetwork(object):
	def __init__(self, num_hidden_units):
		self.num_hidden_units = num_hidden_units
		print("Initialized Shallow Neural Network")

	def fit(self, X, Y, alpha, epochs):
		self.X = X
		self.Y = Y
		self.m = self.X.shape[1]
		self.learning_rate = alpha
		self.hidden_layer = Layer(self.num_hidden_units, self.X.shape[0])
		self.output_layer = Layer(1, self.num_hidden_units)
		self.losses = []
		for epoch in tqdm(range(epochs)):
			self.forwardProp()
			self.backProp()
		# self.plotCostOverTime()

	def forwardProp(self):
		self.hidden_layer.Z = \
			np.dot(self.hidden_layer.W, self.X) + self.hidden_layer.B
		self.hidden_layer.A = sigmoid(self.hidden_layer.Z)
		self.output_layer.Z = \
			np.dot(self.output_layer.W, self.hidden_layer.A) + self.output_layer.B
		self.output_layer.A = sigmoid(self.output_layer.Z)
		self.losses.append(self.getBinaryCrossEntropyLoss(self.output_layer.A, self.Y))

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

	def getBinaryCrossEntropyLoss(self, predictions, Y):
		cost = (-1 / self.m) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
		return cost

	def plotCostOverTime(self):
		plt.plot(self.losses)
		plt.show()

	def predict(self, X):
		hidden_layer_Z = np.dot(self.hidden_layer.W, X) + self.hidden_layer.B
		hidden_layer_A = sigmoid(hidden_layer_Z)
		output_layer_Z = np.dot(self.output_layer.W, hidden_layer_A) + self.output_layer.B
		probabilities = sigmoid(output_layer_Z)
		predictions = np.rint(probabilities)
		return predictions

	def evaluate(self, X, Y):
		predictions = self.predict(X).flatten()
		return getAccuracy(predictions, Y.flatten())

def getAccuracy(y_predictions, y_test):
	num_correct = np.array([True if i == j else False for i,j in zip(y_predictions, y_test)]).sum()
	return num_correct / len(y_test)


def main():
	# X, Y = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = no_structure
	print(X.shape, Y.shape)
	shallow_nn = ShallowNeuralNetwork(4)
	shallow_nn.fit(X.T, Y.T, 0.1, 10000)
	accuracy = shallow_nn.evaluate(X.T, Y.T)
	print("Accuracy", accuracy)
	plot_decision_boundary_custom(lambda x: shallow_nn.predict(x), X.T, Y.T)

def trainTF():
	X, Y = load_planar_dataset()
	model = Sequential()
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=[tf.keras.metrics.BinaryAccuracy()]
	)
	training_history = model.fit(X, Y, epochs=1000)
	plt.plot(training_history.history["loss"])
	plt.show()
	# print(model.evaluate(X, Y, return_dict=True)['binary_accuracy'])
	# plot_decision_boundary(lambda x: model.predict(x), X.T, Y.T)

def trainPyTorch():
	torch.set_default_dtype(torch.float64)
	X, Y = load_planar_dataset()
	X_tensor = torch.from_numpy(X).to(torch.float64)
	Y_tensor = torch.from_numpy(Y).to(torch.float64)

	dataset = TensorDataset(X_tensor, Y_tensor)
	training_data_loader = DataLoader(dataset, batch_size=X.shape[0], num_workers=2)

	model = nn.Sequential(
		nn.Linear(2, 4),
		nn.Sigmoid(),
		nn.Linear(4,1),
		nn.Sigmoid()
	)

	lossFunc = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	losses = []
	for epoch in tqdm(range(2000)):
		for i, data in enumerate(training_data_loader, 0):
			inputs, labels = data
			optimizer.zero_grad()

			predictions = model(inputs)
			loss = lossFunc(predictions, labels)
			# Computes the gradients (derivatives)
			loss.backward()
			# Backpropogates
			optimizer.step()
			losses.append(loss.item())

	plt.plot(losses)
	plt.show()

	probabilities = model(X_tensor)
	predictions = torch.round(probabilities)
	num_correct = (predictions == Y_tensor).float().sum()
	accuracy = 100 * num_correct / len(dataset)
	print(accuracy)

main()
# trainTF()
# trainPyTorch()
