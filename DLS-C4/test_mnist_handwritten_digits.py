"""
Testing the MNIST Handwritten Digits Dataset.
Images are of size 28x28 but flattened.
Num classes = 10 for each of the handwritten digits (0 - 9)
Training set: 60,0000 samples
Test set: 10,000 samples

Results:
- Surprisingly, with such a simple conv architecture that just has 1 hidden layer that
does a convolution with 1 3x3 filter is able to reach around 86% accuracy after just 6
iterations. I'm also surprised how slow it trains. Maybe it's because there's so many samples?
- TF, 5 epochs: 92% accuracy
- TFLeNet, 5 epochs, lr 0.001: 94% accuracy

Learnings:
- The main difference between CNNs and FFNNs is that the calculation for Z is not W * X + b but
rather the convolution of W on X then adding b. And instead of having multiple units and a bias
term for each unit, you have multiplef filters and a bias term for each filter.
- Max pooling preserves the most important features and has no learnable parameters. Some people
only count layers as layers with weights so they don't count the pooling layer as an actual layer.
- There are 2 main advantages to CNNs
	1. Paramter Sharing: A feature detector (e.g. vertical edge detection) that is useful in one part
	of the image is probaby useful in another part of the image, so CNNs can get away with having way
	less parameters.
	2. Sparsity of connections: Each output value in the output matrix after a convolution was determined
	based on only a small region from the input. It's almost like if you had 2 layers but instead of being
	fully connected, they were only partially connected so a unit in layer 2 only had weights from some of
	the units in layer 1. This also decreases the number of parametrs.
- When passing images to TF, it expects a 4-d array (you have to specify the number of channels) so you
may have to reshape data. Additionally you don't need to add an input layer.
- Normalizing values can give slightly better accuracy

ToDo:
- Implement my own conv and pooling layers
- Try on other image datasets not just MNIST?
	- Fashion MNIST?
- Create my own dataset and try it (like write my own digits)?
- Try it on letters and Japanese characters
- Adding batch GD?
"""

import sys
sys.path.append('../')
from ml_projects.custom_neural_network.CustomNeuralNetwork import CustomNeuralNetwork
from ml_projects.custom_neural_network.layers import Dense as CustomDense
from ml_projects.custom_neural_network.layers import Conv2D as CustomConv2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D, Activation
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
num_classes = 10

def testCustomModel():
	nn = CustomNeuralNetwork("sparse_categorical_cross_entropy")
	nn.addInputLayer((28,28,1))
	nn.add(CustomConv2D(10, 3, "relu"))
	# nn.add(CustomDense(4, "sigmoid"))
	# nn.add(CustomDense(6, "softmax"))
	# nn.summary()
	nn.fit(X_train, Y_train, 0.01, 1)

def displayDataPoint(index):
	plt.imshow(X_train[index], cmap=plt.get_cmap('gray'))
	plt.show()

def testTF():
	X_train = X_train.reshape(-1, 28, 28, 1)
	X_train = tf.cast(X_train, tf.float64)
	X_test = X_test.reshape(-1, 28, 28, 1)
	X_test = tf.cast(X_test, tf.float64)

	model = Sequential()
	model.add(Conv2D(1, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	# model.summary()
	history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
	fig, axs = plt.subplots(2, 1, figsize=(15,15))
	axs[0].plot(history.history['loss'])
	axs[0].plot(history.history['val_loss'])
	axs[0].title.set_text('Training Loss vs Validation Loss')
	axs[0].legend(['Train', 'Val'])
	axs[1].plot(history.history['sparse_categorical_accuracy'])
	axs[1].plot(history.history['val_sparse_categorical_accuracy'])
	axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
	axs[1].legend(['Train', 'Val'])
	plt.show()
	print(model.evaluate(X_test, Y_test, return_dict=True)['sparse_categorical_accuracy'])

def testTFLeNet1(x_train, y_train, x_test, y_test):
	# Pad images to be 32 x 32 as per original LeNet
	X_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]]) / 255
	X_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]]) / 255
	X_train = tf.cast(tf.expand_dims(X_train, axis=3, name=None), tf.float64)
	X_test = tf.cast(tf.expand_dims(X_test, axis=3, name=None), tf.float64)

	model = Sequential()
	model.add(Input(shape=(32,32,1)))
	model.add(Conv2D(6, 5, activation='tanh'))
	model.add(AveragePooling2D(2))
	model.add(Activation('sigmoid'))
	model.add(Conv2D(16, 5, activation='tanh'))
	model.add(AveragePooling2D(2))
	model.add(Activation('sigmoid'))
	model.add(Conv2D(120, 5, activation='tanh'))
	model.add(Flatten())
	model.add(Dense(84, activation='tanh'))
	model.add(Dense(10, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	model.summary()
	# history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
	# fig, axs = plt.subplots(2, 1, figsize=(15,15))
	# axs[0].plot(history.history['loss'])
	# axs[0].plot(history.history['val_loss'])
	# axs[0].title.set_text('Training Loss vs Validation Loss')
	# axs[0].legend(['Train', 'Val'])
	# axs[1].plot(history.history['sparse_categorical_accuracy'])
	# axs[1].plot(history.history['val_sparse_categorical_accuracy'])
	# axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
	# axs[1].legend(['Train', 'Val'])
	# plt.show()
	# print(model.evaluate(X_test, Y_test, return_dict=True)['sparse_categorical_accuracy'])

# testTF()
# testTFLeNet1(X_train, Y_train, X_test, Y_test)
testCustomModel()
