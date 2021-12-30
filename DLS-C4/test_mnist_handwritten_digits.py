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
- Understand element-wise multiplication (* & np.multiply) vs dot product, vs matrix multiplication.

ToDo:
	# Implement iteratively then vectorize all calculations at once, then vectorize across samples.
	# Then make sure it works for padding and strides
	# Then add other types of layers like max pooling and avg pooling
	# Then test it out.

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
	test_mat = np.array([
		[3,5,1,4,6],
		[2,9,7,8,1],
		[0,4,1,3,1],
		[5,6,0,4,2],
		[7,8,9,5,3]
	])


	# Get top left 3x3 slice of the 5x5 matrix
	# print(test_mat[0:3,0:3])
	# print(test_mat[1:4,1:4])
	# slices = [0,1,2,3]
	# print(test_mat[slices])


	# # Get all 6 3x3 slices of array
	# for row in range(3):
	# 	for col in range(3):
	# 		print(test_mat[row:row+3, col:col+3])

	test_2 = np.array([
		[3,5,1],
		[2,9,7],
		[0,4,1],
	])

	test_3 = np.array([
		[1.764, 0.4, 0.978],
		[2.240, 1.867, -0.977],
		[0.950, -0.151, -0.103]
	])

	one_through_nine  = np.array([
		[1,2,3],
		[4,5,6],
		[7,8,9]
	])

	random = np.array([
		[[1,2,3],
		 [4,5,6],
		 [7,8,9]],
		[[10,11,12],
		 [13,14,15],
		 [16,17,18]]
	])
	# reshaped = random.reshape(2,9)
	# print(reshaped, reshaped.shape)
	# # dot = np.vdot(one_through_nine.flatten(), reshaped)
	# # dot = np.dot(np.array([1,2,3]), np.array([4,5,6]))
	# # new_arr = np.tile(one_through_nine.flatten(), (2,1))
	# # print(new_arr, new_arr.shape)
	# # dot = new_arr * reshaped
	# # print(dot)
	# print(one_through_nine.flatten() * reshaped)
	# print(one_through_nine)
	# data = np.random.normal(size=(100,2,2,2))
	# indexes = np.array([np.arange(0,5), np.arange(1,6), np.arange(2,7)])
	# print(data[indexes])
	b = [[[2]], [[3]]]
	print(one_through_nine * random + b)


	# print(np.vdot(test_2, test_3))

	# print(np.vdot(test_2, test_3))

	# nn = CustomNeuralNetwork("sparse_categorical_cross_entropy")
	# nn.addInputLayer((28,28,1))
	# nn.add(CustomConv2D(2, 3, "relu"))
	# # nn.add(CustomDense(4, "sigmoid"))
	# # nn.add(CustomDense(6, "softmax"))
	# # nn.summary()
	# nn.fit(X_train, Y_train, 0.01, 1)

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
