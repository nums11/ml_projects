"""
Testing the MNIST Handwritten Digits Dataset.
Images are of size 28x28 but flattened.
Num classes = 10 for each of the handwritten digits (0 - 9)
Training set: 60,0000 samples
Test set: 10,000 samples

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

ToDo:
- Test it out against KNN?
- Try on other image datasets not just MNIST?
- Create my own dataset and try it?
- Try it on letters and Japanese characters
"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten

mndata = MNIST('DLS-C4/data')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()
num_classes = 10

# Is it possible to convolve flattened images or should I unflaten?

def testTF():
	model = Sequential()
	model.add(Input(shape=(28,28,1)))
	model.add(Conv2D(1, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	model.summary()


	# model.add(Input(shape=(2,)))
	# model.add(Dense(4, activation='relu'))
	# model.add(Dense(6, activation='softmax'))
	# model.compile(
	# 	optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
	# 	loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	# 	metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	# )
	# model.summary()
	# training_history = model.fit(X, Y, epochs=1000)
	# plt.plot(training_history.history["loss"])
	# plt.show()
	# print(model.evaluate(X, Y, return_dict=True)['sparse_categorical_accuracy'])


testTF()