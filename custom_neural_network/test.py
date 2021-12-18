"""
Results:
- For some reason the custom model with the exact same config as the shallow network
is able to fit all the other datasets with actually pretty good accuracy
- Results can differ across runs.

Learnings:
- The number of output units should be equal to the number of classes for multi-class
classification
- Use tf categorical_accuracy or sparse_categorical_accuracy (depending on your Y shape)
for multi-class classification not tf accuracy
- Relu appears to converge a lot faster than sigmoid
- Make sure you are using either categorical or sparse_categorical correctly because
results will be incorrect otherwise.

ToDo:
- Fix custom model
- Test with PT
- Test remaining datasets
"""

from planar_data_utils import *
from CustomNeuralNetwork import CustomNeuralNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense

# Fix definition of the categorical_cross_entropy function but it might
# be the same for sparse
def testCustomModel():
	# planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	Y_one_hot = np.squeeze(np.eye(6)[Y.reshape(-1)])

	# nn = CustomNeuralNetwork("binary_cross_entropy")
	# nn.addInputLayer(2)
	# nn.addLayer(4, "sigmoid")
	# nn.addLayer(1, "sigmoid")

	nn = CustomNeuralNetwork("categorical_cross_entropy")
	nn.addInputLayer(2)
	nn.addLayer(4, "relu")
	nn.addLayer(6, "softmax")
	# nn.summary()

	loss = nn.fit(X, Y_one_hot, 0.01, 20)
	plt.plot(loss)
	plt.show()
	print("Accuracy", nn.evaluate(X, Y_one_hot, 'categorical_accuracy'))
	# nn.printWeightsDebug()
	# plot_decision_boundary(lambda x: nn.predict(x), X.T, Y.T)

# Figure out how to get sparse categorical cross entropy to work (I think I would need to change my output)
def testTFModel():
	# planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	Y_one_hot = tf.one_hot(Y, 6)

	# model = Sequential()
	# model.add(Input(shape=(2,)))
	# model.add(Dense(4, activation='sigmoid'))
	# model.add(Dense(1, activation='sigmoid'))
	# model.compile(
	# 	optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
	# 	loss=tf.keras.losses.BinaryCrossentropy(),
	# 	metrics=[tf.keras.metrics.BinaryAccuracy()]
	# )
	# training_history = model.fit(X, Y, epochs=1000)
	# plt.plot(training_history.history["loss"])
	# print(model.evaluate(X, Y, return_dict=True)['binary_accuracy'])

	model = Sequential()
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='relu'))
	model.add(Dense(6, activation='softmax'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	# model.summary()
	training_history = model.fit(X, Y, epochs=1000)
	plt.plot(training_history.history["loss"])
	plt.show()
	print(model.evaluate(X, Y, return_dict=True)['sparse_categorical_accuracy'])
	# plot_decision_boundary(lambda x: model.predict(x), X.T, Y.T)

testCustomModel()
# testTFModel()