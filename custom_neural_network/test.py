"""
Results:
- For some reason the custom model with the exact same config as the shallow network
is able to fit all the other datasets with actually pretty good accuracy
- Results can differ across runs.
- relu as a hidden layer activation function appears to be performing consistently
worse than sigmoid in both my custom model and tensorflow. Why is this the case

ToDo:
- Test on blobs (mutli-class) with softmax
- Test with PT
"""

from planar_data_utils import *
from CustomNeuralNetwork import CustomNeuralNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense

def testCustomModel():
	# planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs

	nn = CustomNeuralNetwork("sparse_categorical_cross_entropy")
	nn.addInputLayer(2)
	nn.addLayer(4, "sigmoid")
	nn.addLayer(1, "softmax")
	# nn.summary()

	loss = nn.fit(X, Y, 0.01, 1)
	plt.plot(loss)
	plt.show()
	print("Accuracy", nn.evaluate(X, Y, 'accuracy'))
	plot_decision_boundary(lambda x: nn.predict(x), X.T, Y.T)

def testTFModel():
	# planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = noisy_moons
	model = Sequential()
	model.add(Input(shape=(2,)))
	model.add(Dense(4, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=[tf.keras.metrics.BinaryAccuracy()]
	)
	# model.summary()
	training_history = model.fit(X, Y, epochs=2000)
	plt.plot(training_history.history["loss"])
	plt.show()
	print(model.evaluate(X, Y, return_dict=True)['binary_accuracy'])
	plot_decision_boundary(lambda x: model.predict(x), X.T, Y.T)

testCustomModel()
# testTFModel()