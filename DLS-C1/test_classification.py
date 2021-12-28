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
- Softmax implementation I grabbed online was wrong.
- You need to train your model to output a shape that is the same as the labels
- I still think TF is better... like what's the point of using an ML framework
if I have to implement my own version of accuracy? At that point I might as well
just use the custom model I built myself.
"""
import sys
sys.path.append('../')
from ml_projects.custom_neural_network.CustomNeuralNetwork import CustomNeuralNetwork
from planar_data_utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Get sparse to work
def testCustomModel():
	# planar = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	# print(Y)
	Y_one_hot = np.squeeze(np.eye(6)[Y.reshape(-1)])
	print(Y_one_hot)

	# nn = CustomNeuralNetwork("binary_cross_entropy")
	# nn.addInputLayer(2)
	# nn.addLayer(4, "sigmoid")
	# nn.addLayer(1, "sigmoid")

	nn = CustomNeuralNetwork("categorical_cross_entropy")
	nn.addInputLayer(2)
	nn.addLayer(4, "sigmoid")
	nn.addLayer(6, "softmax")
	# nn.summary()

	# loss = nn.fit(X, Y_one_hot, 0.01, 1000)
	# plt.plot(loss)
	# plt.show()
	# print("Accuracy", nn.evaluate(X, Y_one_hot, 'categorical_accuracy'))

	# nn = CustomNeuralNetwork("categorical_cross_entropy")
	# nn.addInputLayer(2)
	# nn.addLayer(10, "sigmoid")
	# nn.addLayer(10, "sigmoid")
	# nn.addLayer(10, "sigmoid")
	# nn.addLayer(2, "softmax")
	# nn.summary()

	# loss = nn.fit(X, Y, 0.01, 1000)
	# plt.plot(loss)
	# plt.show()
	# print("Accuracy", nn.evaluate(X, Y, 'sparse_categorical_accuracy'))
	# plot_decision_boundary(lambda x: nn.predict(x), X, Y)

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

def testPytorchModel():
	torch.set_default_dtype(torch.float64)

	# X, Y = load_planar_dataset()
	noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
	X, Y = blobs
	# Y = np.array([Y]).T
	# print(Y)

	X_tensor = torch.from_numpy(X).to(torch.float64)
	Y_tensor = nn.functional.one_hot(torch.from_numpy(Y)).to(torch.float64)

	dataset = TensorDataset(X_tensor, Y_tensor)
	training_data_loader = DataLoader(dataset, batch_size=X.shape[0], num_workers=2)

	model = nn.Sequential(
		nn.Linear(2, 4),
		nn.Sigmoid(),
		nn.Linear(4, 6),
		nn.Softmax()
	)

	lossFunc = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	losses = []
	for epoch in tqdm(range(1000)):
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

	probabilities = model(X_tensor).detach().numpy()
	# Binary Accuracy
	# predictions = torch.round(probabilities)
	# num_correct = (predictions == Y_tensor).float().sum()
	# accuracy = 100 * num_correct / len(dataset)
	# print(accuracy)

	# 
	# probabilities = np.argmax(probabilities, axis=1)
	# num_correct = np.array([True if i == j else False for i,j in zip(probabilities, Y)]).sum()
	# accuracy = num_correct / len(Y)
	# print("accuracy", accuracy)

testCustomModel()
# testTFModel()
# testPytorchModel()