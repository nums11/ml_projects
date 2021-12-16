from planar_data_utils import *
from CustomNeuralNetwork import CustomNeuralNetwork
import matplotlib.pyplot as plt

def testCustomModel():
	nn = CustomNeuralNetwork("binary_cross_entropy")
	nn.addInputLayer(2)
	nn.addLayer(5, "relu")
	nn.addLayer(4, "relu")
	nn.addLayer(1, "sigmoid")
	nn.summary()

	X, Y = load_planar_dataset()
	loss = nn.fit(X, Y, 0.0001, 100000)
	plt.plot(loss)
	plt.show()
	print(nn.evaluate(X, Y, 'binary_accuracy'))

testCustomModel()