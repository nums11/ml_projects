from planar_data_utils import *
from custom_neural_network import CustomNeuralNetwork

def testCustomModel():
	nn = CustomNeuralNetwork()
	nn.addInputLayer(5)
	nn.addLayer(5)
	nn.addLayer(4)
	nn.summary()

testCustomModel()