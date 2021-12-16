import numpy as np

def binaryCrossEntroypy(predictions, Y):
	predictions = predictions.flatten()
	Y = Y.flatten()
	m = len(predictions)
	epsilon = 1e-5    
	return (-1 / m) * np.sum(Y * np.log(predictions + epsilon) + (1 - Y) * np.log(1 - predictions + epsilon))

loss_functions = {
	'binary_cross_entropy': binaryCrossEntroypy
}