import numpy as np

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, filters):
	mat = samples[0]
	filter_ = filters[0]
	print("Convolving")
	print("mat", mat)
	print("filter_", filter_)
	return 0