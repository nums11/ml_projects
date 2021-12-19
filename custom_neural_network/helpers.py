import numpy as np

def oneHot(arr):
	return np.squeeze(np.eye(6)[arr.reshape(-1)])