import numpy as np

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, layer):
	image = samples[:,:,0]
	print("Convolving")

	n = image.shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([layer.num_filters, output_shape, output_shape])
	for row in range(output_shape):
		for col in range(output_shape):
			# one step of the convolution vectorized across filters
			image_slice = image[row:row+f, col:col+f]
			element_wise_mult = image_slice * layer.W
			# sums the inner 2d matrices of the 3d element-wise result
			inner_sum = np.sum(np.sum(element_wise_mult, axis=1), axis=1)
			outputs[:, row, col] = inner_sum

	layer.Z = outputs + layer.B
	print(layer.Z, layer.Z.shape)

	return 0