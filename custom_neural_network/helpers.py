import numpy as np
import time

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, layer):
	# image = samples[:,:,0]
	image = np.array([
		[1,2,3,4,5],
		[6,7,8,9,10],
		[11,12,13,14,15],
		[16,17,18,19,20],
		[21,22,23,24,25]
	])

	image2 = image * 2
	samples = np.array([image, image2])
	m = len(samples)
	print(samples, samples.shape)


	# print("image")
	# print(image)
	# print("weights")
	# print(layer.W)
	# print("biases")
	# print(layer.B)

	n = samples[0].shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([m, layer.num_filters, output_shape, output_shape])
	for row in range(output_shape):
		for col in range(output_shape):
			# # one step of the convolution vectorized across filters
			# image_slice = image[row:row+f, col:col+f]
			# element_wise_mult = image_slice * layer.W
			# # sums the inner 2d matrices of the 3d element-wise result
			# inner_sum = np.sum(np.sum(element_wise_mult, axis=1), axis=1)
			# outputs[:, row, col] = inner_sum

			# one step of the convolution vectorized across filters
			image_slices = samples[:, row:row+f, col:col+f]
			print("Image slices")
			print(image_slices, image_slices.shape)
			print("Layer W")
			print(layer.W, layer.W.shape)
			assert(image_slices.shape == (m, f, f))

			total_mult = np.array([slice * layer.W for slice in image_slices])
			inner_sum = np.sum(np.sum(total_mult, axis=2), axis=2)

			print(inner_sum, inner_sum.shape)
			outputs[:,:,row,col] = inner_sum
			print(outputs)

	print("outputs")
	print(outputs, outputs.shape)
	# Z = outputs + layer.B
	# print("Z")
	# print(Z.shape)

	"""
	I just need to make sure that my layer.Z is of a certain shape and thus my layer.A
	is also of that shape since these are the only things that are vectorized across samples.
	Working backward from this I can figure out what my X shape will need to be.
	"""

	# return Z
	return 0