import numpy as np

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, layer):
	mat = samples[:,:,0]
	print("Convolving")
	# print("mat", mat, mat.shape)
	# print("filter_", filter_, filter_.shape)

	n = mat.shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([layer.num_filters, output_shape, output_shape])
	for row in range(output_shape):
		for col in range(output_shape):
			current_slice = mat[row:row+f, col:col+f]
			# output[row,col] = np.vdot(current_slice, filter_) + bias_
			calc = np.vdot(current_slice, layer.W)
			print("calc", calc, calc.shape)
			# outputs[:,]

	print("output", outputs, outputs.shape)


	"""
	I'm trying to vectorize one step of the convolution across all filters. The issue
	is that vdot flattens the whole weight matrix making it (18,1) but I really only want
	it to flatten the inner weight matrices so that the whole matrix is of shape (2, 9)
	then I would be able to do the dot product to make the calculation.

	If I can figure out how to flatten the inner weight matrices only so that the weight matrix
	shape becomes 2x9 then I should be able to do a dot product with the slice from the input
	and things should work.

	Lol it seems like all i had to use was the * operator because it takes the slice and multiplies
	it against each of the weight matrices

	So element-wise multiplication isn't so simple to work for the whole convolution. However, it is
	the second half of vectorizing all steps of the convolution. If I was able to grab all slices
	at once, then I could do a simple element-wise multiplication of the fiter across the matrix of
	all slices to get the result of all steps of the convolution for that filter
	"""

	# Then logic for convolving all filters at once
	# vdot flattens the whole weight matrix but I only want the individual filters
	# inside the matrix flattened

	return 0