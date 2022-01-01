import numpy as np
import time

def oneHot(arr):
	num_classes = len(set(arr))
	return np.squeeze(np.eye(num_classes)[arr.reshape(-1)])

def convolve(samples, layer):
	image = np.array([
		[1,2,3,4,5],
		[6,7,8,9,10],
		[11,12,13,14,15],
		[16,17,18,19,20],
		[21,22,23,24,25],
	])
	image2 = np.array(image*2)
	image3 = np.array(image*3)
	samples = np.array([image, image2, image3])
	# print(samples, samples.shape)


	m = len(samples)
	print("m",m)
	n = samples[0].shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([m, layer.num_filters, output_shape, output_shape])
	for row in range(output_shape):
		for col in range(output_shape):
			print("row", row, "col", col)
			# One step of the convolution vectorized across filters and samples
			# Grab the current fxf slice across all samples
			image_slices = samples[:, row:row+f, col:col+f]
			assert(image_slices.shape == (m, f, f))
			# Element-wise multiply the fxf slices across samples with all of the filters
			total_mult = np.array([slice * layer.W for slice in image_slices])
			assert(total_mult.shape == (m, layer.num_filters, f, f))
			# Sum the result to complete this step of the convolution across all filters and samples
			inner_sum = np.sum(np.sum(total_mult, axis=2), axis=2)
			assert(inner_sum.shape == (m, layer.num_filters))

			outputs[:,:,row,col] = inner_sum

	# print("outputs\n", outputs, outputs.shape)
	# print("B\n", layer.B, layer.B.shape)

	Z = outputs + layer.B
	# print("Z\n", Z, Z.shape)
	assert(Z.shape == (m, layer.num_filters, output_shape, output_shape))

	print("Done with conv")
	return Z
	# return 0



def convolve3(samples, layer):
	image = np.array([
		[1,2,3,4,5],
		[6,7,8,9,10],
		[11,12,13,14,15],
		[16,17,18,19,20],
		[21,22,23,24,25],
	])
	image2 = np.array(image*2)
	samples = np.array([image, image2])
	print("samples\n",samples, samples.shape)
	print("W\n", layer.W, layer.W.shape)

	m = len(samples)
	print("m",m)
	n = samples[0].shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	outputs = np.empty([m, layer.num_filters, output_shape, output_shape])
	for row in range(1):
		for col in range(1):
			print("row", row, "col", col)
			# One step of the convolution vectorized across filters and samples
			# Grab the current fxf slice across all samples
			current_slice_all_samples = samples[:, row:row+f, col:col+f]
			assert(current_slice_all_samples.shape == (m, f, f))

			correct_mult = np.array([slice_for_sample * layer.W for slice_for_sample in current_slice_all_samples])
			print("correct_mult\n", correct_mult, correct_mult.shape)

			# transposed = layer.W.T
			# print("layer.W.T\n", layer.W.T, layer.W.T.shape)

			new_mult = current_slice_all_samples * layer.W
			# new_mult = np.matmul(current_slice_all_samples, layer.W)

			print("new_mult\n", new_mult, new_mult.shape)

			# Element-wise multiply the fxf slices across samples with all of the filters
			# total_mult = np.array([slice * layer.W for slice in image_slices])
			# assert(total_mult.shape == (m, layer.num_filters, f, f))
			# # Sum the result to complete this step of the convolution across all filters and samples
			# inner_sum = np.sum(np.sum(total_mult, axis=2), axis=2)
			# assert(inner_sum.shape == (m, layer.num_filters))

			# outputs[:,:,row,col] = inner_sum

	# print("outputs\n", outputs, outputs.shape)
	# print("B\n", layer.B, layer.B.shape)

	# Z = outputs + layer.B
	# # print("Z\n", Z, Z.shape)
	# assert(Z.shape == (m, layer.num_filters, output_shape, output_shape))

	# print("Done with conv")
	# return Z
	return 0


def convolve2(samples, layer):
	image = np.array([
		[1,2,3,4,5],
		[6,7,8,9,10],
		[11,12,13,14,15],
		[16,17,18,19,20],
		[21,22,23,24,25],
	])
	image2 = np.array(image*2)
	samples = np.array([image, image2])
	# print(samples, samples.shape)

	m = len(samples)
	print("m",m)
	n = samples[0].shape[0]
	f = layer.filter_size
	output_shape = n - f + 1
	print("output_shape", output_shape)
	outputs = np.empty([m, layer.num_filters, output_shape, output_shape])
	all_slices_all_samples = []
	for row in range(output_shape):
		for col in range(output_shape):
			print("row", row, "col", col)
			# Grab the current fxf slice across all samples
			current_slice_all_samples = samples[:, row:row+f, col:col+f]
			all_slices_all_samples.append(current_slice_all_samples)
	all_slices_all_samples = np.array(all_slices_all_samples)
	print("all_slices_all_samples\n", all_slices_all_samples, all_slices_all_samples.shape)

	# num_slices = all_slices_all_samples.shape[0]
	# num_samples = all_slices_all_samples.shape[1]
	# total_mult = np.empty([m, layer.num_filters, f, f])
	# for slice in range(num_slices):
	# 	for sample in range(num_samples):
	# 		slice_for_sample = all_slices_all_samples[slice][sample]
	# 		mult = slice_for_sample * layer.W
	# 		total_mult[]
	# 		print("mult", mult, mult.shape)


	# Multiply all fxf slices across all samples with all of the filters
	# print("W shape", layer.W.shape)

	# assert(image_slices.shape == (m, f, f))
	# # Element-wise multiply the fxf slices across samples with all of the filters
	# total_mult = np.array([slice * layer.W for slice in image_slices])
	# assert(total_mult.shape == (m, layer.num_filters, f, f))
	# # Sum the result to complete this step of the convolution across all filters and samples
	# inner_sum = np.sum(np.sum(total_mult, axis=2), axis=2)
	# assert(inner_sum.shape == (m, layer.num_filters))

	# outputs[:,:,row,col] = inner_sum

	# # print("outputs\n", outputs, outputs.shape)
	# # print("B\n", layer.B, layer.B.shape)

	# Z = outputs + layer.B
	# # print("Z\n", Z, Z.shape)
	# assert(Z.shape == (m, layer.num_filters, output_shape, output_shape))

	# print("Done with conv")
	# return Z
	return 0

def flatten(matrix):
	return matrix.reshape(matrix.shape[0], -1)