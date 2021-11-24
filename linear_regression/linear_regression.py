"""
Linear Regression Cost function:
J(Θ_0, Θ_1) = 1/2m Σ_i=1 to m (h_Θ(x_i) - y_i)^2
where h_Θ = Θ_0 + Θ_1x1 + Θ_2x2 + ... + Θ_nxn

The cost function is the sum of the individul losses
across all samples where the loss is defined as the squared
difference between the hypotheses prediction and the actual label.

The partial derivative of J with respect to any weight j
also known as the slope of the loss curve in the direction of j:
ddΘ_j = 1/m Σ_i=1 to m ((h_Θ(xi) - y_i))xi_j

In Batch GD we look across all samples when updating the weight
but in mini-batch GD you would only look across b samples where b
is the batch size.

So the linear regression algo is:
initalize weights Θ_0,...,Θ_j = 0
for some number of iterations or until convergence
	Θ_j = Θ_j - α * ddΘ_j of J
	repeat for all j

Learnings:
- The choice of a correct learning rate is super important. I thought
.001 was small but this was actually too big and caused divergence where
I started seeing a bunch of nans in the partial derivative.
- Since there is no feature for Θ_0, xi_0 = 1 so the partial derivative
term for Θ_0 doesn't multiply by a feature at the end
- GD updates need to be simultaneous; you don't want to calculate differen
hypotheses for the later weights since the earlier weights were updated (
this is not actually GD). You can ensure this by caching the hypothesis
calculations before updating.
- Use numpy arrays since they have built-in functionality like element-wise
operations between lists (e.g. you can do list1 * list2 on np arrays but not
on standard python lists)

Stil to do
- do multiple linear regression
- Try on bigger datasets
- Implement Batch GD
- compare to sklearn
- 
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

class LinearRegression(object):
	"""docstring for LinearRegression"""
	def __init__(self):
		print("Initialized linear regression model")

	def fit(self, x, y, alpha, num_iterations):
		self.x = x
		self.y = np.array(y)
		self.m = len(x) # num samples
		self.n = len(self.x.columns) + 1 # num weights
		self.weights = [0] * self.n

		arr1 = np.array([10,20,30])
		arr2 = np.array([5,6,7])
		arr3 = [10,20,30]
		arr4 = [5,6,7]
		print(arr1 * arr2)

		# for i in range(num_iterations):
		# 	print("Iteration", i)
		# 	y_predictions = self.predict()
		# 	self.weights[0] = self.weights[0] - alpha * ((1/self.m) * np.sum(y_predictions - self.y))
		# 	for j in range(1, self.n):
		# 		self.weights[j] = self.weights[j] - alpha * ((1/self.m) * np.sum((y_predictions - y) * x))
		# 	# self.theta1 = self.theta1 - alpha * ((1/m) * np.sum((y_predictions - y) * x))
		# print("Finished fitting", self.weights)
		# self.printError(x,y)
		# self.plotPredictions(x,y)

	def predict(self):
		y_predictions = []

		for i, row in self.x.iterrows():
			# calculate weighted sum
			prediction = np.sum(self.elementWiseMultiply(
				self.weights[1:], row))
			# Add bias term
			prediction += self.weights[0]
			y_predictions.append(prediction)

		return y_predictions

	def elementWiseMultiply(self, lista, listb):
		return [a * b for a,b in zip(lista,listb)]

	def plotPredictions(self, x, y):
		y_predictions = self.predict(x)
		plt.plot(x, y, 'o')
		plt.plot(x, y_predictions)
		plt.show()

	def printError(self,x,y):
		total_squared_error = 0
		m = len(x)
		predictions = self.predict(x)
		total_squared_error = np.sum((predictions - y) ** 2)
		total_error = math.sqrt(total_squared_error)
		avg_error = total_error / m
		print("Total Error", total_error)
		print("Avg Error", avg_error)


def main():
	# df = pd.read_csv("baseball_heights_and_weights.csv")
	# x = df["height"]
	# y = df["weight"]
	df = pd.read_csv("manhattan.csv")
	df.drop(['neighborhood', 'borough'], inplace=True, axis=1)
	x = df.loc[:, df.columns != 'rent']
	y = df[['rent']]
	# print(len(x.columns))
	regr_model = LinearRegression()
	regr_model.fit(x, y, .0001, 100)

main()
