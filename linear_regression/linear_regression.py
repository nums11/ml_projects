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
on standard python lists). Additionally the np.zeros array uses int64 which
is smaller than int so it can avoid integer overflows.

Stil to do
- do multiple linear regression
- Try on bigger datasets
- Implement Mini-Batch GD
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
		self.x = np.array(x)
		self.y = np.array(y)
		self.m = len(x) # num samples
		self.n = len(x.columns) # num features
		self.weights = np.zeros(self.n + 1)

		for i in range(num_iterations):
			print("Iteration", i)
			y_predictions = self.predict()
			self.weights[0] = self.weights[0] - alpha * ((1/self.m) * np.sum(y_predictions - self.y))
			for j in range(1, self.n + 1):
				self.weights[j] = self.weights[j] - alpha * ((1/self.m) * np.sum((y_predictions - self.y) * self.x[:,j-1]))
		print("Finished fitting", self.weights)
		self.printError()
		# self.plotPredictions()

	def predict(self):
		y_predictions = []
		for row in self.x:
			y_predictions.append(self.weights[0] + np.sum(self.weights[1:] * row))
		return np.array(y_predictions)

	def plotPredictions(self):
		y_predictions = self.predict()
		plt.plot(self.x, self.y, 'o')
		plt.plot(self.x, y_predictions)
		plt.show()

	def printError(self):
		total_squared_error = 0
		predictions = self.predict()
		total_squared_error = np.sum((predictions - self.y) ** 2)
		total_error = math.sqrt(total_squared_error)
		avg_error = total_error / self.m
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
	# print(x.dtypes)
	regr_model = LinearRegression()
	regr_model.fit(x, y, .0001, 20)

main()
