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
- GD updates need to be simultaneous; you don't want to calculate different
hypotheses for the later weights since the earlier weights were updated (
this is not actually GD). You can ensure this by caching the hypothesis
calculations before updating.
- Use numpy arrays since they have built-in functionality like element-wise
operations between lists (e.g. you can do list1 * list2 on np arrays but not
on standard python lists). Additionally the np.zeros array uses int64 which
is smaller than int so it can avoid integer overflows.
- Confirm the size of matrices for debugging
- Normalizing doesn't just help your model converge faster but it also protects
agains exploding weights which could lead to NAN erros.

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
from sklearn import preprocessing

class LinearRegression(object):
	def __init__(self):
		print("Initialized linear regression model")

	def fit(self, x, y, alpha, num_iterations):
		self.x = np.array(x)
		self.y = np.array(y).flatten()
		self.m = len(x) # num samples
		self.n = len(x.columns) # num features
		self.weights = np.zeros(self.n + 1)

		caught = False
		for i in range(num_iterations):
			print("Iteration", i)
			y_predictions = self.predict(self.x)
			dtheta_0 = (1/self.m) * np.sum(y_predictions - self.y)
			self.weights[0] = self.weights[0] - alpha * dtheta_0
			for j in range(1, self.n + 1):
				dtheta_j = (1/self.m) * np.dot(y_predictions - self.y, self.x[:,j-1])
				# print("dtheta_j", dtheta_j)
				if math.isinf(dtheta_j) or math.isnan(dtheta_j):
					print("y_predictions - self.y", y_predictions - self.y)
					print("self.x[:,j-1]",self.x[:,j-1])
					print("dot",np.dot(y_predictions - self.y, self.x[:,j-1]))
					print("weights", self.weights)
					caught = True
				if caught:
					break
				self.weights[j] = self.weights[j] - alpha * dtheta_j
			if caught:
				break
		print("Finished fitting", self.weights)
		self.printError()
		# self.plotPredictions()

	def predict(self, x):
		y_predictions = []
		for row in x:
			y_predictions.append(self.weights[0] + np.dot(self.weights[1:], row))
		return np.array(y_predictions)

	def plotPredictions(self):
		y_predictions = self.predict(self.x)
		plt.plot(self.x, self.y, 'o')
		plt.plot(self.x, y_predictions)
		plt.show()

	def printError(self):
		total_squared_error = 0
		predictions = self.predict(self.x)
		total_squared_error = np.sum((predictions - self.y) ** 2)
		total_error = math.sqrt(total_squared_error)
		avg_error = total_error / self.m
		print("Total Error", total_error)
		print("Avg Error", avg_error)

	def zscoreNormalize(self):
		pass


def main():
	# df = pd.read_csv("baseball_heights_and_weights.csv")
	# x = df["height"]
	# y = df["weight"]
	df = pd.read_csv("manhattan.csv")
	df.drop(['neighborhood', 'borough', 'rental_id'], inplace=True, axis=1)

	scaler = preprocessing.StandardScaler()
	# df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)

	x = df.loc[:, df.columns != 'rent']
	x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)
	y = df[['rent']]

	regr_model = LinearRegression()
	regr_model.fit(x, y, .01, 1000)

	# Trying to see if I get an interpretable output from a prediction
	# It doesn't appear to be. Trying not to scale the rent
	# In order to make sure it works I want to be able to see that the
	# predictions being made are atually making sense
	# I wonder if I just need to remove the rental id
	# Not really fix it
	# Maybe I actually just need to train for more iterations
	# The error is going down with more iterations. I can try increasing
	# the learning rate as well
	# 25.481178562268887
	# Seems to be working. I'll plot convergence somehow?

	first_row = np.array(x.iloc[[0]])
	prediction = regr_model.predict(first_row)
	print(prediction)

main()
