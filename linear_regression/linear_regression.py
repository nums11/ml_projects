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
- Normalizing the labels makes the results less interpretable so just normalize
features.
- Make sure you calculate RMSE properly. You have to divide by m before taking
the sqrt.
- The np functions can actually be slower than the standard python operations
(add, subtract, square, etc.) so just use the standard python operations

Result on manhattan.csv:
- LR of 0.5 converges after about 20 iterations with a root mean squared error
of 1353 as opposed to the sklearn model which has an RMSE of 1349

Possible Additions
- Compare to sklearn
- Implement Mini-Batch GD
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression as SKLinearRegression
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

class LinearRegression(object):
	def __init__(self):
		print("Initialized linear regression model")

	def fit(self, x, y, alpha, num_iterations):
		self.x = np.array(x)
		self.y = np.array(y).flatten()
		self.m = len(x) # num samples
		self.n = len(x.columns) # num features
		self.weights = np.zeros(self.n + 1)
		self.rmses = []

		for i in tqdm(range(num_iterations)):
			y_predictions = self.predict(self.x)
			dtheta_0 = (1/self.m) * np.sum(y_predictions - self.y)
			self.weights[0] = self.weights[0] - alpha * dtheta_0
			for j in range(1, self.n + 1):
				dtheta_j = (1/self.m) * np.dot(y_predictions - self.y, self.x[:,j-1])
				# print("dtheta_j", dtheta_j)
				self.weights[j] = self.weights[j] - alpha * dtheta_j
			self.rmses.append(self.getRMSE())

		print("RMSE", self.getRMSE())
		self.plotLossOverTime()
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

	def getRMSE(self):
		predictions = self.predict(self.x)
		return math.sqrt(((self.y - predictions) ** 2).mean())

	def plotLossOverTime(self):
		plt.plot(self.rmses)
		plt.show()


def main():
	# df = pd.read_csv("baseball_heights_and_weights.csv")
	# x = df["height"]
	# y = df["weight"]
	df = pd.read_csv("manhattan.csv")
	neighborhood_one_hot = pd.get_dummies(df['neighborhood'])
	df = df.drop('neighborhood',axis = 1)
	df = df.join(neighborhood_one_hot)
	df.drop(['borough', 'rental_id'], inplace=True, axis=1)

	x = df.loc[:, df.columns != 'rent']
	scaler = preprocessing.StandardScaler()
	x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)
	y = df[['rent']]

	regr_model = LinearRegression()
	regr_model.fit(x, y, .5, 20)

	sk_model = SKLinearRegression()
	sk_model.fit(x, y)
	y_predictions = sk_model.predict(x)
	print("RMSE sklearn", mean_squared_error(y, y_predictions, squared=False))

main()
