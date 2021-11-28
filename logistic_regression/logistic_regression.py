"""
Logistic Regression Hypothesis:
hΘ = g(z) where z = weighted sum and g = sigmoid function
(also known as logistic function) = 1 / (1 + e^-z)

Logistic Regresion Cost Func:
J(Θ) = 1 / m Σ_i=1 to m Cost(hΘ(xi, yi)) where
Cost(hΘ(xi, yi)) = 
	-log(hΘ_xi) if y = 1
	-log(1 - hΘ_xi) if y = 0
or
J(Θ) = -1 / m [Σ_i=1 to m yi * log(hΘ_xi) + (1 - yi) * log(1 - hΘ_xi)]

The partial derivative term in GD stays the same:
ddΘ_j = 1/m Σ_i=1 to m ((h_Θ(xi) - y_i))xi_j

Learnings:
- The only real difference between linear regression and logistic
regression is that the definition of the hypothesis changed (the
weighted sum is now passed through sigmoid). Other than that,
everything else stays the same including GD.
- Don't forget to check for null columns in dataframes
- Cost should always be decreasing. If it's not, you may have forgotten
a negative sign.
- In training the outputted prediction value is continuous but in evaluation
that when you add the threshold.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

class LogisticRegression(object):
	"""docstring for LogisticRegression"""
	def __init__(self):
		print("Initialized logistic regression model")

	def fit(self, x, y, alpha, num_iterations):
		self.x = np.array(x)
		self.y = np.array(y).flatten()
		self.m = len(x) # num samples
		self.n = len(x.columns) # num features
		self.weights = np.zeros(self.n + 1)
		self.costs = []

		for i in tqdm(range(num_iterations)):
			y_predictions = self.predict(self.x)
			dtheta_0 = (1/self.m) * np.sum(y_predictions - self.y)
			self.weights[0] = self.weights[0] - alpha * dtheta_0
			for j in range(1, self.n + 1):
				dtheta_j = (1/self.m) * np.dot(y_predictions - self.y, self.x[:,j-1])
				self.weights[j] = self.weights[j] - alpha * dtheta_j
			# self.costs.append(self.getCost())

		# self.plotLossOverTime()

	def predict(self, x):
		y_predictions = []
		for row in x:
			y_predictions.append(self.sigmoid(self.weights[0] + np.dot(self.weights[1:], row)))
		return np.array(y_predictions)

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def getCost(self):
		predictions = self.predict(self.x)
		cost = (-1 / self.m) * np.sum(self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions))
		return cost

	def plotLossOverTime(self):
		plt.plot(self.costs)
		plt.show()

	def test(self, x_test, y_test):
		predictions = self.predict(np.array(x_test))
		discrete_predictions = [1 if x >= 0.5 else 0 for x in predictions]
		y_test = np.array(y_test).flatten()
		return getAccuracy(discrete_predictions, y_test)

def getAccuracy(y_predictions, y_test):
	num_correct = np.array([True if i == j else False for i,j in zip(y_predictions, y_test)]).sum()
	return num_correct / len(y_test)

def main():
	df = pd.read_csv("train.csv")
	df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)
	embarked_one_hot = pd.get_dummies(df['Embarked'])
	df = df.drop('Embarked',axis = 1)
	df = df.join(embarked_one_hot)
	sex_one_hot = pd.get_dummies(df['Sex'])
	df = df.drop('Sex',axis = 1)
	df = df.join(sex_one_hot)
	df.fillna(df.mean(), inplace=True)

	x = df.loc[:, df.columns != 'Survived']
	scaler = preprocessing.StandardScaler()
	x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)
	y = df[['Survived']]

	log_reg_classifier = LogisticRegression()
	log_reg_classifier.fit(x, y, 0.1, 500)

	test_df = pd.read_csv("test.csv")
	test_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], inplace=True, axis=1)
	test_embarked_one_hot = pd.get_dummies(test_df['Embarked'])
	test_df = test_df.drop('Embarked',axis = 1)
	test_df = test_df.join(test_embarked_one_hot)
	test_sex_one_hot = pd.get_dummies(test_df['Sex'])
	test_df = test_df.drop('Sex',axis = 1)
	test_df = test_df.join(test_sex_one_hot)
	test_df.fillna(test_df.mean(), inplace=True)

	x_test = test_df.loc[:, test_df.columns != 'Survived']
	test_scaler = preprocessing.StandardScaler()
	x_test = pd.DataFrame(test_scaler.fit_transform(x_test.values), columns=x_test.columns, index=x_test.index)
	y_test = pd.read_csv("gender_submission.csv")[['Survived']]

	test_accuracy = log_reg_classifier.test(x_test, y_test)
	print("Custom accuracy", test_accuracy)

	skl_log_reg_class = SKLogisticRegression(random_state=0).fit(x, np.array(y).flatten())
	y_predictions = skl_log_reg_class.predict(x_test)
	print("SKL Accuracy", getAccuracy(y_predictions, np.array(y_test).flatten()))

main()