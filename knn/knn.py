"""
KNN algo:
Fitting is simply storing the normalized data points.
Predicting takes in k
for each data point:
	calculate the euclidean distance from this new point
	to that data point
classify based on the labels of the k-nearest points

Learnings:
- KNN does not have a cost function in the sense that no
parameters are minimzed during training.
"""
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

class KNN(object):
	def __init__(self):
		print("Initialized KNN Model")
		
	def fit(self, x, y):
		self.x = np.array(x)
		self.y = np.array(y).flatten()

	def predict(self, unk_data_point, k):
		distances = []
		for i, data_point in enumerate(self.x):
			distances.append((self.getEuclidieanDistance(data_point, unk_data_point), i))
		distances.sort(key=lambda tup: tup[0])
		k_nearest_points = distances[:k]
		k_nearest_labels = [self.y[x[1]] for x in k_nearest_points]
		c = Counter(k_nearest_labels)
		return c.most_common()[0][0]

	def getEuclidieanDistance(self, p1, p2):
		return np.linalg.norm(p1 - p2)

	def test(self, x_test, y_test, k):
		num_correct = 0
		quality_diff = 0
		for i in tqdm(range(len(x_test))):
			prediction = self.predict(np.array(x_test.iloc[[i]]), k)
			actual = y_test.iloc[i][0]
			if prediction == actual:
				num_correct += 1
			else:
				quality_diff += abs(prediction - actual)
		accuracy = num_correct / len(x_test)
		avq_quality_diff = quality_diff / (len(x_test) - num_correct)
		return (accuracy, avq_quality_diff)

def main():
	df = pd.read_csv("winequality-red.csv")
	x = df.loc[:, df.columns != 'quality']
	scaler = preprocessing.StandardScaler()
	x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)
	y = df[['quality']]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	knn_classifier = KNN()
	knn_classifier.fit(x_train, y_train)

	k_test_values = list(range(3,12))
	accuracies = []
	avg_quality_diffs = []
	for k in k_test_values:
		print("Testing k =", k)
		accuracy, avq_quality_diff = knn_classifier.test(x_test, y_test, k)
		accuracies.append(accuracy)
		avg_quality_diffs.append(avq_quality_diff)

	plt.plot(k_test_values, accuracies)
	plt.show()


main()