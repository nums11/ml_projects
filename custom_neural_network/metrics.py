import numpy as np

def Accuracy(y_predictions, y_test):
	num_correct = np.array([True if i == j else False for i,j in zip(y_predictions, y_test)]).sum()
	return num_correct / len(y_test)

metrics = {
	'accuracy': Accuracy
}