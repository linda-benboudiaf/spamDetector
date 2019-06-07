import main

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

classifiers = [
	("Random Forest Classifier", RandomForestClassifier()),
]

def run():
	return run_algos(classifiers)

def run_algos(algo_list):
	print(algo_list)
	confusion_matrixes = {}
	for (name, algo) in algo_list:
		algo.fit(main.X_train, main.y_train)
		output = algo.predict(main.X_test)
		# accuracy = the sum of all True positives and True Negative to the total number of test instances.
		accuracy = round(accuracy_score(main.y_test, output), 4)*100
		# precision = the ratio of true positive to true and false positives.
		precision = round(precision_score(main.y_test, output), 4)*100
		# recall  = s the ratio of true positives to the number of true positive and false negatives
		recall = round(recall_score(main.y_test, output), 4)*100

		print(name, ':  Accuracy - ', accuracy, ', Precision - ', precision, ', Recall - ', recall)

		confusion_matrixes[name] = (confusion_matrix(main.y_test, output))
	return confusion_matrixes
