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
	("Nearest Neighbors", KNeighborsClassifier(3)),
	("AdaBoost", AdaBoostClassifier()),
	("Naive Bayes", GaussianNB()),
	("Random Forest Classifier", RandomForestClassifier()),
]

def run():
	return run_algos(classifiers)

def run_algos(algo_list):
	confusion_matrixes = {}
	for (name, algo) in algo_list:
		algo.fit(main.X_train, main.y_train)
		output = algo.predict(main.X_test)

		accuracy = round(accuracy_score(main.y_test, output), 4)*100
		precision = round(precision_score(main.y_test, output), 4)*100
		recall = round(recall_score(main.y_test, output), 4)*100
		f1_score_ = round(f1_score(main.y_test, output), 4)*100
		auc = round(roc_auc_score(main.y_test, output), 4)*100

		print(name, ':  Accuracy - ', accuracy, ', Precision - ', precision, ', Recall - ', recall,
		', F1_score - ', f1_score_, ' AUC - ', auc)

		confusion_matrixes[name] = (confusion_matrix(main.y_test, output))
	return confusion_matrixes
