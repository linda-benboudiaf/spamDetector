# Authors Linda Benboudiaf

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix, f1_score, accuracy_score

datasets = pd.read_csv('/home/lbenboudiaf/Bureau/spamDetector/KNN-XGBoost/DataSets/spambase.csv')
#Shuffle the data
datasets = datasets.sample(frac=1)

## We igsnore the two collumns word_freq_george,word_freq_650 and replace it by Not Data Found NaN.
datasets = datasets.drop(labels=['word_freq_george','word_freq_650'], axis=1)

# Split Data
X = datasets.iloc[:,0:55] #Data, don't worry it doesn't include the last colunm.
y = datasets.iloc[:,55] #Target
class_names = datasets.columns.values # Get headers names after deleting georges and 650 freq...
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20, shuffle=True) # we shuffle again

# Feature Scaling -> Caractersitique scalaire
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train.astype(float))
X_test = sc_X.transform(X_test.astype(float))

# in order to determine the best value to 'K'
# in this case we have 2 classes so it is better ti have an odd number like 3, 7, 11 ...
import math
print('Value for K Math.sqrt(len of X_train) -------> ',math.sqrt(len(X_train))) # it gives 30.34 so we take 29 as first best value to 'K'
#Define the Model: K-NN
# p=2 because we want to identifie weather the email is a spam or not.
# We take the euclidean distance between a given data point and the actual data point.
# EuclideanDistance = srqt(pow(x-xi,2) + pw(y-yi,2));
#classifier = KNeighborsClassifier(n_neighbors= 3, p=2,metric= 'euclidean', weights='distance')
#classifier.fit(X_train, y_train)

# Predict the test set results
#y_pred = classifier.predict(X_test)

print("Please wait for graph representation ....")

accuracy = [] #We agregate the Accuracy averages for 18 neighbors.
f1_scores = [] #Metrics...
index = range(3,61)
for i in index:
    classifier = KNeighborsClassifier(n_neighbors = i,metric= 'euclidean', weights='uniform', leaf_size= 30) #27 classifiers
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) # Predict the class labels for the provided data
    conf_matrix = confusion_matrix(y_test, y_pred) # What we predit <VS> what actually is on test data.
    res = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)) # Calculate Accuracy of our predit.
    accuracy.append(res)
    f1_scores.append(list(zip(y_test, y_pred)))

print('In the range of 3 to 29 we have this values of accuracy')
print(accuracy)

# Evaluate the Model.
print('We evaluate the Matrix of Confusion')
mc = confusion_matrix(y_test, y_pred)
print(mc)

# Graph representation

''''https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.figure.html'''
plt.figure(figsize=(10, 6), num='Knn Algorithm on SpamBase Dataset')
plt.plot(index, accuracy, color='green', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy ratio according to K values')
plt.xlabel('K Values')
plt.ylabel('Accuracy average')
plt.show()

#print(f1_score(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))

