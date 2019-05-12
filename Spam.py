import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = np.loadtxt('/home/lbenboudiaf/Bureau/spamDetector/DataSets/spambase.csv', delimiter=",")
X = data[:,0:56] #Data
y = data [:,57] #Target

#We take 20% for testing and 80% as a Training Data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit_transform(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#classifier = KNeighborsClassifier(n_neighbors = 4)
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
#print(confusion_matrix(y_test, y_pred))  #Print Confusion Matrix
#print(classification_report(y_test, y_pred))

accuracy = [] #We agregate the Accuracy averages 18.
f1_scores = []
index = range(2, 20)
for i in index:
    classifier = KNeighborsClassifier(n_neighbors = i) #18 classifiers
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) # Predict the class labels for the provided data
    conf_matrix = confusion_matrix(y_test, y_pred) # What we predit <VS> what actually is on test data.
    res = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)) # Calculate Accuracy of our predit.
    accuracy.append(res)
    #accuracy.append((conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)))
    #print(f1_scores(y_test, y_pred))
    #f1_scores.append(f1_scores(y_test, y_pred))

print(accuracy)
#plt.plot(index, f1_scores)
