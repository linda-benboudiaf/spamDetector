import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import classification_report, confusion_matrix, f1_score

data = np.loadtxt('/home/lbenboudiaf/Bureau/DataScience/Projet/Data/spambase.csv', delimiter=",")
X = data[:,0:57] #Data
y = data [:,57] #Target

# We extermine the George and word_freq_650

#We take 20% for testing and 80% as a Training Data.
#We can make Trainning Dat more important in order to get more reliable Accuracy.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.020)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit_transform(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_test)
#classifier = KNeighborsClassifier(n_neighbors = 4)
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
#print(confusion_matrix(y_test, y_pred))  #Print Confusion Matrix
#print(classification_report(y_test, y_pred))


accuracy = [] #We agregate the Accuracy averages for 18 neighbors.
f1_scores = [] #Metrics...
index = range(2, 20)
for i in index:
    classifier = KNeighborsClassifier(n_neighbors = i) #18 classifiers
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) # Predict the class labels for the provided data
    conf_matrix = confusion_matrix(y_test, y_pred) # What we predit <VS> what actually is on test data.
    res = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)) # Calculate Accuracy of our predit.
    accuracy.append(res)
    #accuracy.append((conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)))
    #print(y_test)
    #print(y_pred)
    #f1_scores(zip(y_test, y_pred))
    f1_scores.append(list(zip(y_test, y_pred)))
print(accuracy)
print(y_pred)
exit()
from math import log
#testList = [(x, log(y)) for x, y in f1_scores]
#print(index)
i, j = zip(*f1_scores)
plt.plot(i,j)
exit()

x, y = np.array(f1_scores).T
print(x, y)
fig, ax = plt.subplot(2,2)
ax.plot(x, y, 'ro')
ax.plot(x, y, 'b-')
ax.set_yscale('log')
fig.show()
print(len(index))
print(len(f1_scores))
plt.scatter(index, f1_scores)
plt.figure(figsize=(12, 6))
plt.plot(range(2, 20), range(2,20), color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Metrics')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
#plt.plot(index, f1_scores)
plt.show()
