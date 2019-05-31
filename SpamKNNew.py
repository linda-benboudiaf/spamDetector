import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix, f1_score, accuracy_score

datasets = pd.read_csv('/home/lbenboudiaf/Bureau/spamDetector/DataSets/spambase.csv')
#Shuffle the data
datasets = datasets.sample(frac=1)

## We igsnore the two collumns word_freq_george,word_freq_650 and replace it by Not Data Found NaN.
#b = re.findall('^[-+]?[0-9]{1,2}$', '0.2')
#print(b)
ignoredColumns = ['word_freq_george', 'word_freq_650']
for column in ignoredColumns:
        #datasets[column] = datasets[column].replace({column : 0.00},{column: np.NaN})
        datasets[column] = datasets[column].replace(range(1,35), 0)
        #datasets[column] = datasets[column].replace(np.arange(0.1,35.50, 0.1), np.NaN)
        #mean = float(datasets[column].mean(skipna=True))
        #datasets[column] = datasets[column].replace(np.NaN, mean)
        #datasets[column] = datasets[column].replace(np.NaN, 0)


# Split Data
X = datasets.iloc[:,0:57] #Data, don't worry it doesn't include the last colunm.
y = datasets.iloc[:,57] #Target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20)
# Feature Scaling -> Caractersitique scalaire
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train.astype(float))
X_test = sc_X.transform(X_test.astype(float))

# in order to determine the best value to 'K'
# in this case we have 2 classes so it is better ti have an odd number like 3, 7, 11 ...
import math
math.sqrt(len(y_test)) # it gives 30.34 so we take 29 as first best value to 'K'

#Define the Model: K-NN
# p=2 because we want to identifie weather the email is a spam or not.
# We take the euclidean distance between a given data point and the actual data point.
# EuclideanDistance = srqt(pow(x-xi,2) + pw(y-yi,2));
classifier = KNeighborsClassifier(n_neighbors= 23, p=2,metric= 'euclidean')
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)


accuracy = [] #We agregate the Accuracy averages for 18 neighbors.
f1_scores = [] #Metrics...
index = range(2, 23)
for i in index:
    classifier = KNeighborsClassifier(n_neighbors = i) #18 classifiers
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) # Predict the class labels for the provided data
    conf_matrix = confusion_matrix(y_test, y_pred) # What we predit <VS> what actually is on test data.
    res = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)) # Calculate Accuracy of our predit.
    accuracy.append(res)
    f1_scores.append(list(zip(y_test, y_pred)))
print(f1_scores)
print(accuracy)

# Evaluate the Model.
mc = confusion_matrix(y_test, y_pred)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
