import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_path = os.path.abspath('../spamDetector/ressource/spambase.data')
df = pd.read_csv(data_path)
spam = defaultdict(list)

spam["target"] = df.iloc[:, -1]

spam["dataset"] = df.iloc[:, :-1]

X = spam["dataset"].to_numpy()

y = spam["target"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


#print(y_pred)  # 0 correspond to Versicolor, 1 to Verginica and 2 to Setosa
#print("classification_report"+"\n")
#print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

error = []

#Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
#plt.show()
