import numpy as np
import pandas as pd
from urllib.request import urlopen
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

dataURL =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = urlopen(dataURL)

data = np.loadtxt(data,delimiter=",")
data = np.delete(data,26,1) # deleting both '650' and 'george' columns
data = np.delete(data,27,1)

x = data[:,:48] # better results when ignoring char_freq_CHAR and capital letters 
y = data[:,-1]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.21, shuffle=True, random_state = 17)
MultiNB = MultinomialNB()
MultiNB.fit(x_train,y_train)
print(MultiNB)
y_expect = y_test
y_predict = MultiNB.predict(x_test)
print(accuracy_score(y_test,y_predict))
skplt.metrics.plot_confusion_matrix(y_test, y_predict, normalize=True)
plt.show()

lr = LogisticRegression(solver = 'newton-cg' , multi_class = 'multinomial')
lr = lr.fit(x_train, y_train)
y_probas = lr.predict_proba(x_test)
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()