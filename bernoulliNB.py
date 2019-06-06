import numpy as np
import pandas as pd
from urllib.request import urlopen
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.naive_bayes import BernoulliNB


from sklearn.ensemble import RandomForestClassifier

dataURL =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = urlopen(dataURL)

data = np.loadtxt(data,delimiter=",")
data = np.delete(data,26,1) # deleting both '650' and 'george' columns
data = np.delete(data,27,1)

x = data[:,:48] 
y = data[:,-1]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.21, random_state = 17)

BernoulliNB = BernoulliNB()
BernoulliNB.fit(x_train,y_train)
y_expect = y_test
y_predict = BernoulliNB.predict(x_test)
accuracy_score(y_expect,y_predict)
print(BernoulliNB)

print(accuracy_score(y_expect,y_predict))

skplt.metrics.plot_confusion_matrix(y_test, y_predict, normalize=True)
plt.show()