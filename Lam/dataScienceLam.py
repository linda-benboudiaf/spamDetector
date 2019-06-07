#import libs
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets
#import pour les graphes
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import feature_selection
from sklearn.metrics import accuracy_score 
import numpy as np

#import data from spambase
dataset = np.loadtxt(r"C:\Users\Nguye\OneDrive\Bureau\spambase.data", delimiter=",")

#check dataset
print (dataset)

#Prendre 0 a 56 colonnes dans les données pour les utiliser en dataset 
X = dataset[:,0:56]
np.delete(X, [27,28,31], axis=1)
#Enlever les colonnes 27 - 28 - 31 pour clear la data selon la doc

#utiliser derniere valeur data comme resultat
y = dataset[:, -1]

#train on data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#test naive bayes bernoulli algorithme 

import matplotlib.pyplot as plt
#executer 20 fois l'algorithme et faire la moyenne de l'accuracy sur les 20 résultats
accuracy = []
index = range(1, 20)
for i in index:
    BernNB = BernoulliNB(alpha = 0.5,binarize = 0.2)
    BernNB.fit(X_train, y_train)
    print(BernNB)
    y_expect = y_test
    y_pred = BernNB.predict(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    accuracy.append(accuracy_score(y_expect, y_pred))
print(np.mean(accuracy))
#Accuracy entre 90 et 92 % selon la test size
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_expect, y_pred, normalize=True)
plt.show()
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)
#Matrice de confusion : permet de savoir pour quel type de données il se trompe ou réussi bien (spam ou pas spam)
y_probas = lr.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()
#Lift courbe : permet de voir le gain de notre algo par rapport à un random.
rf = RandomForestClassifier()
skplt.estimators.plot_learning_curve(rf, X, y)
plt.show()
pca = PCA(random_state=1)
pca.fit(X)
skplt.decomposition.plot_pca_2d_projection(pca, X, y)
#PCA projection 2 d pour classer selon 2 critères d'importance + print le graphe
plt.show()
data = load_iris()
data.target[[1, 25, 50]]
list(data.target_names)
rf = RandomForestClassifier()
rf.fit(X, y)
plt.show()
selector = SelectKBest(f_classif, k=5)
selector.fit(X, y)
# Get columns to keep
cols = selector.get_support(indices=True)
# Create new dataframe with only desired columns, or overwrite existing
features_df_new = X[cols]
print(features_df_new)