# Authors Linda Benboudiaf

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from xgboost import plot_importance

datasets = pd.read_csv('/home/lbenboudiaf/Bureau/spamDetector/KNN-XGBoost/DataSets/spambase.csv')
#Shuffle the data
datasets = datasets.sample(frac=1)
datasets = datasets.drop(labels=['word_freq_george','word_freq_650'], axis=1)

# Split Data
X = datasets.iloc[:,0:55].values #Data, don't worry it doesn't include the last colunm.
y = datasets.iloc[:,55].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=True)

dtrain = xgb.DMatrix(X_train, label= y_train)
dtest = xgb.DMatrix(X_test, label= y_test)

dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

''' http://www.datacorner.fr/xgboost'''
#--------------------  XGBoost Works with Parameters  ------------------------
param = {
    'max_depth': 6,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2 }# the number of classes that exist in this dataset
num_round = 20  # the number of training iterations

#-------------  numpy array  ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
xgb.to_graphviz(bst, num_trees=2)
bst.dump_model('dump.raw.txt')

preds = bst.predict(dtest)

# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))

# -------------  svm file  ---------------------
# training and testing - svm file
bst_svm = xgb.train(param, dtrain_svm, num_round)
preds = bst.predict(dtest_svm)

# extracting most confident predictions
best_preds_svm = [np.argmax(line) for line in preds]
accuracy = precision_score(y_test, best_preds_svm, average='macro')
print("Svm file precision:",precision_score(y_test, best_preds_svm, average='macro'))
# ------------------ End svm File --------------------------

# dump the models
bst.dump_model('dump.raw.txt')
bst_svm.dump_model('dump_svm.raw.txt')

# save the models for later
joblib.dump(bst, 'bst_model.pkl', compress=True)
joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)

# Metrics
model = XGBClassifier()
model.fit(X,y)
xgb.plot_importance(model)
#xgb.to_graphviz(model, num_trees=2)
plt.show()


