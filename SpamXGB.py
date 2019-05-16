# Authors Linda Benboudiaf

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score

data = np.loadtxt('/home/lbenboudiaf/Bureau/spamDetector/DataSets/spambase.csv', delimiter=",")
X = data[:,0:56] #Data
y = data [:,57] #Target or classes.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

dtrain = xgb.DMatrix(X_train, label= y_train)
dtest = xgb.DMatrix(X_test, label= y_test)

dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

#--------------------  XGBoost Works with Parameters  ------------------------
param = {
    'max_depth': 6,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2 }# the number of classes that exist in this datset
num_round = 20  # the number of training iterations

#-------------  numpy array  ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
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
print("Svm file precision:",precision_score(y_test, best_preds_svm, average='macro'))
# ------------------ End svm File --------------------------

# dump the models
bst.dump_model('dump.raw.txt')
bst_svm.dump_model('dump_svm.raw.txt')

# save the models for later
joblib.dump(bst, 'bst_model.pkl', compress=True)
joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)