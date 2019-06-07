#!/bin/bash
cd $HOME/SpamDetector
echo '#####----------------------------------#####'
echo '### Start Running SpamDetector Project ###'
echo '## ------------------------------------##'

echo 'Start XGBoost Algorithm'
echo 'Generating Metrics ...'
sleep 5
python3 ./KNN-XGBoost/SpamXGB.py

echo 'Start Knn Algorithm'
sleep 5
python3 ./KNN-XGBoost/SpamKnn.py

echo 'Start Knn with LogisticRegression'
sleep 5
python3 ./KNN-XGBoost/SpamKnnLG.py
