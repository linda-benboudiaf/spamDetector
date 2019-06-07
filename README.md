# spamDetector
The ultimate Spam Detector

## Description
Create a spam filter so good that it will predict whether a received email is a spam or not.

## The project
The first part of this work is to understand the data you will be given, pay attention to the description of the data
on the UCI website, your goal is to clean the data of every features that could lead to misfiting your algorithm, for
example the non-spam data comes from work related emails and often contains the word george and the area
code 650 but for a stronger and wider spam filter you will want to omit those cases.
It is also your job to understand every columns of the data (all the 57) omit the one that will not matter, because
those are real data they will contain gaps and missing values, so it’s also your job to clean those and present
them in the best shape for your algorithm.

---

For each of the algorithms you will implement you will compute metrics and statistics about their accuracy and
present them in your reports (I like nice graphs). You will implement several algorithms with sklearn and play with
the tunning (depending of the algorithms you chose)
Keep in mind that we search an accurate algorithm but also a fast one, so the final decision will have to be
supported by evidences and do not hesitate to search the internet for more metrics to use.

## Algorithms
1. Nearest neighbors
<img src="/screenshots/Nearest_neighbors.png" alt="drawing" width="400"/>

2. Naive Bayes
<img src="/screenshots/NaiveBayes.png" alt="drawing" width="400"/>

3. AdaBoost
<img src="/screenshots/AdaBoost.png" alt="drawing" width="400"/>

4. Random Forest Classifier
<img src="/screenshots/random.PNG" alt="drawing" width="400"/>

## Run
```
python algorithms.py
```
