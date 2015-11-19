# -*- coding: utf-8 -*-
# Import libraries
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cross_validation import train_test_split

# データ読み込み
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns
# とりあえず、データわけちゃいましょう
features = student_data[student_data.columns[0:29]]
target = student_data[student_data.columns[30]]

# Exploring the Data
n_students = student_data.shape[0]
n_features = len(list(student_data.columns[:-1]))
# 一列の時は、列名指定が要らないんですね…
n_passed = len(list(target[target == 'yes']))
n_failed = len(list(target[target == 'no']))
grad_rate = float(n_passed) / (n_passed + n_failed)

print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

#Preparing the Data
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows
print y_all.head()  

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)

#targetのバイナリ化の為にいじいじ
from sklearn.preprocessing import label_binarize

def preprocess_target(Y):
    Y = label_binarize(Y, classes = ["no","yes"])
    Y = pd.DataFrame(Y)
    Y.columns = [target_col]

    return Y

y_all = preprocess_target(y_all)
print y_all.head()
# ここまで

print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size= num_test, random_state=0)
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])

# Note: If you need a validation set, extract it from within training data

# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fit model to training data
train_classifier(clf, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label= 1)

train_f1_score = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)

# Predict on test data
print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

#自分でちょっといじってます。
from sklearn.grid_search import GridSearchCV
print X_all.head(3)
print y_all.head(3)

X = np.asarray(X_all)
y = np.asarray(y_all)

parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
grsrch  = GridSearchCV(clf, parameters,scoring = 'f1', cv = 3)
grsrch.fit(X, y)
print grsrch.grid_scores_
clf =  grsrch.best_estimator_

#ここまで

train_predict(clf, X_train, y_train, X_test, y_test)
# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant

# TODO: Train and predict using two other models
# TODO: Fine-tune your model and report the best F1 score