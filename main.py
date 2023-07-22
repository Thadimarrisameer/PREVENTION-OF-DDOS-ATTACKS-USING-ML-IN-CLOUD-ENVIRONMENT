import pandas as pd
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

df = pd.read_csv('DDoS Dataset.csv')
X = df.drop(columns=['Target'])
Y = df['Target']
print(df.head())

plt.figure(figsize=(13, 11))
sns.heatmap(df.corr(), annot=True)
plt.show()

# navie bayes
from sklearn.naive_bayes import GaussianNB

NB_model = GaussianNB()
X_train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=21)
NB_model.fit(X_train, Y_Train)
X_train_prediction = NB_model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_Train)
print("NB_Training accuracy: ", train_acc)

X_test_prediction = NB_model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("NB_Testing accuracy: ", test_acc)

# Knearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

sts = MinMaxScaler()
X_trs = sts.fit_transform(X)
KNN_model = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, Y_Train, Y_test = train_test_split(X_trs, Y, test_size=0.2, stratify=Y, random_state=21)
KNN_model.fit(X_train, Y_Train)
X_train_prediction = KNN_model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_Train)
print("\nKNN_Training accuracy: ", train_acc)

X_test_prediction = KNN_model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("KNN_Testing accuracy: ", test_acc)

# Logistic regression

from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression(random_state=0)
X_train, X_test, Y_Train, Y_test = train_test_split(X_trs, Y, test_size=0.2, stratify=Y, random_state=22)
log_classifier.fit(X_train, Y_Train)

logreg_X_train_prediction = log_classifier.predict(X_train)
train_acc = accuracy_score(logreg_X_train_prediction, Y_Train)
print("\nLog_regression_Training accuracy: ", train_acc)

logreg_X_test_prediction = log_classifier.predict(X_test)
test_acc = accuracy_score(logreg_X_test_prediction, Y_test)
print("Log_regression_Testing accuracy: ", test_acc)
